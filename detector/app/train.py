import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
from torch.amp import GradScaler, autocast

# ---------------------------------
# ARGUMENT PARSING
# ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="Dataset", help="Path to dataset folder")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
parser.add_argument("--full-finetune", action="store_true", help="Unfreeze full model")
args = parser.parse_args()

# ---------------------------------
# DEVICE
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ---------------------------------
# TRANSFORMS
# ---------------------------------
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# ---------------------------------
# DATASET LOAD
# ---------------------------------
train_dir = os.path.join(args.data, "Train")
val_dir = os.path.join(args.data, "Validation")

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

print("\nClasses:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=0)

print(f"\nTrain samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

# ---------------------------------
# MODEL
# ---------------------------------
def get_model():

    print("\nUsing ResNet18 (optimized)")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if args.full_finetune:
        print("Full model fine-tuning enabled")
        for param in model.parameters():
            param.requires_grad = True

    model = model.to(memory_format=torch.channels_last)

    return model


model = get_model().to(device)

# ---------------------------------
# LOSS + OPTIMIZER + SCHEDULER
# ---------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)

scaler = GradScaler("cuda")

# ---------------------------------
# TRAINING
# ---------------------------------
best_val_acc = 0.0

for epoch in range(args.epochs):
    print(f"\nEpoch {epoch+1}/{args.epochs}")

    # ------- TRAIN -------
    model.train()
    train_loss = 0
    train_correct = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_loss += loss.item()

    train_acc = train_correct / len(train_data)
    train_loss /= len(train_loader)

    # ------- VALIDATION -------
    model.eval()
    val_loss = 0
    val_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_data)
    val_loss /= len(val_loader)

    scheduler.step()

    print(f"\nResults:")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pt")
        print("Saved new best model → model.pt")

# ---------------------------------
# FINAL REPORT
# ---------------------------------
print("\nTraining complete!")
print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))
