import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse

# ---------------------------------
# ARGUMENT PARSING
# ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="Dataset", help="Path to dataset folder")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
parser.add_argument("--deep", action="store_true", help="Use ResNet34 instead of ResNet18")
parser.add_argument("--full-finetune", action="store_true", help="Unfreeze full model")
args = parser.parse_args()

# ---------------------------------
# DEVICE
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Using device: {device}")

# ---------------------------------
# TRANSFORMS
# ---------------------------------
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


# ---------------------------------
# DATASET LOAD
# ---------------------------------
train_dir = os.path.join(args.data, "train")
val_dir = os.path.join(args.data, "val")

train_data = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=2)

print(f"\n📁 Train samples: {len(train_data)}")
print(f"📁 Val samples: {len(val_data)}")

# ---------------------------------
# MODEL
# ---------------------------------

def get_model():
    if args.deep:
        print("\n🧠 Using ResNet34")
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        print("\n🧠 Using ResNet18")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze model first
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    # Allow classifier head to train
    for param in model.fc.parameters():
        param.requires_grad = True

    if args.full_finetune:
        print("🔓 Full model fine-tuning enabled")
        for param in model.parameters():
            param.requires_grad = True

    return model


def unfreeze_last_block(model):
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True


model = get_model().to(device)

# ---------------------------------
# LOSS + OPTIMIZER + SCHEDULER
# ---------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)

# ---------------------------------
# TRAINING
# ---------------------------------
best_val_acc = 0.0

for epoch in range(args.epochs):
    print(f"\n📘 Epoch {epoch+1}/{args.epochs}")

    # Progressive unfreezing
    if epoch == 3 and not args.full_finetune:
        print("🔓 Unfreezing final ResNet block...")
        unfreeze_last_block(model)

    # ------- TRAIN -------
    model.train()
    train_loss = 0
    train_correct = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_loss += loss.item()

    train_acc = train_correct / len(train_data)
    train_loss /= len(train_loader)

    # ------- VALIDATE -------
    model.eval()
    val_loss = 0
    val_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
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

    print(f"\n📊 Results:")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model.pt")
        print("💾 Saved new best model → model.pt")

# ---------------------------------
# FINAL REPORT
# ---------------------------------
print("\n✅ Training complete!")

print("\n📋 Final Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))
