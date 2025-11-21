import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse

# -----------------------------
# Model definition
# -----------------------------
def get_detector_model(pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # real vs fake
    return model

# -----------------------------
# Dataloaders with augmentation
# -----------------------------
def make_dataloaders(train_dir, val_dir, batch_size=32):
    IMG_SIZE = 224

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# -----------------------------
# Training function
# -----------------------------
def train_local(train_dir, val_dir, epochs=10, batch_size=16, checkpoint_path="model.pt", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"\n🚀 Using device: {device}\n")

    # Debug directory print
    train_dir = os.path.abspath(train_dir)
    val_dir = os.path.abspath(val_dir)

    print(f"📁 Using Train Directory: {train_dir}")
    print(f"📁 Using Val Directory:   {val_dir}\n")

    print("🔍 Checking folder structure...")
    print("Train:", os.listdir(train_dir))
    print("Validation:", os.listdir(val_dir))

    # Load model
    model = get_detector_model(pretrained=True).to(device)

    # Load data
    train_loader, val_loader = make_dataloaders(train_dir, val_dir, batch_size=batch_size)

    print("\n📚 Classes found:", train_loader.dataset.classes)
    print(f"🖼️ Training images:   {len(train_loader.dataset)}")
    print(f"🖼️ Validation images: {len(val_loader.dataset)}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"\n====== Epoch {epoch}/{epochs} ======")

        # ------------------------------
        # TRAINING
        # ------------------------------
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        train_prog = tqdm(train_loader, desc="Training", ncols=100)

        for inputs, labels in train_prog:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_prog.set_postfix(loss=loss.item())

        train_loss = running_loss / total
        train_acc = correct / total

        # ------------------------------
        # VALIDATION
        # ------------------------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        y_true, y_pred = [], []

        val_prog = tqdm(val_loader, desc="Validation", ncols=100)

        with torch.no_grad():
            for inputs, labels in val_prog:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

                val_prog.set_postfix(loss=loss.item())

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved new best model → {checkpoint_path}")

    print("\n🎉 Training complete!")
    print("📈 Final Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['fake', 'real']))

    return checkpoint_path

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="./Dataset/Train", help="Path to training data")
    parser.add_argument("--val",   default="./Dataset/Validation", help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--out",    default="model.pt")
    args = parser.parse_args()

    train_local(args.train, args.val, epochs=args.epochs, batch_size=args.batch, checkpoint_path=args.out)
