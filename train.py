import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import classification_report
from app.models import get_detector_model
from app.data import make_dataloaders
import os

def train_local(data_dir: str, epochs: int = 5, batch_size: int = 32, checkpoint_path: str = "model.pt", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = get_detector_model()
    model = model.to(device)

    train_loader, val_loader = make_dataloaders(data_dir, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    print("Training complete.")
    # Print classification report on validation set
    print(classification_report(y_true, y_pred, target_names=['real','fake']))
    return checkpoint_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./dataset", help="training data root")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="model.pt")
    args = parser.parse_args()
    train_local(args.data, epochs=args.epochs, batch_size=args.batch, checkpoint_path=args.out)