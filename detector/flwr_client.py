import flwr as fl
import torch
import numpy as np
from app.models import get_detector_model
from app.train import train_local
from app.data import make_dataloaders
import argparse
import os

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        new_state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(new_state_dict, strict=False)

    def fit(self, parameters, config=None):
        # set global weights
        self.set_parameters(parameters)
        # local training (very small local epochs for demo)
        train_local_local(self.model, self.train_loader, self.device, epochs=1)
        # return updated params
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, acc = evaluate_local(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}

# utility local training/eval functions (minimal)
def train_local_local(model, train_loader, device, epochs=1):
    import torch.nn as nn
    from torch.optim import AdamW
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()

def evaluate_local(model, val_loader, device):
    import torch.nn as nn
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss += criterion(out,y).item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return loss/total, correct/total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="app/Test_images")
    args = parser.parse_args()
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_detector_model()
    train_loader, val_loader = make_dataloaders(args.data, batch_size=16)
    # Wrap client and start
    client = FlowerClient(model, train_loader, val_loader, device)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())