from app.model import get_detector_model
from app.train import train_local
from app.data import make_dataloaders
import torch
import flwr as fl
from pathlib import Path

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
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train_local(self.model, self.train_loader, self.device, epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        return 0.0, len(self.val_loader.dataset), {}

if __name__ == "__main__":
    data_path = Path(__file__).parent / "app" / "Test_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_detector_model()
    train_loader, val_loader = make_dataloaders(str(data_path), batch_size=16)

    client = FlowerClient(model, train_loader, val_loader, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8085", client=client)
