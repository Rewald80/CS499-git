import flwr as fl
import torch
from pathlib import Path
from app.model import get_detector_model

MODEL_PATH = Path(__file__).parent / "model.pt"

def start_server():
    # Setup your strategy without on_fit_end_fn
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
    )

    # Load initial global model
    global_model = get_detector_model()
    # start the server
    history = fl.server.start_server(
        server_address="127.0.0.1:8085",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )

    print(f"Saving global model to {MODEL_PATH}")

if __name__ == "__main__":
    start_server()
