import flwr as fl

config = fl.server.ServerConfig(num_rounds=3)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=config
)