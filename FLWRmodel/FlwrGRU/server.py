import flwr as fl

# Define a strategy with the required number of clients, here fed avg has been used
strategy = fl.server.strategy.FedAvg(
    min_available_clients=3,  # Minimum number of clients that need to be connected, 
    #must run three different instances of client in seperate terminals of any kind
)

# Starting the Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)

# import flwr as fl
# from flwr.common import Context
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg
# from client import client_fn  # Ensure this is correctly defined

# def server_fn(context: Context) -> ServerAppComponents:
#     strategy = FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=6,
#         min_evaluate_clients=6,
#         min_available_clients=6,
#     )
#     config = ServerConfig(num_rounds=6)
#     return ServerAppComponents(config=config, strategy=strategy)

# app = ServerApp(server_fn=server_fn)