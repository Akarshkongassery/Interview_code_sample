from client import client_fn
import flwr as fl

client = client_fn(1)
fl.client.start_client(server_address="127.0.0.1:8080", client=client)