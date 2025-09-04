import flwr as fl
from MLaDL import mlAdl


def client_fn(num):

    X_train, y_train, X_test, y_test, model, class_weights = mlAdl(num)
    
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()
    
        def fit(self, parameters, config):
            model.set_weights(parameters)
            history = model.fit(
                X_train, y_train,
                epochs=15,
                batch_size=64,
                validation_split=0.2,
                class_weight=class_weights,
                verbose=1
            )
            return model.get_weights(), len(X_train), {}
    
        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
            return loss, len(X_test), {"accuracy": accuracy}
    
    client = FlowerClient()
    return client