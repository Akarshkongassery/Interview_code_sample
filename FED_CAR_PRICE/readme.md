Project 2: Federated Car Price Prediction

Overview
This project explores predicting car prices using machine learning and federated learning techniques.

The workflow includes:
- **Data preprocessing** with `pandas` and `scikit-learn`
- **Feature scaling and encoding** of numeric and categorical variables
- **Neural network model** implemented in PyTorch with embeddings for categorical features
- **Training and evaluation** using MSE loss
- **Federated learning simulation** with [Flower (flwr)](https://flower.dev), partitioning data across multiple clients and training collaboratively

Dataset
- Input: `car_price_dataset.csv`
- Target variable: `Price`
- Features include brand, model, fuel type, transmission type, and numeric attributes.

Requirements
Install dependencies:

pip install numpy pandas scikit-learn torch matplotlib flwr datasets
