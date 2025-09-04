import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models

def mlAdl(cid):
    # df = pd.read_csv(f"data/client{cid}.csv")
    
    xRaw = np.load(f"dataV2/client_{cid}_data.npy", allow_pickle=True)
    y = np.load(f"dataV2/client_{cid}_labels.npy", allow_pickle=True).flatten()
    
    
    # target_column = 'label_crosslayer'
    # features = df.select_dtypes(include=[np.number]).drop(columns=[target_column])
    # labels = df[target_column]
    
    # scaler = MinMaxScaler()
    # data_scaled = scaler.fit_transform(features)
    # data_scaled = scaler.fit_transform(xRaw)
    
    # sequence_length = 20
    # X, y = [], []
    # for i in range(len(data_scaled) - sequence_length):
    #     X.append(data_scaled[i:i+sequence_length])
    #     y.append(labels.iloc[i+sequence_length])
    # X = np.array(X)
    # y = np.array(y)

    samples, timesteps, features = xRaw.shape
    X = xRaw.reshape(-1, features)  # flatten time dimension for scaling
    X = MinMaxScaler().fit_transform(X)
    X = X.reshape(samples, timesteps, features)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    
    model = models.Sequential([
        layers.Input(shape=(X.shape[1], X.shape[2])),
        layers.GRU(128),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return x_train, y_train, x_test, y_test, model, class_weights
