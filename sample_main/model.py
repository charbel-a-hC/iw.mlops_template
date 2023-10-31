import keras
from typing import List
from keras import layers

def create_model(dense_layers: List[int]):

    inputs = keras.Input(shape=(784,), name="digits")
    
    for idx, layer in enumerate(dense_layers):
        if idx == 0:
            x = layers.Dense(layer, activation="relu")(inputs)
        else:
            x = layers.Dense(layer, activation="relu")(x)
    
    outputs = layers.Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model