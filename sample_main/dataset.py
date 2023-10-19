import keras
import numpy as np
import tensorflow as tf

def get_dataset(batch_size: int, train_samples: int):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))

    # Reserve 10,000 samples for validation.
    x_val = x_train[-train_samples:]
    y_val = y_train[-train_samples:]
    x_train = x_train[:-train_samples]
    y_train = y_train[:-train_samples]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset