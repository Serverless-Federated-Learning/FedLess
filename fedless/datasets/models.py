import numpy as np
import requests
import tensorflow as tf
from requests import RequestException
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model

from fedless.datasets.dataset_loaders import DatasetNotLoadedError


def cnn_4layer_fc_model(n_classes, n1=128, n2=192, n3=256, n4=256, dropout_rate=0.2, input_shape=(28, 28)):
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

    y = Conv2D(filters=n2, kernel_size=(2, 2), strides=2, padding="same", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")(y)

    y = Conv2D(filters=n3, kernel_size=(2, 2), strides=2, padding="same", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="same")(y)

    y = Conv2D(filters=n4, kernel_size=(3, 3), strides=2, padding="same", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    # y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units=n_classes, activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_A


def cnn_3layer_fc_model(n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28)):
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

    y = Conv2D(filters=n2, kernel_size=(2, 2), strides=2, padding="valid", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(y)

    y = Conv2D(filters=n3, kernel_size=(3, 3), strides=2, padding="valid", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    # y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units=n_classes, activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_A


def cnn_2layer_fc_model(n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28)):
    model_A, x = None, None

    x = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape(input_shape)(x)
    y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)

    y = Conv2D(filters=n2, kernel_size=(3, 3), strides=2, padding="valid", activation=None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    # y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units=n_classes, activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)

    model_A = Model(inputs=x, outputs=y)

    model_A.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model_A


def lstm_2layer_fc_model(
    units: int = 256,
    vocab_size: int = 82,
    sequence_length: int = 80,
    embedding_size: int = 8,
):

    model = tf.keras.Sequential()
    tf.keras.layers.InputLayer((sequence_length, vocab_size))
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.InputLayer((sequence_length, vocab_size)))
    model.add(
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
        )
    )
    model.add(tf.keras.layers.LSTM(units, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units))
    model.add(tf.keras.layers.Dense(vocab_size))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fetch_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except RequestException as e:
        raise DatasetNotLoadedError(e) from e

    return response


def lstm_1layer_fc_model(
    units: int = 256,
    vocab_size: int = 82,
    sequence_length: int = 80,
    embedding_size: int = 8,
):

    model = tf.keras.Sequential()
    tf.keras.layers.InputLayer((sequence_length, vocab_size))
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.InputLayer((sequence_length, vocab_size)))
    model.add(
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
        )
    )
    model.add(tf.keras.layers.LSTM(units))
    model.add(tf.keras.layers.Dense(vocab_size))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def remove_last_layer(model, loss="mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters
    """

    new_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss, metrics=["accuracy"])

    return new_model
