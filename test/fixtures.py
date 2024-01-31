import numpy as np
import pytest
import tensorflow as tf
from _pytest.fixtures import fixture
from tensorflow.python.data import Dataset

from fedless.common.models import Hyperparams
from fedless.common.serialization import Base64StringConverter, NpzWeightsSerializer
from .stubs import (
    DatasetLoaderStub,
    ModelLoaderStub,
)

SAMPLES = 10
FEATURE_DIM = 5
CLASSES = 3


@pytest.fixture
def simple_model() -> tf.keras.Model:
    """Used in test_serialize.py"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(5, input_shape=(3,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Softmax(),
        ]
    )
    model.compile(loss="mse", optimizer="sgd")
    return model


@pytest.fixture
def dataset():
    features = np.random.randn(SAMPLES, FEATURE_DIM)
    labels = np.random.randint(low=0, high=CLASSES, size=SAMPLES)
    return Dataset.from_tensor_slices((features, labels))


@fixture
def data_loader(dataset):
    return DatasetLoaderStub(dataset)


@fixture
def model_loader(model):
    return ModelLoaderStub(model)


@pytest.fixture
def model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(FEATURE_DIM, input_shape=(FEATURE_DIM,)),
            tf.keras.layers.Dense(CLASSES, activation="softmax"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
    return model


@pytest.fixture
def weights_serializer():
    return NpzWeightsSerializer(compressed=False)


@pytest.fixture
def string_serializer():
    return Base64StringConverter()


@fixture
def hyperparams():
    # noinspection PyTypeChecker
    return Hyperparams(batch_size=1, epochs=2)
