from typing import List

import numpy as np
import tensorflow as tf

from fedless.common.serialization import (
    ModelLoader,
    WeightsSerializer,
    StringSerializer,
)
from fedless.datasets.dataset_loaders import DatasetLoader


# noinspection PyMissingOrEmptyDocstring
class DatasetLoaderStub(DatasetLoader):
    def __init__(self, dataset: tf.data.Dataset):
        self.dataset = dataset

    def load(self) -> tf.data.Dataset:
        return self.dataset


# noinspection PyMissingOrEmptyDocstring
class ModelLoaderStub(ModelLoader):
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def load(self) -> tf.keras.Model:
        return self.model


# noinspection PyMissingOrEmptyDocstring
class WeightsSerializerStub(WeightsSerializer):
    def __init__(self, weights: List[np.ndarray], blob: bytes):
        self.weights = weights
        self.blob = blob

    def serialize(self, weights: List[np.ndarray]) -> bytes:
        return self.blob

    def deserialize(self, blob: bytes) -> List[np.ndarray]:
        return self.weights


# noinspection PyMissingOrEmptyDocstring
class StringSerializerStub(StringSerializer):
    def __init__(self, blob: bytes, string: str):
        self.blob = blob
        self.string = string

    def to_str(self, obj: bytes) -> str:
        return self.string

    def from_str(self, rep: str) -> bytes:
        return self.blob
