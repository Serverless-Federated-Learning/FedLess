from typing import Optional, Dict

import numpy as np
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError
import tensorflow as tf
from typing import Dict, List, Optional


# import requests
# import tempfile
# import os
from fedless.common.cache import cache

from pydantic import BaseModel, Field


class MNISTConfig(BaseModel):
    """Configuration parameters for sharded MNIST dataset"""

    type: str = Field("mnist", const=True)
    indices: List[int] = None
    split: str = "train"
    proxies: Optional[Dict] = None
    location: str = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    )


class MNIST(DatasetLoader):
    def __init__(
        self,
        indices: Optional[List[int]] = None,
        split: str = "train",
        proxies: Optional[Dict] = None,
        location: str = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    ):
        self.split = split
        self.indices = indices
        self.proxies = proxies or {}
        self.location = location

    @cache
    def load(self) -> tf.data.Dataset:
        # response = requests.get(
        #     self.location,
        #     proxies=self.proxies,
        # )
        # TODO remove for deploy
        # fp, path = tempfile.mkstemp(prefix="mnist",dir="/home/ubuntu/mnist_temp")
        # with os.fdopen(fp, "wb") as f:
        #     f.write(response.content)
        tx_file_path = tf.keras.utils.get_file(
            cache_subdir="data", origin=self.location
        )

        with np.load(tx_file_path, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        if self.split.lower() == "train":
            features, labels = x_train, y_train
        elif self.split.lower() == "test":
            features, labels = x_test, y_test
        else:
            raise DatasetNotLoadedError(f"Mnist split {self.split} does not exist")

        if self.indices:
            features, labels = features[self.indices], labels[self.indices]

        def _scale_features(features, label):
            return tf.cast(features, tf.float32) / 255.0, tf.cast(label, tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels))

        return ds.map(_scale_features)
