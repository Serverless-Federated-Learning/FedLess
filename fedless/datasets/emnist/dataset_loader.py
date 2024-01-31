from enum import Enum
from typing import Dict, List, Optional

import emnist
import numpy as np
import scipy.io as sio
import tensorflow as tf
from pydantic import BaseModel, Field

from fedless.common.cache import cache
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError


class EMNISTConfig(BaseModel):

    type: str = Field("emnist", const=True)
    indices: List[int] = None
    split: str = "train"
    label_mapping: Optional[dict]
    proxies: Optional[Dict] = None
    dataset_url: Optional[str] = None


class EMNIST(DatasetLoader):
    def __init__(
        self,
        label_mapping: dict = None,
        indices: Optional[List[int]] = None,
        split: str = "train",
        proxies: Optional[Dict] = None,
        dataset_url: Optional[str] = None,
    ):
        self.split = split
        self.indices = indices
        self.proxies = proxies or {}
        self.dataset_url = dataset_url

        if label_mapping:
            self.label_mapping = self.convert_mapping(label_mapping)
        else:
            self.label_mapping = None

    def convert_mapping(self, mapping):
        return {int(k): int(v) for k, v in mapping.items()}

    @staticmethod
    def load_emnist_from_url(dataset_url):

        mat = sio.loadmat(dataset_url)
        data = mat["dataset"]

        x_train = data["train"][0, 0]["images"][0, 0]
        x_train = x_train.reshape((x_train.shape[0], 28, 28), order="F")

        y_train = data["train"][0, 0]["labels"][0, 0]
        y_train = np.squeeze(y_train)
        y_train -= 1

        x_test = data["test"][0, 0]["images"][0, 0]
        x_test = x_test.reshape((x_test.shape[0], 28, 28), order="F")
        y_test = data["test"][0, 0]["labels"][0, 0]
        y_test = np.squeeze(y_test)
        y_test -= 1

        return (x_train, y_train), (x_test, y_test)

    @cache
    def load(self) -> tf.data.Dataset:

        if self.dataset_url is not None:
            download_url = tf.keras.utils.get_file(origin=self.dataset_url, cache_subdir="data")

            if self.split.lower() == "train":
                (features, labels), (_, _) = self.load_emnist_from_url(download_url)

            elif self.split.lower() == "test":
                (_, _), (features, labels) = self.load_emnist_from_url(download_url)
            else:
                raise DatasetNotLoadedError(f"EMNIST split {self.split} does not exist")
        else:
            if self.split.lower() == "train":
                features, labels = emnist.extract_training_samples("letters")
            elif self.split.lower() == "test":
                features, labels = emnist.extract_test_samples("letters")
            else:
                raise DatasetNotLoadedError(f"EMNIST split {self.split} does not exist")

        if self.indices:
            features, labels = features[self.indices], labels[self.indices]

        if self.label_mapping is not None:
            labels = np.vectorize(self.label_mapping.get)(labels)

        def _scale_features(features, label):
            return tf.cast(features, tf.float32) / 255.0, tf.cast(label, tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(
            len(features), reshuffle_each_iteration=True
        )

        return ds.map(_scale_features)
