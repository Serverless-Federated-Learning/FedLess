import json
import logging
import os
import pickle
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import requests
import tensorflow as tf
from pydantic import BaseModel, Field
from six.moves import urllib
from tensorflow import keras

from fedless.common.cache import cache
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CIFARDataset(str, Enum):
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


class CIFARConfig(BaseModel):

    dataset: CIFARDataset
    type: str = Field("cifar", const=True)
    indices: List[int] = None
    split: str = "train"
    label_mapping: Optional[dict]
    dataset_url: Optional[str] = None
    proxies: Optional[Dict] = None


class CIFAR(DatasetLoader):
    def __init__(
        self,
        dataset: str,
        label_mapping: dict = None,
        indices: Optional[List[int]] = None,
        split: str = "train",
        proxies: Optional[Dict] = None,
        dataset_url: str = None,
    ):
        self.split = split
        self.indices = indices
        self.proxies = proxies or {}
        self.dataset = dataset
        self.dataset_url = dataset_url

        if label_mapping:
            self.label_mapping = self.convert_mapping(label_mapping)
        else:
            self.label_mapping = None

    def convert_mapping(self, mapping):
        return {int(k): int(v) for k, v in mapping.items()}

    # def fetch_data_url(self, url: str):
    #     try:
    #         response = requests.get(url, params=self.http_params)
    #         response.raise_for_status()
    #         return response.json()
    #     except ValueError as e:
    #         raise DatasetFormatError(f"Invalid JSON returned from ${url}") from e
    #     except RequestException as e:
    #         raise DatasetNotLoadedError(e) from e

    @staticmethod
    def load_batch(fpath, label_key="labels"):

        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="bytes")
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode("utf8")] = v
            d = d_decoded
        data = d["data"]
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    @staticmethod
    def load_cifar_data_from_url(dataset, url):

        if dataset == "cifar10":
            num_train_samples = 50000

            x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
            y_train = np.empty((num_train_samples,), dtype="uint8")

            train_batches, test_batch = url

            for i in range(1, 6):
                (
                    x_train[(i - 1) * 10000 : i * 10000, :, :, :],
                    y_train[(i - 1) * 10000 : i * 10000],
                ) = CIFAR.load_batch(train_batches[i - 1])

            x_test, y_test = CIFAR.load_batch(test_batch)

        else:
            train_url, test_url = url
            # fpath = os.path.join(url, "train")
            x_train, y_train = CIFAR.load_batch(train_url, label_key="fine_labels")

            # fpath = os.path.join(url, "test")
            x_test, y_test = CIFAR.load_batch(test_url, label_key="fine_labels")

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

        x_test = x_test.astype(x_train.dtype)
        y_test = y_test.astype(y_train.dtype)

        return (x_train, y_train), (x_test, y_test)

    @cache
    def load(self) -> tf.data.Dataset:

        # tx_file_path = tf.keras.utils.get_file(cache_subdir="data", origin=self.location)
        logger.info("Starting to load CIFAR data")
        if "100" in self.dataset:
            if self.dataset_url is not None:
                # train_url = keras.utils.get_file(origin=self.dataset_url + "/train", cache_subdir="data")
                train_url, _ = urllib.request.urlretrieve(self.dataset_url + "/train", "cifar_100_train")
                logger.info("Fetched CIFAR100 train")
                # test_url = keras.utils.get_file(origin=self.dataset_url + "/test", cache_subdir="data")
                test_url, _ = urllib.request.urlretrieve(self.dataset_url + "/test", "cifar_100_test")
                logger.info("Fetched CIFAR100 test")
                logger.info("Starting data load from downloaded files")
                (x_train, y_train), (x_test, y_test) = CIFAR.load_cifar_data_from_url(
                    self.dataset, (train_url, test_url)
                )
            else:
                (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        else:
            if self.dataset_url is not None:
                train_batches = []
                for i in range(1, 6):
                    logger.info(f"Fetching CIFAR 10 batch {str(i)}")
                    fpath = os.path.join(self.dataset_url, "data_batch_" + str(i))
                    train_batches.append(urllib.request.urlretrieve(fpath, f"cifar_10_train_{i}")[0])
                    # train_batches.append(keras.utils.get_file(origin=fpath, cache_subdir="data"))
                logger.info("Fetched CIFAR10 train data batches")
                # test_batch = keras.utils.get_file(origin=self.dataset_url + "/test_batch", cache_subdir="data")
                test_batch, _ = urllib.request.urlretrieve(self.dataset_url + "/test_batch", "cifar_10_test")
                logger.info("Fetched CIFAR10 test data batches")
                (x_train, y_train), (x_test, y_test) = CIFAR.load_cifar_data_from_url(
                    self.dataset, (train_batches, test_batch)
                )
                logger.info("Loaded complete CIFAR10 data")
            else:
                (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            # for index, cls_ in enumerate(self.private_classes):
            #     y_train[y_train == cls_] = index + len(self.public_classes)
            #     y_test[y_test == cls_] = index + len(self.public_classes)

        # with np.load(tx_file_path, allow_pickle=True) as f:
        #     x_train, y_train = f["x_train"], f["y_train"]
        #     x_test, y_test = f["x_test"], f["y_test"]

        if self.split.lower() == "train":
            features, labels = x_train, y_train
        elif self.split.lower() == "test":
            features, labels = x_test, y_test
        else:
            raise DatasetNotLoadedError(f"CIFAR split {self.split} does not exist")

        if self.indices:
            features, labels = features[self.indices], labels[self.indices]

        if self.label_mapping is not None:
            labels = np.vectorize(self.label_mapping.get)(labels)

        def _scale_features(features, label):
            return tf.cast(features, tf.float32) / 255.0, tf.cast(label, tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels))

        return ds.map(_scale_features)
