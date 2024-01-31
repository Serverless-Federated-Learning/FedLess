import random
from typing import Dict, Iterator, List, Optional

import numpy as np
from tensorflow import keras

from fedless.common.models import DatasetLoaderConfig
from fedless.datasets.dataset_loaders import DatasetNotLoadedError
from fedless.datasets.dataset_partitioning import PartitionDataset
from fedless.datasets.mnist.dataset_loader import MNISTConfig

# from pydantic import (BaseModel, Field)


def create_mnist_train_data_loader_configs(
    n_devices: int, n_shards: int, proxies: Optional[Dict] = None
) -> Iterator[DatasetLoaderConfig]:
    if n_shards % n_devices != 0:
        raise ValueError(f"Can not equally distribute {n_shards} dataset shards among {n_devices} devices...")

    (_, y_train), (_, _) = keras.datasets.mnist.load_data()
    num_train_examples, *_ = y_train.shape

    sorted_labels_idx = np.argsort(y_train, kind="stable")
    sorted_labels_idx_shards = np.split(sorted_labels_idx, n_shards)
    shards_per_device = len(sorted_labels_idx_shards) // n_devices
    np.random.shuffle(sorted_labels_idx_shards)

    for client_idx in range(n_devices):
        client_shards = sorted_labels_idx_shards[client_idx * shards_per_device : (client_idx + 1) * shards_per_device]
        indices = np.concatenate(client_shards)
        # noinspection PydanticTypeChecker,PyTypeChecker
        yield {
            "train_data": DatasetLoaderConfig(
                type="mnist", params=MNISTConfig(indices=indices.tolist(), proxies=proxies)
            )
        }


def create_val_indices(train_labels, indices, val_ratio, original_targets):
    assert val_ratio >= 0

    partition_sizes = [
        (1 - val_ratio),
        val_ratio,
    ]

    partitions = PartitionDataset(
        labels=train_labels,
        indices=indices,
        sizes=partition_sizes,
        original_targets=original_targets,
        type="origin",
    ).get_partitioned_indices()

    return list(partitions[0]), list(partitions[1])


def create_mnist_data_loader_configs(
    n_devices: int,
    dirichlet_alpha: int = 100,
    partitioning_type: str = "random",
    sizes: List[int] = None,
    label_mapping: dict = None,
    split: str = "train",
    val_ratio: int = None,
    proxies: Optional[Dict] = None,
    dataset_url: str = None,
) -> Iterator[DatasetLoaderConfig]:

    (_, y_train), (_, y_test) = keras.datasets.mnist.load_data()

    if split == "train":
        y = y_train
    elif split == "test":
        y = y_test
    else:
        raise DatasetNotLoadedError(f"MNIST split {split} does not exist")

    if label_mapping is not None:
        idx_filter_mask = [y == int(i) for i in label_mapping.keys()]
        idx = np.where(np.any(idx_filter_mask, axis=0))[0]
        y_filtered = y[idx]
    else:
        y_filtered = y
        idx = np.indices(y.shape)[0]

    if split == "test" and val_ratio is not None:
        raise AssertionError("Cannot create validation split from test dataset")

    train_configs = []
    val_configs = []
    test_configs = []
    if split == "train":
        if val_ratio is not None:
            idx, val_idx = create_val_indices(
                train_labels=y_filtered,
                indices=idx,
                val_ratio=val_ratio,
                original_targets=y,
            )

            for client_idx in range(n_devices):
                val_loader_config = DatasetLoaderConfig(
                    type="mnist",
                    params=MNISTConfig(
                        indices=val_idx,
                        proxies=proxies,
                        split=split,
                        label_mapping=label_mapping,
                    ),
                )
                val_configs.append(val_loader_config)

        if sizes is not None:
            y_filtered_train = y[idx]
            train_partitions = PartitionDataset(
                labels=y_filtered_train,
                indices=idx,
                sizes=sizes,
                original_targets=y,
                type=partitioning_type,
                alpha=dirichlet_alpha,
            ).get_partitioned_indices()

            for client_idx in range(n_devices):
                # noinspection PydanticTypeChecker,PyTypeChecker
                train_configs.append(
                    DatasetLoaderConfig(
                        type="mnist",
                        params=MNISTConfig(
                            indices=train_partitions[client_idx],
                            proxies=proxies,
                            split=split,
                            label_mapping=label_mapping,
                        ),
                    )
                )

        else:
            for client_idx in range(n_devices):
                # noinspection PydanticTypeChecker,PyTypeChecker
                train_configs.append(
                    DatasetLoaderConfig(
                        type="mnist",
                        params=MNISTConfig(
                            indices=list(idx),
                            proxies=proxies,
                            split=split,
                            label_mapping=label_mapping,
                        ),
                    )
                )

        return train_configs, val_configs

    else:
        for client_idx in range(n_devices):
            # noinspection PydanticTypeChecker,PyTypeChecker
            test_configs.append(
                DatasetLoaderConfig(
                    type="mnist",
                    params=MNISTConfig(
                        indices=list(idx),
                        proxies=proxies,
                        split=split,
                        label_mapping=label_mapping,
                    ),
                )
            )

        return test_configs


def create_mnist_data_loader_configs_random(
    split: str,
    cardinality: int,
    n_rounds: int,
    label_mapping: dict = None,
    proxies: Optional[Dict] = None,
    dataset_url: str = None,
):

    (_, y_train), (_, y_test) = keras.datasets.mnist.load_data()

    if split == "train":
        y = y_train
    elif split == "test":
        y = y_test
    else:
        raise DatasetNotLoadedError(f"MNIST split {split} does not exist")

    if label_mapping is not None:
        idx_filter_mask = [y == int(i) for i in label_mapping.keys()]
        idx = np.where(np.any(idx_filter_mask, axis=0))[0]
    else:
        idx = np.indices(y.shape)[0]

    random_loaders = []
    for _ in range(n_rounds):
        filtered_indices = random.sample(list(idx), cardinality)

        random_loaders.append(
            DatasetLoaderConfig(
                type="mnist",
                params=MNISTConfig(
                    indices=filtered_indices,
                    proxies=proxies,
                    split=split,
                    label_mapping=label_mapping,
                ),
            )
        )

    return random_loaders
