import random
from typing import Dict, Iterator, List, Optional

import emnist
import numpy as np
import tensorflow as tf

from fedless.common.models import DatasetLoaderConfig
from fedless.datasets.dataset_loaders import DatasetNotLoadedError
from fedless.datasets.dataset_partitioning import PartitionDataset
from fedless.datasets.emnist.dataset_loader import EMNIST, EMNISTConfig
from fedless.datasets.mnist.helpers import create_mnist_data_loader_configs


def get_fedmd_emnist_data_configs(
    clients,
    n_rounds,
    fedmd_params,
    private_data_dirichlet_alpha,
    proxies=None,
    class_distribution=None,
    emnist_url=None,
):

    if class_distribution:
        private_class_mapping = class_distribution.private_class_mapping
        public_class_mapping = class_distribution.public_class_mapping
    else:
        private_class_mapping, public_class_mapping = None, None

    private_mnist_configs_train, _ = list(
        create_mnist_data_loader_configs(
            n_devices=clients,
            sizes=[1.0 / clients for _ in range(clients)],
            dirichlet_alpha=private_data_dirichlet_alpha,
            partitioning_type="non_iid_dirichlet",
            split="train",
            proxies=proxies,
            label_mapping=private_class_mapping,
        )
    )

    private_mnist_configs_test = list(
        create_mnist_data_loader_configs(
            n_devices=clients,
            split="test",
            proxies=proxies,
            label_mapping=private_class_mapping,
        )
    )

    public_mnist_configs_train, _ = list(
        create_emnist_data_loader_configs(
            n_devices=clients,
            partitioning_type="random",
            split="train",
            proxies=proxies,
            label_mapping=public_class_mapping,
            dataset_url=emnist_url,
        )
    )

    public_mnist_configs_test = list(
        create_emnist_data_loader_configs(
            n_devices=clients,
            partitioning_type="random",
            split="test",
            proxies=proxies,
            label_mapping=public_class_mapping,
            dataset_url=emnist_url,
        )
    )

    public_mnist_alignment_configs_train = list(
        create_emnist_data_loader_configs_random(
            split="train",
            cardinality=fedmd_params.logit_alignment_hyperparams.n_alignment,
            n_rounds=n_rounds,
            proxies=proxies,
            label_mapping=public_class_mapping,
            dataset_url=emnist_url,
        )
    )

    mnist_configs = list(
        map(
            lambda w, x, y, z: {"train_data": w, "test_data": x, "public_train_data": y, "public_test_data": z},
            private_mnist_configs_train,
            private_mnist_configs_test,
            public_mnist_configs_train,
            public_mnist_configs_test,
        )
    )
    for mnist_config in mnist_configs:
        mnist_config["public_alignment_data"] = public_mnist_alignment_configs_train

    return mnist_configs


def get_feddf_emnist_data_configs(
    clients,
    n_rounds,
    feddf_params,
    private_data_dirichlet_alpha,
    proxies=None,
    class_distribution=None,
    emnist_url=None,
    mnist_url=None,
):

    if class_distribution:
        private_class_mapping = class_distribution.private_class_mapping
        public_class_mapping = class_distribution.public_class_mapping
    else:
        private_class_mapping, public_class_mapping = None, None

    private_mnist_configs_train, private_mnist_configs_val = list(
        create_mnist_data_loader_configs(
            n_devices=clients,
            val_ratio=0.1,
            sizes=[1.0 / clients for _ in range(clients)],
            dirichlet_alpha=private_data_dirichlet_alpha,
            partitioning_type="non_iid_dirichlet",
            split="train",
            proxies=proxies,
            label_mapping=private_class_mapping,
        )
    )

    private_mnist_configs_test = list(
        create_mnist_data_loader_configs(
            n_devices=clients,
            split="test",
            proxies=proxies,
            label_mapping=private_class_mapping,
        )
    )

    public_mnist_alignment_configs_train = list(
        create_emnist_data_loader_configs_random(
            split="train",
            cardinality=feddf_params.n_distillation_data,
            n_rounds=n_rounds,
            proxies=proxies,
            label_mapping=public_class_mapping,
            dataset_url=emnist_url,
        )
    )

    mnist_configs = list(
        map(
            lambda w, x, y: {"train_data": w, "test_data": x, "val_data": y},
            private_mnist_configs_train,
            private_mnist_configs_test,
            private_mnist_configs_val,
        )
    )
    for mnist_config in mnist_configs:
        mnist_config["public_alignment_data"] = public_mnist_alignment_configs_train

    return mnist_configs


def create_emnist_data_loader_configs_random(
    split: str,
    cardinality: int,
    n_rounds: int,
    label_mapping: dict = None,
    proxies: Optional[Dict] = None,
    dataset_url: str = None,
):

    if dataset_url is not None:
        download_url = tf.keras.utils.get_file(origin=dataset_url, cache_subdir="data")

        if split.lower() == "train":
            (_, y), (_, _) = EMNIST.load_emnist_from_url(download_url)

        elif split.lower() == "test":
            (_, _), (y, _) = EMNIST.load_emnist_from_url(download_url)
        else:
            raise DatasetNotLoadedError(f"EMNIST split {split} does not exist")
    else:
        if split.lower() == "train":
            _, y = emnist.extract_training_samples("letters")
        elif split.lower() == "test":
            _, y = emnist.extract_test_samples("letters")
        else:
            raise DatasetNotLoadedError(f"EMNIST split {split} does not exist")

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
                type="emnist",
                params=EMNISTConfig(
                    indices=filtered_indices,
                    proxies=proxies,
                    split=split,
                    label_mapping=label_mapping,
                    dataset_url=dataset_url,
                ),
            )
        )

    return random_loaders


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


def create_emnist_data_loader_configs(
    n_devices: int,
    sizes: List[int] = None,
    dirichlet_alpha: int = 100,
    partitioning_type: str = "random",
    label_mapping: dict = None,
    split: str = "train",
    val_ratio: int = None,
    proxies: Optional[Dict] = None,
    dataset_url: str = None,
) -> Iterator[DatasetLoaderConfig]:

    if dataset_url is not None:
        download_url = tf.keras.utils.get_file(origin=dataset_url, cache_subdir="data")

        if split.lower() == "train":
            (_, y), (_, _) = EMNIST.load_emnist_from_url(download_url)

        elif split.lower() == "test":
            (_, _), (_, y) = EMNIST.load_emnist_from_url(download_url)
        else:
            raise DatasetNotLoadedError(f"EMNIST split {split} does not exist")
    else:
        if split.lower() == "train":
            _, y = emnist.extract_training_samples("letters")
        elif split.lower() == "test":
            _, y = emnist.extract_test_samples("letters")
        else:
            raise DatasetNotLoadedError(f"EMNIST split {split} does not exist")

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
                    type="emnist",
                    params=EMNISTConfig(
                        indices=val_idx,
                        proxies=proxies,
                        split=split,
                        label_mapping=label_mapping,
                        dataset_url=dataset_url,
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
                        type="emnist",
                        params=EMNISTConfig(
                            indices=train_partitions[client_idx],
                            proxies=proxies,
                            split=split,
                            label_mapping=label_mapping,
                            dataset_url=dataset_url,
                        ),
                    )
                )

        else:
            for client_idx in range(n_devices):
                # noinspection PydanticTypeChecker,PyTypeChecker
                train_configs.append(
                    DatasetLoaderConfig(
                        type="emnist",
                        params=EMNISTConfig(
                            indices=list(idx),
                            proxies=proxies,
                            split=split,
                            label_mapping=label_mapping,
                            dataset_url=dataset_url,
                        ),
                    )
                )

        return train_configs, val_configs

    else:
        for client_idx in range(n_devices):
            # noinspection PydanticTypeChecker,PyTypeChecker
            test_configs.append(
                DatasetLoaderConfig(
                    type="emnist",
                    params=EMNISTConfig(
                        indices=list(idx),
                        proxies=proxies,
                        split=split,
                        label_mapping=label_mapping,
                        dataset_url=dataset_url,
                    ),
                )
            )

        return test_configs
