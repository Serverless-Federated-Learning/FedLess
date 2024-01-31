import os
import random
from typing import Dict, Iterator, List, Optional

import numpy as np
from six.moves import urllib
from tensorflow import keras

from fedless.common.models import DatasetLoaderConfig
from fedless.datasets.cifar.dataset_loader import CIFAR, CIFARConfig
from fedless.datasets.dataset_loaders import DatasetNotLoadedError
from fedless.datasets.dataset_partitioning import PartitionDataset


def get_feddf_cifar_data_configs(
    clients,
    n_rounds,
    feddf_params,
    private_data_dirichlet_alpha,
    proxies=None,
    class_distribution=None,
    dataset_url_cifar10=None,
    dataset_url_cifar100=None,
):

    if class_distribution:
        private_class_mapping = class_distribution.private_class_mapping
        public_class_mapping = class_distribution.public_class_mapping
    else:
        private_class_mapping, public_class_mapping = None, None

    private_cifar_configs_train, private_cifar_configs_val = create_cifar_data_loader_configs(
        n_devices=clients,
        dataset="cifar10",
        sizes=[1.0 / clients for _ in range(clients)],
        dirichlet_alpha=private_data_dirichlet_alpha,
        partitioning_type="non_iid_dirichlet",
        split="train",
        val_ratio=0.1,
        proxies=proxies,
        label_mapping=private_class_mapping,
        dataset_url=dataset_url_cifar10,
    )

    private_cifar_configs_test = create_cifar_data_loader_configs(
        n_devices=clients,
        dataset="cifar10",
        split="test",
        proxies=proxies,
        label_mapping=private_class_mapping,
        dataset_url=dataset_url_cifar10,
    )

    public_cifar_alignment_configs_train = create_cifar_data_loader_configs_random(
        dataset="cifar100",
        split="train",
        cardinality=feddf_params.n_distillation_data,
        n_rounds=n_rounds,
        proxies=proxies,
        label_mapping=public_class_mapping,
        dataset_url=dataset_url_cifar100,
    )

    cifar_configs = list(
        map(
            lambda w, x, y: {"train_data": w, "val_data": x, "test_data": y},
            list(private_cifar_configs_train),
            list(private_cifar_configs_val),
            private_cifar_configs_test,
        )
    )
    for cifar_config in cifar_configs:
        cifar_config["public_alignment_data"] = public_cifar_alignment_configs_train

    return cifar_configs


def get_fedmd_cifar_data_configs(
    clients,
    n_rounds,
    fedmd_params,
    private_data_dirichlet_alpha,
    proxies=None,
    class_distribution=None,
    dataset_url_cifar10=None,
    dataset_url_cifar100=None,
):

    if class_distribution:
        private_class_mapping = class_distribution.private_class_mapping
        public_class_mapping = class_distribution.public_class_mapping
    else:
        private_class_mapping, public_class_mapping = None, None

    private_cifar_configs_train, _ = create_cifar_data_loader_configs(
        n_devices=clients,
        dataset="cifar100",
        sizes=[1.0 / clients for _ in range(clients)],
        dirichlet_alpha=private_data_dirichlet_alpha,
        partitioning_type="non_iid_dirichlet",
        split="train",
        proxies=proxies,
        label_mapping=private_class_mapping,
        dataset_url=dataset_url_cifar100,
    )

    private_cifar_configs_test = create_cifar_data_loader_configs(
        n_devices=clients,
        dataset="cifar100",
        sizes=None,
        partitioning_type="non_iid_dirichlet",
        split="test",
        proxies=proxies,
        label_mapping=private_class_mapping,
        dataset_url=dataset_url_cifar100,
    )

    public_cifar_configs_train, _ = create_cifar_data_loader_configs(
        n_devices=clients,
        dataset="cifar10",
        partitioning_type="random",
        split="train",
        proxies=proxies,
        label_mapping=public_class_mapping,
        dataset_url=dataset_url_cifar10,
    )

    public_cifar_configs_test = create_cifar_data_loader_configs(
        n_devices=clients,
        dataset="cifar10",
        partitioning_type="random",
        split="test",
        proxies=proxies,
        label_mapping=public_class_mapping,
        dataset_url=dataset_url_cifar10,
    )

    public_cifar_alignment_configs_train = create_cifar_data_loader_configs_random(
        dataset="cifar10",
        split="train",
        cardinality=fedmd_params.logit_alignment_hyperparams.n_alignment,
        n_rounds=n_rounds,
        proxies=proxies,
        label_mapping=public_class_mapping,
        dataset_url=dataset_url_cifar10,
    )

    cifar_configs = list(
        map(
            lambda w, x, y, z: {"train_data": w, "test_data": x, "public_train_data": y, "public_test_data": z},
            private_cifar_configs_train,
            private_cifar_configs_test,
            public_cifar_configs_train,
            public_cifar_configs_test,
        )
    )
    for cifar_config in cifar_configs:
        cifar_config["public_alignment_data"] = public_cifar_alignment_configs_train

    return cifar_configs


def create_cifar_data_loader_configs_random(
    dataset: str,
    split: str,
    cardinality: int,
    n_rounds: int,
    label_mapping: dict = None,
    proxies: Optional[Dict] = None,
    dataset_url: str = None,
):

    if "100" in dataset:
        if dataset_url is not None:
            train_url, _ = urllib.request.urlretrieve(dataset_url + "/train", "/datasets/cifar_100_train")
            test_url, _ = urllib.request.urlretrieve(dataset_url + "/test", "/datasets/cifar_100_test")
            (_, y_train), (_, y_test) = CIFAR.load_cifar_data_from_url(dataset, (train_url, test_url))
    else:
        if dataset_url is not None:
            train_batches = []
            for i in range(1, 6):
                fpath = os.path.join(dataset_url, "data_batch_" + str(i))
                train_batches.append(urllib.request.urlretrieve(fpath, f"/datasets/cifar_10_train_{i}")[0])
            test_batch, _ = urllib.request.urlretrieve(dataset_url + "/test_batch", "/datasets/cifar_10_test")
            (_, y_train), (_, y_test) = CIFAR.load_cifar_data_from_url(dataset, (train_batches, test_batch))
        else:
            (_, y_train), (_, y_test) = keras.datasets.cifar10.load_data()

    if split == "train":
        y = y_train
    elif split == "test":
        y = y_test
    else:
        raise DatasetNotLoadedError(f"CIFAR split {split} does not exist")

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
                type="cifar",
                params=CIFARConfig(
                    dataset=dataset,
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


def create_cifar_data_loader_configs(
    n_devices: int,
    dataset: str,
    partitioning_type: str = "origin",
    dirichlet_alpha: int = 100,
    sizes: List[float] = None,
    label_mapping: dict = None,
    split: str = "train",
    val_ratio: int = None,
    proxies: Optional[Dict] = None,
    dataset_url: str = None,
) -> Iterator[DatasetLoaderConfig]:

    if "100" in dataset:
        if dataset_url is not None:
            # train_url = keras.utils.get_file(origin=dataset_url + "/train", cache_subdir="data", extract=True)
            # test_url = keras.utils.get_file(origin=dataset_url + "/test", cache_subdir="data", extract=True)
            train_url, _ = urllib.request.urlretrieve(dataset_url + "/train", "/datasets/cifar_100_train")
            test_url, _ = urllib.request.urlretrieve(dataset_url + "/test", "/datasets/cifar_100_test")
            (_, y_train), (_, y_test) = CIFAR.load_cifar_data_from_url(dataset, (train_url, test_url))
        else:
            (_, y_train), (_, y_test) = keras.datasets.cifar100.load_data()
    else:
        if dataset_url is not None:
            train_batches = []
            for i in range(1, 6):
                fpath = os.path.join(dataset_url, "data_batch_" + str(i))
                train_batches.append(urllib.request.urlretrieve(fpath, f"/datasets/cifar_10_train_{i}")[0])
            test_batch, _ = urllib.request.urlretrieve(dataset_url + "/test_batch", "/datasets/cifar_10_test")
            (_, y_train), (_, y_test) = CIFAR.load_cifar_data_from_url(dataset, (train_batches, test_batch))
        else:
            (_, y_train), (_, y_test) = keras.datasets.cifar10.load_data()

    if split == "train":
        y = y_train.squeeze()
    elif split == "test":
        y = y_test.squeeze()
    else:
        raise DatasetNotLoadedError(f"CIFAR split {split} does not exist")

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
                    type="cifar",
                    params=CIFARConfig(
                        dataset=dataset,
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
                        type="cifar",
                        params=CIFARConfig(
                            dataset=dataset,
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
                        type="cifar",
                        params=CIFARConfig(
                            dataset=dataset,
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
                    type="cifar",
                    params=CIFARConfig(
                        dataset=dataset,
                        indices=list(idx),
                        proxies=proxies,
                        split=split,
                        label_mapping=label_mapping,
                        dataset_url=dataset_url,
                    ),
                )
            )

        return test_configs
