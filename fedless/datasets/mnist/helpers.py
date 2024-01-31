from typing import Iterator, Optional, Dict

import numpy as np
from tensorflow import keras
from typing import Dict, Iterator, Optional
from fedless.datasets.mnist.dataset_loader import MNISTConfig

from fedless.common.models import DatasetLoaderConfig

# from pydantic import (BaseModel, Field)


def create_mnist_train_data_loader_configs(
    n_devices: int, n_shards: int, proxies: Optional[Dict] = None
) -> Iterator[DatasetLoaderConfig]:
    if n_shards % n_devices != 0:
        raise ValueError(
            f"Can not equally distribute {n_shards} dataset shards among {n_devices} devices..."
        )

    (_, y_train), (_, _) = keras.datasets.mnist.load_data()
    num_train_examples, *_ = y_train.shape

    sorted_labels_idx = np.argsort(y_train, kind="stable")
    sorted_labels_idx_shards = np.split(sorted_labels_idx, n_shards)
    shards_per_device = len(sorted_labels_idx_shards) // n_devices
    np.random.shuffle(sorted_labels_idx_shards)

    for client_idx in range(n_devices):
        client_shards = sorted_labels_idx_shards[
            client_idx * shards_per_device : (client_idx + 1) * shards_per_device
        ]
        indices = np.concatenate(client_shards)
        # noinspection PydanticTypeChecker,PyTypeChecker
        yield DatasetLoaderConfig(
            type="mnist", params=MNISTConfig(indices=indices.tolist(), proxies=proxies)
        )
