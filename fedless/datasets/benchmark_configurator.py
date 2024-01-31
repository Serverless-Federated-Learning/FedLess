import logging

# import uuid
# from itertools import cycle
# from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from fedless.common.models import DatasetLoaderConfig
from fedless.controller.models import ExperimentConfig
from fedless.datasets.char_prediction.helpers import (
    get_feddf_char_prediction_data_configs,
    get_fedmd_char_prediction_data_configs,
)
from fedless.datasets.cifar.helpers import (
    get_feddf_cifar_data_configs,
    get_fedmd_cifar_data_configs,
)
from fedless.datasets.emnist.helpers import (
    get_feddf_emnist_data_configs,
    get_fedmd_emnist_data_configs,
)
from fedless.datasets.fedscale.google_speech.dataset_loader import FedScaleConfig
from fedless.datasets.leaf.dataset_loader import LEAFConfig
from fedless.datasets.mnist.dataset_loader import MNISTConfig
from fedless.datasets.mnist.helpers import create_mnist_train_data_loader_configs

# import click


FILE_SERVER = "http://10.103.195.168:80"
logger = logging.getLogger(__name__)

# only for global test data
def create_mnist_test_config(proxies) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(type="mnist", params=MNISTConfig(split="test", proxies=proxies))


# returns the configs and the number of clients available for testing
# noinspection PydanticTypeChecker,PyTypeChecker
def create_data_configs(
    dataset: str,
    clients: int,
    n_rounds: int,
    config: ExperimentConfig,
    private_data_dirichlet_alpha: float,
    proxies: Optional[Dict] = None,
) -> Tuple[List[Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]], int,]:
    dataset = dataset.lower()
    if dataset == "mnist":
        mn_configs = list(create_mnist_train_data_loader_configs(n_devices=clients, n_shards=600, proxies=proxies))
        return mn_configs, len(mn_configs)

    elif dataset in ["femnist", "shakespeare"]:
        configs = []
        for client_idx in range(clients):
            train = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/datasets/leaf/data/{dataset}/data/"
                    f"train/user_{client_idx}_train_9.json",
                ),
            )
            test = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/datasets/leaf/data/{dataset}/data/" f"test/user_{client_idx}_test_9.json",
                ),
            )
            configs.append({"train_data": train, "test_data": test})
        return configs, len(configs)

    elif dataset == "speech":
        configs = []
        num_test_clients = 216
        for client_idx in range(clients):
            train = DatasetLoaderConfig(
                type="speech",
                params=FedScaleConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/datasets/google_speech/npz_570/train/client_{client_idx}.npz",
                ),
            )
            # if number of test clients is smaller tha number of clients just reloop the assignment
            test = DatasetLoaderConfig(
                type="speech",
                params=FedScaleConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/datasets/google_speech/npz/test/client_{client_idx%num_test_clients}.npz",
                ),
            )
            configs.append({"train_data": train, "test_data": test})
        return configs, min(len(configs), num_test_clients)

    elif dataset == "fedmd_mnist":

        emnist_configs = get_fedmd_emnist_data_configs(
            clients=clients,
            n_rounds=n_rounds,
            fedmd_params=config.clients.hyperparams.fedmd,
            private_data_dirichlet_alpha=private_data_dirichlet_alpha,
            proxies=proxies,
            class_distribution=config.class_distribution,
            emnist_url=f"{FILE_SERVER}/data/emnist/emnist-letters.mat",
        )

        return emnist_configs, len(emnist_configs)

    elif dataset == "fedmd_cifar":

        cifar_configs = get_fedmd_cifar_data_configs(
            clients=clients,
            n_rounds=n_rounds,
            fedmd_params=config.clients.hyperparams.fedmd,
            private_data_dirichlet_alpha=private_data_dirichlet_alpha,
            proxies=proxies,
            class_distribution=config.class_distribution,
            dataset_url_cifar10=f"{FILE_SERVER}/data/cifar10/cifar-10-batches-py",
            dataset_url_cifar100=f"{FILE_SERVER}/data/cifar100/cifar-100-python",
        )

        return cifar_configs, len(cifar_configs)

    elif dataset == "fedmd_shakespeare":

        configs = get_fedmd_char_prediction_data_configs(
            clients=clients,
            n_rounds=n_rounds,
            fedmd_params=config.clients.hyperparams.fedmd,
            proxies=proxies,
            file_server=FILE_SERVER,
        )

        return configs, len(configs)

    elif dataset == "feddf_cifar":

        cifar_configs = get_feddf_cifar_data_configs(
            clients=clients,
            n_rounds=n_rounds,
            feddf_params=config.aggregator.agg_functions[0].hyperparams.feddf_hyperparams,
            private_data_dirichlet_alpha=private_data_dirichlet_alpha,
            class_distribution=config.class_distribution,
            dataset_url_cifar10=f"{FILE_SERVER}/data/cifar10/cifar-10-batches-py",
            dataset_url_cifar100=f"{FILE_SERVER}/data/cifar100/cifar-100-python",
        )

        return cifar_configs, len(cifar_configs)

    elif dataset == "feddf_mnist":

        cifar_configs = get_feddf_emnist_data_configs(
            clients=clients,
            n_rounds=n_rounds,
            feddf_params=config.aggregator.agg_functions[0].hyperparams.feddf_hyperparams,
            private_data_dirichlet_alpha=private_data_dirichlet_alpha,
            class_distribution=config.class_distribution,
            emnist_url=f"{FILE_SERVER}/data/emnist/emnist-letters.mat",
        )

        return cifar_configs, len(cifar_configs)

    elif dataset == "feddf_shakespeare":

        configs = get_feddf_char_prediction_data_configs(
            clients=clients,
            n_rounds=n_rounds,
            feddf_params=config.aggregator.agg_functions[0].hyperparams.feddf_hyperparams,
            file_server=FILE_SERVER,
        )

        return configs, len(configs)

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
