import logging
import sys
import numpy as np

from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from fedless.datasets.dataset_loaders import DatasetLoader
from fedless.datasets.fedscale.google_speech.dataset_loader import (
    FedScale,
    FedScaleConfig,
)
from fedless.datasets.fedscale.google_speech.model import create_speech_cnn

from fedless.datasets.leaf.model import create_femnist_cnn, create_shakespeare_lstm
from fedless.datasets.mnist.helpers import create_mnist_train_data_loader_configs
from fedless.datasets.mnist.model import create_mnist_cnn
from fedless.common.models import (
    BinaryStringFormat,
    DatasetLoaderConfig,
    MongodbConnectionConfig,
    NpzWeightsSerializerConfig,
    SerializedParameters,
    WeightsSerializerConfig,
)
from fedless.common.persistence.client_daos import ModelDao, ParameterDao
from fedless.common.serialization import (
    Base64StringConverter,
    NpzWeightsSerializer,
    serialize_model,
)
from fedless.datasets.mnist.dataset_loader import MNISTConfig
from fedless.datasets.leaf.dataset_loader import LEAFConfig
from fedless.datasets.leaf.dataset_loader import LEAF
from fedless.datasets.mnist.dataset_loader import MNIST

logger = logging.getLogger(__name__)


class FileServerNotProvided(Exception):
    pass


def create_model(dataset) -> tf.keras.Sequential:
    if dataset.lower() == "femnist":
        return create_femnist_cnn()
    elif dataset.lower() == "shakespeare":
        return create_shakespeare_lstm()
    elif dataset.lower() == "mnist":
        return create_mnist_cnn()
    elif dataset.lower() == "speech":
        return create_speech_cnn((32, 32, 1), 35)
    else:
        raise NotImplementedError()


def init_store_model(
    session: str,
    model: tf.keras.Sequential,
    strategy: str,
    database: MongodbConnectionConfig,
    store_json_serializable: bool = False,
):
    is_scaffold = strategy == "scaffold"

    parameters_dao = ParameterDao(db=database)
    models_dao = ModelDao(db=database)

    serialized_model = serialize_model(model)
    weights = model.get_weights()
    weights_serialized = NpzWeightsSerializer(compressed=False).serialize(weights)
    weights_format = BinaryStringFormat.NONE
    if store_json_serializable:
        weights_serialized = Base64StringConverter.to_str(weights_serialized)
        weights_format = BinaryStringFormat.BASE64
    params = SerializedParameters(
        blob=weights_serialized,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig(compressed=False)
        ),
        string_format=weights_format,
    )

    # initialize global control for SCAFFOLD
    if is_scaffold:
        # with zeros (same shape as model weights)
        server_controls = [
            np.zeros(shape=layer.shape, dtype=np.float32) for layer in weights
        ]
        server_controls_serialized = NpzWeightsSerializer(compressed=False).serialize(
            server_controls
        )
        if store_json_serializable:
            server_controls_serialized = Base64StringConverter.to_str(
                server_controls_serialized
            )
            weights_format = BinaryStringFormat.BASE64
        server_controls = SerializedParameters(
            blob=server_controls_serialized,
            serializer=WeightsSerializerConfig(
                type="npz", params=NpzWeightsSerializerConfig(compressed=False)
            ),
            string_format=weights_format,
        )
        logger.debug(
            f"[SCAFFOLD] Global controls initialize (with zeros) and successfully serialized. Total size is {sys.getsizeof(server_controls_serialized) // 10 ** 6}MB. "
            f"Saving initial global controls to database"
        )
    else:
        server_controls = None

    logger.debug(
        f"Model loaded and successfully serialized. Total size is {sys.getsizeof(weights_serialized) // 10 ** 6}MB. "
        f"Saving initial parameters to database"
    )
    parameters_dao.save(
        session_id=session, round_id=0, params=params, global_controls=server_controls
    )
    models_dao.save(session_id=session, model=serialized_model)


# only for global test data
def create_mnist_test_config(proxies) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        type="mnist", params=MNISTConfig(split="test", proxies=proxies)
    )


# returns the configs and the number of clients available for testing
# noinspection PydanticTypeChecker,PyTypeChecker
def create_data_configs(
    dataset: str,
    clients: int,
    file_server: Optional[str] = None,
    proxies: Optional[Dict] = None,
) -> Tuple[
    List[Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]],
    int,
]:
    dataset = dataset.lower()
    if dataset == "mnist":
        mn_configs = list(
            create_mnist_train_data_loader_configs(
                n_devices=clients, n_shards=600, proxies=proxies
            )
        )
        return mn_configs, len(mn_configs)

    elif dataset in ["femnist", "shakespeare"]:
        if file_server is None:
            raise FileServerNotProvided(
                f"No file server is specified for dataset: {dataset}"
            )
        configs = []
        for client_idx in range(clients):
            train = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{file_server}/data/leaf/data/{dataset}/data/"
                    f"train/user_{client_idx}_train_9.json",
                ),
            )
            test = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{file_server}/data/leaf/data/{dataset}/data/"
                    f"test/user_{client_idx}_test_9.json",
                ),
            )
            configs.append((train, test))
        return configs, len(configs)
    elif dataset == "speech":
        configs = []
        num_test_clients = 216
        for client_idx in range(clients):
            train = DatasetLoaderConfig(
                type="speech",
                params=FedScaleConfig(
                    dataset=dataset,
                    location=f"{file_server}/data/google_speech/npz/train/client_{client_idx}.npz",
                ),
            )
            # if number of test clients is smaller tha number of clients just reloop the assignment
            test = DatasetLoaderConfig(
                type="speech",
                params=FedScaleConfig(
                    dataset=dataset,
                    location=f"{file_server}/data/google_speech/npz/test/client_{client_idx%num_test_clients}.npz",
                ),
            )
            configs.append((train, test))
        return configs, min(len(configs), num_test_clients)
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")


class DatasetLoaderBuilder:
    """Convenience class to construct loaders from config"""

    @staticmethod
    def from_config(config: DatasetLoaderConfig) -> DatasetLoader:
        """
        Construct loader from config
        :raises NotImplementedError if the loader does not exist
        """
        if config.type == "leaf":
            params: LEAFConfig = config.params
            return LEAF(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
                user_indices=params.user_indices,
            )
        elif config.type == "mnist":
            params: MNISTConfig = config.params
            # location is added by default here
            return MNIST(
                split=params.split, indices=params.indices, proxies=params.proxies
            )
        elif config.type == "speech":
            params: FedScaleConfig = config.params
            return FedScale(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
                user_indices=params.user_indices,
            )
        else:
            raise NotImplementedError(
                f"Dataset loader {config.type} is not implemented"
            )
