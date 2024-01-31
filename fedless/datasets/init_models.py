import logging
import sys
from typing import List

import tensorflow as tf

from fedless.common.models import (
    BinaryStringFormat,
    ClientConfig,
    MongodbConnectionConfig,
    NpzWeightsSerializerConfig,
    SerializedParameters,
    WeightsSerializerConfig,
)
from fedless.common.persistence.client_daos import (
    ClientModelDao,
    ClientParameterDao,
    ModelDao,
    ParameterDao,
)
from fedless.common.serialization import (
    Base64StringConverter,
    NpzWeightsSerializer,
    serialize_model,
)
from fedless.datasets.fedscale.google_speech.model import create_speech_cnn
from fedless.datasets.leaf.model import create_femnist_cnn, create_shakespeare_lstm
from fedless.datasets.mnist.model import create_mnist_cnn
from fedless.datasets.models import (
    cnn_2layer_fc_model,
    cnn_3layer_fc_model,
    cnn_4layer_fc_model,
    lstm_1layer_fc_model,
    lstm_2layer_fc_model,
)

logger = logging.getLogger(__name__)


CANDIDATE_MODELS_CNN = {
    "2_layer_CNN": cnn_2layer_fc_model,
    "3_layer_CNN": cnn_3layer_fc_model,
    "4_layer_CNN": cnn_4layer_fc_model,
}
CANDIDATE_MODELS_LSTM = {"1_layer_LSTM": lstm_1layer_fc_model, "2_layer_LSTM": lstm_2layer_fc_model}


def create_model(dataset, config) -> tf.keras.Sequential:
    if dataset.lower() == "femnist":
        return {"global_model": create_femnist_cnn()}
    elif dataset.lower() == "shakespeare":
        return {"global_model": create_shakespeare_lstm()}
    elif dataset.lower() == "mnist":
        return {"global_model": create_mnist_cnn()}
    elif dataset.lower() == "speech":
        return {"global_model": create_speech_cnn((32, 32, 1), 35)}
    elif dataset.lower() == "fedmd_mnist":
        n_classes = len(config.class_distribution.private_class_mapping) + len(
            config.class_distribution.public_class_mapping
        )
        return {"local_models": create_config_cnn(config.clients.functions, (28, 28), n_classes=n_classes)}
    elif dataset.lower() == "fedmd_cifar":
        n_classes = len(config.class_distribution.private_class_mapping) + len(
            config.class_distribution.public_class_mapping
        )
        return {"local_models": create_config_cnn(config.clients.functions, (32, 32, 3), n_classes=n_classes)}

    elif dataset.lower() == "fedmd_shakespeare":
        return {"local_models": create_config_lstm_fedmd(config.clients.functions)}

    elif dataset.lower() == "feddf_cifar":
        return {
            "local_models": create_config_cnn(
                config.clients.functions,
                (32, 32, 3),
                n_classes=10,
            )
        }
    elif dataset.lower() == "feddf_mnist":
        return {
            "local_models": create_config_cnn(
                config.clients.functions,
                (28, 28),
                n_classes=10,
            )
        }

    elif dataset.lower() == "feddf_shakespeare":
        return {"local_models": create_config_lstm_fedmd(config.clients.functions)}

    else:
        raise NotImplementedError()


def create_config_cnn(function_configs, input_shape, n_classes):

    if isinstance(function_configs, List):
        models = []

        for _, item in enumerate(function_configs):
            model_name = item.function.model_config.model_type
            model_params = item.function.model_config.params.dict()
            current_model = CANDIDATE_MODELS_CNN[model_name](
                n_classes=n_classes,
                input_shape=input_shape,
                **model_params,
            )
            model_params_str = "_".join("{!s}_{!r}".format(key, val) for (key, val) in model_params.items())
            models.append((current_model, model_name + "_" + model_params_str))
        return models
    else:
        return CANDIDATE_MODELS_CNN[function_configs.function.model_config.model_type](
            n_classes=n_classes,
            input_shape=input_shape,
            **function_configs.function.model_config.params.dict(),
        )


def create_config_lstm_fedmd(function_configs):
    models = []

    for _, item in enumerate(function_configs):
        model_name = item.function.model_config.model_type
        model_params = item.function.model_config.params.dict()
        model_params_str = "_".join("{!s}_{!r}".format(key, val) for (key, val) in model_params.items())
        current_model = CANDIDATE_MODELS_LSTM[model_name](**model_params)
        models.append((current_model, model_name + "_" + model_params_str))
    return models


def init_store_model(
    session: str,
    model: tf.keras.Sequential,
    database: MongodbConnectionConfig,
    store_json_serializable: bool = False,
):
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
        serializer=WeightsSerializerConfig(type="npz", params=NpzWeightsSerializerConfig(compressed=False)),
        string_format=weights_format,
    )
    logger.debug(
        f"Model loaded and successfully serialized. Total size is {sys.getsizeof(weights_serialized) // 10 ** 6}MB. "
        f"Saving initial parameters to database"
    )
    parameters_dao.save(session_id=session, round_id=0, params=params)
    models_dao.save(session_id=session, model=serialized_model)


def init_store_model_client(
    session: str,
    models: List[tf.keras.Sequential],
    database: MongodbConnectionConfig,
    clients: List[ClientConfig],
    store_json_serializable: bool = False,
):
    client_parameters_dao = ClientParameterDao(db=database)
    client_models_dao = ClientModelDao(db=database)

    for model, client in zip(models, clients):
        model_arch, model_type = model
        # logit_model = remove_last_layer(model_arch, loss="mean_absolute_error")
        serialized_model = serialize_model(model_arch, model_type)
        # serialized_logit_model = serialize_model(logit_model, model_type)
        weights = model_arch.get_weights()
        weights_serialized = NpzWeightsSerializer(compressed=False).serialize(weights)
        weights_format = BinaryStringFormat.NONE
        if store_json_serializable:
            weights_serialized = Base64StringConverter.to_str(weights_serialized)
            weights_format = BinaryStringFormat.BASE64

        params = SerializedParameters(
            blob=weights_serialized,
            serializer=WeightsSerializerConfig(type="npz", params=NpzWeightsSerializerConfig(compressed=False)),
            string_format=weights_format,
        )
        logger.info(
            f"Model loaded and successfully serialized for client {client.client_id}. Total size is {sys.getsizeof(weights_serialized) // 10 ** 6}MB. "
            f"Saving initial parameters to database"
        )
        client_parameters_dao.save(session_id=session, client_id=client.client_id, round_id=0, params=params)
        client_models_dao.save(session_id=session, client_id=client.client_id, model=serialized_model)
