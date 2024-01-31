import base64

import numpy as np
import pytest
import requests_mock
import tensorflow.keras as keras
from pydantic import ValidationError

from fedless.client import (
    default_handler,
    ClientError,
)
from fedless.common.models import (
    Hyperparams,
    ClientInvocationParams,
    ClientResult,
    LEAFConfig,
    DatasetLoaderConfig,
    PayloadModelLoaderConfig,
    ModelLoaderConfig,
    LocalDifferentialPrivacyParams,
)
from fedless.common.providers import (
    lambda_proxy_handler,
    gcloud_http_error_handler,
    openwhisk_action_handler,
)
from fedless.common.serialization import deserialize_parameters
from .common import resource_folder_path

FEMNIST_DATA_DIR = resource_folder_path() / "leaf" / "femnist"
FEMNIST_DATA_FILE_PATH = FEMNIST_DATA_DIR / "femnist_test_data.json"


@pytest.fixture
def model_loader_config():
    payload = (
        resource_folder_path() / "leaf" / "femnist" / "dummy_model_payload.data"
    ).read_text()
    return ModelLoaderConfig(
        type="payload", params=PayloadModelLoaderConfig(payload=payload)
    )


@pytest.fixture
def data_loader_config():
    return DatasetLoaderConfig(
        type="leaf",
        params=LEAFConfig(dataset="femnist", location=FEMNIST_DATA_DIR),
    )


@pytest.fixture
def remote_data_loader_config(requests_mock: requests_mock.Mocker):
    url = "https://test-url.com"
    requests_mock.get(
        url,
        content=FEMNIST_DATA_FILE_PATH.read_bytes(),
    )
    return DatasetLoaderConfig(
        type="leaf",
        params=LEAFConfig(dataset="femnist", location=url),
    )


@pytest.mark.integ
def test_femnist_training_with_local_data(model_loader_config, data_loader_config):
    result = default_handler(
        data_config=data_loader_config,
        model_config=model_loader_config,
        hyperparams=Hyperparams(batch_size=1, epochs=10, shuffle_data=False),
    )
    losses = result.history["loss"]
    weights = deserialize_parameters(result.parameters)
    assert isinstance(weights, list)
    assert isinstance(weights[0], np.ndarray)
    assert losses[0] > losses[-1]


@pytest.mark.integ
@pytest.mark.parametrize("num_microbatches", [None, 1, 2, 4, 8])
def test_femnist_training_with_local_data_and_dp(
    model_loader_config, data_loader_config, num_microbatches
):
    result = default_handler(
        data_config=data_loader_config,
        model_config=model_loader_config,
        hyperparams=Hyperparams(
            batch_size=8,
            optimizer="SGD",
            epochs=8,
            shuffle_data=False,
            local_privacy=LocalDifferentialPrivacyParams(
                l2_norm_clip=1.0,
                noise_multiplier=1.0,
                num_microbatches=num_microbatches,
            ),
        ),
    )
    losses = result.history["loss"]
    weights = deserialize_parameters(result.parameters)
    assert isinstance(weights, list)
    assert isinstance(weights[0], np.ndarray)
    assert losses[0] > losses[-1]


@pytest.mark.integ
def test_cardinality_correct(model_loader_config, data_loader_config):
    result = default_handler(
        data_config=data_loader_config,
        model_config=model_loader_config,
        hyperparams=Hyperparams(batch_size=1, epochs=10, shuffle_data=False),
    )
    assert result.cardinality == 8


@pytest.mark.integ
def test_femnist_training_with_remote_data(
    model_loader_config, remote_data_loader_config
):
    result = default_handler(
        data_config=remote_data_loader_config,
        model_config=model_loader_config,
        hyperparams=Hyperparams(batch_size=10, epochs=5, shuffle_data=True),
    )
    losses = result.history["loss"]
    weights = deserialize_parameters(result.parameters)

    assert isinstance(weights, list)
    assert isinstance(weights[0], np.ndarray)
    assert losses[0] > losses[-1]


@pytest.mark.integ
def test_femnist_training_with_custom_optimizer_and_loss(
    model_loader_config, remote_data_loader_config
):
    result = default_handler(
        data_config=remote_data_loader_config,
        model_config=model_loader_config,
        hyperparams=Hyperparams(
            batch_size=1,
            epochs=10,
            optimizer=keras.optimizers.serialize(keras.optimizers.Adam(lr=0.001)),
            metrics=["accuracy"],
            loss="sparse_categorical_crossentropy",
        ),
    )
    losses = result.history["loss"]
    accuracies = result.history["accuracy"]
    weights = deserialize_parameters(result.parameters)

    assert isinstance(weights, list)
    assert isinstance(weights[0], np.ndarray)
    assert losses[0] > losses[-1]
    assert accuracies[0] < accuracies[-1]


@pytest.mark.integ
def test_femnist_training_wraps_value_error(
    model_loader_config, remote_data_loader_config
):
    with pytest.raises(ClientError):
        default_handler(
            data_config=remote_data_loader_config,
            model_config=model_loader_config,
            hyperparams=Hyperparams(
                batch_size=1,
                epochs=10,
                optimizer=keras.optimizers.serialize(keras.optimizers.Adam(lr=0.001)),
                metrics=["invalid-metric"],  # Line responsible for value error
            ),
        )


@pytest.mark.integ
def test_femnist_training_with_lambda_proxy(
    model_loader_config, remote_data_loader_config
):
    @lambda_proxy_handler((ValidationError, ClientError))
    def lambda_handler(event, context):
        config = ClientInvocationParams.parse_obj(event["body"])
        return default_handler(
            data_config=config.data,
            model_config=config.model,
            hyperparams=config.hyperparams,
        )

    event = {
        "body": ClientInvocationParams(
            data=remote_data_loader_config,
            model=model_loader_config,
            hyperparams=Hyperparams(batch_size=10, epochs=5),
        ).json()
    }

    response = lambda_handler(event, context={})
    result: ClientResult = ClientResult.parse_raw(response["body"])
    losses = result.history["loss"]
    assert response["statusCode"] == 200
    assert losses[0] > losses[-1]


@pytest.mark.integ
def test_femnist_training_with_gcloud_handler(
    model_loader_config, remote_data_loader_config
):
    @gcloud_http_error_handler((ValidationError, ClientError))
    def gcloud_handler(request):
        body: bytes = request.get_data()
        config = ClientInvocationParams.parse_raw(body)
        return default_handler(
            data_config=config.data,
            model_config=config.model,
            hyperparams=config.hyperparams,
        )

    class RequestStub:
        def get_data(self):
            return bytes(
                ClientInvocationParams(
                    data=remote_data_loader_config,
                    model=model_loader_config,
                    hyperparams=Hyperparams(batch_size=10, epochs=5),
                ).json(),
                encoding="utf-8",
            )

    body, status, _ = gcloud_handler(RequestStub())
    result: ClientResult = ClientResult.parse_raw(body)
    losses = result.history["loss"]
    assert status == 200
    assert losses[0] > losses[-1]


@pytest.mark.integ
def test_femnist_training_with_openwhisk_handler(
    model_loader_config, remote_data_loader_config
):
    @openwhisk_action_handler((ValidationError, ClientError))
    def ow_handler(request):
        config = ClientInvocationParams.parse_obj(request["body"])

        return default_handler(
            data_config=config.data,
            model_config=config.model,
            hyperparams=config.hyperparams,
        )

    request = {
        "__ow_body": base64.b64encode(
            bytes(
                ClientInvocationParams(
                    data=remote_data_loader_config,
                    model=model_loader_config,
                    hyperparams=Hyperparams(batch_size=10, epochs=5),
                ).json(),
                encoding="utf-8",
            )
        )
    }

    response = ow_handler(request)
    result: ClientResult = ClientResult.parse_raw(response["body"])
    losses = result.history["loss"]
    assert response["statusCode"] == 200
    assert losses[0] > losses[-1]
