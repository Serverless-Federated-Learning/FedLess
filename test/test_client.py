import itertools
from copy import copy
from typing import List
from unittest.mock import patch

import tensorflow as tf

from fedless.common.serialization import SerializationError, ModelLoadError
from fedless.client import (
    run,
    ClientError,
    default_handler,
)
from fedless.common.models import SerializedParameters
from fedless.datasets.dataset_loaders import DatasetNotLoadedError
from .fixtures import *
from .stubs import DatasetLoaderStub, ModelLoaderStub


@patch("fedless.client.DatasetLoaderBuilder")
@patch("fedless.client.ModelLoaderBuilder")
def test_default_handler_calls_run_correctly(
    model_builder,
    data_builder,
    hyperparams,
    dataset,
    model,
    weights_serializer,
    string_serializer,
):
    data_builder.from_config.return_value = data_loader = DatasetLoaderStub(dataset)
    model_builder.from_config.return_value = model_loader = ModelLoaderStub(model)
    with patch("fedless.client.run") as mock_run:
        # noinspection PyTypeChecker
        default_handler(
            None, None, hyperparams, None, weights_serializer, string_serializer, True
        )
        mock_run.assert_called_with(
            data_loader=data_loader,
            test_data_loader=None,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
            verbose=True,
        )


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
    ],
)
@patch("fedless.client.DatasetLoaderBuilder")
@patch("fedless.client.ModelLoaderBuilder")
@patch("fedless.client.run")
def test_default_handler_wraps_errors(
    mock_run, model_builder, data_builder, error, hyperparams
):
    # Mock loaders
    data_builder.from_config.return_value = data_loader = DatasetLoaderStub(dataset)
    model_builder.from_config.return_value = model_loader = ModelLoaderStub(model)

    # Set custom error message and make run throw it
    error_message = f"Error Message: {error.__name__}"
    mock_run.side_effect = error(error_message)

    with pytest.raises(ClientError, match=error_message):
        # noinspection PyTypeChecker
        default_handler(None, None, hyperparams, weights_serializer, string_serializer)


def test_run_returns_weights(
    model_loader, data_loader, hyperparams, weights_serializer, string_serializer
):
    result = run(
        data_loader=data_loader,
        model_loader=model_loader,
        hyperparams=hyperparams,
        weights_serializer=weights_serializer,
        string_serializer=string_serializer,
    )
    assert isinstance(result.parameters, SerializedParameters)


def test_run_batches_data(
    model_loader: ModelLoaderStub,
    data_loader: DatasetLoaderStub,
    hyperparams,
    weights_serializer,
    string_serializer,
):
    with patch.object(
        data_loader.dataset, "batch", wraps=data_loader.dataset.batch
    ) as mocked_batch:
        run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
            validation_split=0.0,
        )
        mocked_batch.assert_called_with(hyperparams.batch_size, drop_remainder=False)


@pytest.mark.parametrize(
    ["optimizer", "loss", "metrics"],
    itertools.product(
        [tf.keras.optimizers.SGD(learning_rate=0.01), tf.keras.optimizers.Adam()],
        ["sparse_categorical_crossentropy"],
        [[], ["accuracy", "mse"]],
    ),
)
def test_run_overwrites_hyperparameters(
    model_loader: ModelLoaderStub,
    data_loader: DatasetLoaderStub,
    weights_serializer,
    string_serializer,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: str,
    metrics: List[str],
):
    hyperparams = Hyperparams(
        batch_size=1,
        epochs=2,
        optimizer=tf.keras.optimizers.serialize(optimizer),
        loss=loss,
        metrics=metrics,
    )

    with patch.object(
        model_loader.model, "compile", wraps=model_loader.model.compile
    ) as mocked_compile, patch.object(tf.keras.optimizers, "get") as mock_get_optimizer:
        mock_get_optimizer.return_value = optimizer

        # Run training
        run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )

        mocked_compile.assert_called_with(
            optimizer=optimizer,
            loss=tf.keras.losses.get(loss),
            metrics=metrics,
        )


def test_run_does_not_overwrite_hyperparameters(
    model_loader: ModelLoaderStub,
    data_loader: DatasetLoaderStub,
    hyperparams,
    weights_serializer,
    string_serializer,
):
    model = model_loader.model
    original_metrics = copy(
        model.metrics
    )  # Needed as metrics are overwritten during compiling
    with patch.object(model, "compile", wraps=model.compile) as mocked_compile:
        run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )

        mocked_compile.assert_called_with(
            optimizer=model.optimizer, loss=model.loss, metrics=original_metrics
        )
