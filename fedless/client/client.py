import logging
import math
import os
import random
import sys
import time
from timeit import default_timer as timer
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pymongo
import tensorflow as tf
from absl import app

# from tensorflow import print as tfprint
from tensorflow import keras as keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import History
from tensorflow_privacy import (
    VectorizedDPKerasAdagradOptimizer as DPKerasAdagradOptimizer,
)
from tensorflow_privacy import VectorizedDPKerasAdamOptimizer as DPKerasAdamOptimizer
from tensorflow_privacy import VectorizedDPKerasSGDOptimizer as DPKerasSGDOptimizer
from tensorflow_privacy import compute_rdp
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    apply_dp_sgd_analysis,
)

from fedless.common.models import (
    BinaryStringFormat,
    ClientConfig,
    ClientInvocationParams,
    ClientResult,
    DatasetLoaderConfig,
    Hyperparams,
    InvocationResult,
    LocalPrivacyGuarantees,
    ModelLoaderConfig,
    MongodbConnectionConfig,
    SerializedParameters,
    SimpleModelLoaderConfig,
    TestMetrics,
)
from fedless.common.persistence.client_daos import (
    ClientConfigDao,
    ClientHistoryDao,
    ClientLogitPredictionsDao,
    ClientModelDao,
    ClientParameterDao,
    ClientResultDao,
    ModelDao,
    ParameterDao,
)
from fedless.common.persistence.mongodb_base_connector import PersistenceError
from fedless.common.serialization import (
    Base64StringConverter,
    ModelLoader,
    ModelLoaderBuilder,
    ModelLoadError,
    NpzWeightsSerializer,
    SerializationError,
    StringSerializer,
    WeightsSerializer,
    deserialize_parameters,
)
from fedless.datasets.dataset_loader_builder import DatasetLoaderBuilder
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError
from fedless.datasets.models import remove_last_layer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ClientError(Exception):
    """Error in client code"""


def run_fedmd_transfer_learning_private(
    session_id: str,
    round_id: int,
    client_id: str,
    client_config: ClientConfig,
    db: pymongo.MongoClient,
):
    # DAOs to access DB
    client_parameter_dao = ClientParameterDao(db=db)
    client_model_dao = ClientModelDao(db=db)

    # Load personal model
    logger.info("Loading personal model from database.")
    model = client_model_dao.load(session_id=session_id, client_id=client_id)

    # Loading latest personal parameters
    latest_weights = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)

    model = ModelLoaderConfig(
        type="simple",
        params=SimpleModelLoaderConfig(
            params=latest_weights,
            model=model.model_json,
            compiled=True,
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
        ),
    )

    logger.info(
        f"Models successfully loaded from database. Serialized parameters/weights: {sys.getsizeof(latest_weights.blob)} bytes"
    )

    # Loading public/private training dataset loader
    data_loader = DatasetLoaderBuilder.from_config(client_config.data.train_data)

    weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)
    model_loader = ModelLoaderBuilder.from_config(model)

    logger.info(f"Successfully loaded configs and model")

    # Train to convergence
    client_result = run_custom_training(
        data_loader=data_loader,
        model_loader=model_loader,
        validation_split=0.2,
        hyperparams=client_config.hyperparams.fedmd.pre_train_hyperparams,
        weights_serializer=weights_serializer,
        test_data_loader=None,
        early_stopping=True,
    )

    client_parameter_dao.save(
        session_id=session_id,
        round_id=round_id,
        client_id=client_id,
        params=client_result.parameters,
    )

    return client_result


def run_fedmd_communicate(
    session_id: str, round_id: int, client_id: str, client_config: ClientConfig, db: pymongo.MongoClient, **kwargs
):

    # DAOs to access DB
    model_dao = ClientModelDao(db=db)
    client_parameter_dao = ClientParameterDao(db=db)
    client_logit_predictions_dao = ClientLogitPredictionsDao(db=db)

    # Load model and latest weights
    logger.info(f"Loading model from database")
    model = model_dao.load(session_id=session_id, client_id=client_id)

    latest_weights = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)

    model = ModelLoaderConfig(
        type="simple",
        params=SimpleModelLoaderConfig(
            params=latest_weights,
            model=model.model_json,
            compiled=True,
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
        ),
    )

    model_loader = ModelLoaderBuilder.from_config(model)

    model = model_loader.load()

    logger.info(
        f"Models successfully loaded from database. Serialized parameters/weights: {sys.getsizeof(latest_weights.blob)} bytes"
    )

    public_alignment_data_loader = DatasetLoaderBuilder.from_config(client_config.data.public_alignment_data[round_id])
    # logit_model_loader = ModelLoaderBuilder.from_config(logit_model)

    weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)

    logger.info(f"Successfully loaded configs and model")

    # Run forward pass on public data and store logits on MongoDB
    # Load data and model
    logger.info(f"Loading dataset...")
    dataset = public_alignment_data_loader.load()
    logger.info(f"Finished loading dataset. Loading model...")
    logit_model = remove_last_layer(model)

    data = dataset.batch(client_config.hyperparams.batch_size)
    logits = logit_model.predict(data, verbose=False)

    logger.info(f"Serializing model logits")

    logits_serialized = SerializedParameters(
        blob=weights_serializer.serialize(logits),
        serializer=weights_serializer.get_config(),
        string_format=(BinaryStringFormat.NONE),
    )

    logger.info(f"Finished serializing model prediction logits of size {sys.getsizeof(logits_serialized)} bytes")

    logger.info(f"Storing client prediction logits in database...")
    client_logit_predictions_dao.save(
        session_id=session_id, client_id=client_id, round_id=round_id, logits=logits_serialized
    )
    logger.info(f"Finished writing to database")

    return ClientResult(
        parameters=logits_serialized,
        cardinality=dataset.cardinality(),
    )


def run_fedmd_digest(
    session_id: str, round_id: int, client_id: str, client_config: ClientConfig, db: pymongo.MongoClient, **kwargs
):
    # DAOs to access DB
    global_logit_predictions_dao = ParameterDao(db=db)
    client_parameter_dao = ClientParameterDao(db=db)
    model_dao = ClientModelDao(db=db)

    # Load model and latest weights
    logger.info(f"Loading model from database")
    model = model_dao.load(session_id=session_id, client_id=client_id)

    latest_weights = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)

    model = ModelLoaderConfig(
        type="simple",
        params=SimpleModelLoaderConfig(
            params=latest_weights,
            model=model.model_json,
            compiled=True,
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
        ),
    )

    model_loader = ModelLoaderBuilder.from_config(model)

    model = model_loader.load()

    logger.info(
        f"Models successfully loaded from database. Serialized parameters/weights: {sys.getsizeof(latest_weights.blob)} bytes"
    )

    public_alignment_data_loader = DatasetLoaderBuilder.from_config(client_config.data.public_alignment_data[round_id])

    weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)

    logger.info(f"Successfully loaded configs and model")

    # Run training on public dataset to approach global logit consensus
    global_logits = deserialize_parameters(global_logit_predictions_dao.load_latest(session_id=session_id))
    logger.info(f"Loading dataset...")
    dataset = public_alignment_data_loader.load()
    logger.info(f"Finished loading dataset. Loading model...")
    logit_model = remove_last_layer(model)

    train_dataset = tf.data.Dataset.from_tensor_slices((list(dataset.map(lambda x, y: x)), global_logits))
    train_cardinality = train_dataset.cardinality()

    train_data = train_dataset.batch(client_config.hyperparams.fedmd.logit_alignment_hyperparams.batch_size)

    history: History = logit_model.fit(
        train_data,
        batch_size=client_config.hyperparams.fedmd.logit_alignment_hyperparams.batch_size,
        epochs=client_config.hyperparams.fedmd.logit_alignment_hyperparams.epochs,
        shuffle=True,
        verbose=False,
    )

    logger.info(f"Serializing model parameters")
    weights_serialized = weights_serializer.serialize(logit_model.get_weights())
    logger.info(f"Finished serializing model parameters of size {sys.getsizeof(weights_serialized)} bytes")
    serialized_params = SerializedParameters(blob=weights_serialized, serializer=weights_serializer.get_config())

    # Save clients personal parameter dao
    client_parameter_dao.save(
        session_id=session_id,
        round_id=round_id,
        client_id=client_id,
        params=serialized_params,
    )

    return ClientResult(
        parameters=serialized_params,
        history=history.history,
        cardinality=train_cardinality,
    )


def run_fedmd_revisit(
    session_id: str, round_id: int, client_id: str, client_config: ClientConfig, db: pymongo.MongoClient, **kwargs
):

    # DAOs to access DB
    model_dao = ClientModelDao(db=db)
    client_parameter_dao = ClientParameterDao(db=db)

    logger.info(f"Loading model from database")

    # Load model and latest weights
    model = model_dao.load(session_id=session_id, client_id=client_id)

    latest_weights = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)

    model = ModelLoaderConfig(
        type="simple",
        params=SimpleModelLoaderConfig(
            params=latest_weights,
            model=model.model_json,
            compiled=True,
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics,
        ),
    )

    logger.info(
        f"Models successfully loaded from database. Serialized parameters/weights: {sys.getsizeof(latest_weights.blob)} bytes"
    )

    private_train_data_loader = DatasetLoaderBuilder.from_config(client_config.data.train_data)

    weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)

    model_loader = ModelLoaderBuilder.from_config(model)

    # Train on private data for few epochs
    logger.info("Training for few epochs on private data")
    client_result = run_custom_training(
        data_loader=private_train_data_loader,
        model_loader=model_loader,
        hyperparams=client_config.hyperparams.fedmd.private_training_hyperparams,
        weights_serializer=weights_serializer,
        test_data_loader=None,
    )

    # Save clients personal parameter dao
    client_parameter_dao.save(
        session_id=session_id,
        round_id=round_id,
        client_id=client_id,
        params=client_result.parameters,
    )

    return client_result


def fedmd_mongodb_handler(
    session_id: str,
    round_id: int,
    client_id: str,
    database: MongodbConnectionConfig,
    evaluate_only: bool = False,
    action: str = "transfer_learning_public",
    invocation_delay: int = 0,
):

    db = pymongo.MongoClient(database.url)

    try:
        # DAOs to access DB
        config_dao = ClientConfigDao(db=db)
        results_dao = ClientResultDao(db=db)
        client_history_dao = ClientHistoryDao(db=db)

        start_time = timer()

        # Load client configuration and prepare call statements
        client_config = config_dao.load(client_id=client_id)

        # Does not move to training if this condition is true
        if evaluate_only:
            model_dao = ClientModelDao(db=db)
            client_parameter_dao = ClientParameterDao(db=db)

            model = model_dao.load(session_id=session_id, client_id=client_id)
            latest_weights = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)

            model = ModelLoaderConfig(
                type="simple",
                params=SimpleModelLoaderConfig(
                    params=latest_weights,
                    model=model.model_json,
                    compiled=True,
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics,
                ),
            )
            model_loader = ModelLoaderBuilder.from_config(model)
            private_test_data_loader = DatasetLoaderBuilder.from_config(client_config.data.test_data)

            test_data = private_test_data_loader.load()
            model = model_loader.load()
            cardinality = test_data.cardinality()

            test_data = test_data.batch(client_config.hyperparams.batch_size)

            evaluation_result = model.evaluate(test_data, return_dict=True)
            test_metrics = TestMetrics(cardinality=cardinality, metrics=evaluation_result)
            return InvocationResult(
                session_id=session_id,
                round_id=round_id,
                client_id=client_id,
                test_metrics=test_metrics,
            )

        params = {
            "session_id": session_id,
            "round_id": round_id,
            "client_id": client_id,
            "client_config": client_config,
            "db": db,
        }

        switcher = {
            "transfer_learning_private": run_fedmd_transfer_learning_private,
            "communicate": run_fedmd_communicate,
            "digest": run_fedmd_digest,
            "revisit": run_fedmd_revisit,
        }

        client_result = switcher[action](**params)

        end_time = timer()
        elapsed_time = (end_time - start_time + (invocation_delay if invocation_delay > 0 else 0)) / 60.0

        logger.info(f"Storing client results in database. Starting now...")
        results_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            result=client_result,
        )

        logger.info(f"Storing client history in database...")
        client_history = client_history_dao.load(client_id)
        client_history.training_times.append(elapsed_time)
        # save the dataset cardinality if
        if not (
            (client_result.cardinality == tf.data.INFINITE_CARDINALITY)
            or (client_result.cardinality == tf.data.UNKNOWN_CARDINALITY)
        ):
            client_history.train_cardinality = client_result.cardinality
        # remove the round if declared missed round
        # the missed round is not removed in case the client did not finish and save the results
        if round_id in client_history.missed_rounds:
            client_history.missed_rounds.remove(round_id)
        client_history_dao.save(client_history)

        logger.info(f"Finished writing to database")

        return InvocationResult(session_id=session_id, round_id=round_id, client_id=client_id)

    except (
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
        PersistenceError,
    ) as e:
        raise ClientError(e) from e
    finally:
        db.close()


def run_custom_training(
    data_loader: DatasetLoader,
    model_loader: ModelLoader,
    hyperparams: Hyperparams,
    weights_serializer: WeightsSerializer,
    string_serializer: Optional[StringSerializer] = None,
    validation_split: float = None,
    test_data_loader: DatasetLoader = None,
    early_stopping: bool = False,
) -> ClientResult:

    # Load data and model
    logger.info(f"Loading dataset...")
    dataset = data_loader.load()
    logger.info(f"Finished loading dataset. Loading model...")
    model = model_loader.load()
    # Set configured optimizer if specified
    loss = keras.losses.get(hyperparams.loss) if hyperparams.loss else model.loss
    optimizer = keras.optimizers.get(hyperparams.optimizer) if hyperparams.optimizer else model.optimizer
    metrics = (
        hyperparams.metrics or model.compiled_metrics.metrics
    )  # compiled_metrics are explicitly defined by the user

    # Batch data, necessary or model fitting will fail
    drop_remainder = bool(
        hyperparams.local_privacy and hyperparams.local_privacy.num_microbatches
    )  # if #samples % batchsize != 0, tf-privacy throws an error during training
    if validation_split:
        cardinality = float(dataset.cardinality())
        train_validation_split_idx = int(cardinality - cardinality * validation_split)
        train_dataset = dataset.take(train_validation_split_idx)
        val_dataset = dataset.skip(train_validation_split_idx)
        train_dataset = train_dataset.batch(hyperparams.batch_size, drop_remainder=drop_remainder)
        val_dataset = val_dataset.batch(hyperparams.batch_size)
        train_cardinality = train_validation_split_idx
        logger.info(
            f"Split train set into training set of size {train_cardinality} "
            f"and validation set of size {cardinality - train_cardinality}"
        )
    else:
        train_dataset = dataset.batch(hyperparams.batch_size, drop_remainder=drop_remainder)
        train_cardinality = dataset.cardinality()
        val_dataset = None

    privacy_guarantees: Optional[LocalPrivacyGuarantees] = None
    if hyperparams.local_privacy:
        logger.info(f"Creating LDP variant of {str(hyperparams.optimizer)} with parameters {hyperparams.local_privacy}")
        privacy_params = hyperparams.local_privacy
        opt_config = optimizer.get_config()
        opt_name = opt_config.get("name", "unknown")
        if opt_name == "Adam":
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        elif opt_name == "Adagrad":
            optimizer = DPKerasAdagradOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        elif opt_name == "SGD":
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        else:
            raise ValueError(f"No DP variant for optimizer {opt_name} found in TF Privacy...")

    logger.info(f"Compiling model")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(f"Running training")
    # Train Model
    # RuntimeError, ValueError

    history: History = model.fit(
        train_dataset,
        epochs=hyperparams.epochs,
        shuffle=hyperparams.is_shuffle,
        validation_data=val_dataset,
        verbose=hyperparams.verbose,
        callbacks=[EarlyStopping(monitor="accuracy", min_delta=hyperparams.min_delta, patience=hyperparams.patience)]
        if early_stopping
        else None,
    )

    test_metrics = None
    if test_data_loader:
        logger.info(f"Test data loader found, loading it now...")
        test_dataset = test_data_loader.load()
        logger.info(f"Running evaluation for updated model")
        metrics = model.evaluate(test_dataset.batch(hyperparams.batch_size), return_dict=True)
        test_metrics = TestMetrics(cardinality=test_dataset.cardinality(), metrics=metrics)
        logger.info(f"Test Metrics: {str(test_metrics)}")

    # serialization error
    logger.info(f"Serializing model parameters")
    weights_serialized = weights_serializer.serialize(model.get_weights())
    if string_serializer:
        weights_serialized = string_serializer.to_str(weights_serialized)
    logger.info(f"Finished serializing model parameters of size {sys.getsizeof(weights_serialized)} bytes")

    return ClientResult(
        parameters=SerializedParameters(
            blob=weights_serialized,
            serializer=weights_serializer.get_config(),
            string_format=(string_serializer.get_format() if string_serializer else BinaryStringFormat.NONE),
        ),
        history=history.history,
        test_metrics=test_metrics,
        cardinality=train_cardinality,
        privacy_guarantees=privacy_guarantees,
    )


def run(
    data_loader: DatasetLoader,
    model_loader: ModelLoader,
    hyperparams: Hyperparams,
    weights_serializer: WeightsSerializer,
    string_serializer: Optional[StringSerializer] = None,
    validation_split: float = None,
    test_data_loader: DatasetLoader = None,
    verbose: bool = False,
) -> ClientResult:
    """
    Loads model and data, trains the model and returns serialized parameters wrapped as :class:`ClientResult`

    :raises DatasetNotLoadedError, ModelLoadError, RuntimeError if the model was never compiled,
     ValueError if input data is invalid or shape does not match the one expected by the model, SerializationError
    """
    # Load data and model
    logger.info(f"Loading dataset...")
    dataset = data_loader.load()
    logger.info(f"Finished loading dataset. Loading model...")
    model = model_loader.load()
    # Set configured optimizer if specified
    loss = keras.losses.get(hyperparams.loss) if hyperparams.loss else model.loss
    optimizer = keras.optimizers.get(hyperparams.optimizer) if hyperparams.optimizer else model.optimizer
    metrics = (
        hyperparams.metrics or model.compiled_metrics.metrics
    )  # compiled_metrics are explicitly defined by the user

    # Batch data, necessary or model fitting will fail
    drop_remainder = bool(
        hyperparams.local_privacy and hyperparams.local_privacy.num_microbatches
    )  # if #samples % batchsize != 0, tf-privacy throws an error during training
    if validation_split:
        cardinality = float(dataset.cardinality())
        train_validation_split_idx = int(cardinality - cardinality * validation_split)
        train_dataset = dataset.take(train_validation_split_idx)
        val_dataset = dataset.skip(train_validation_split_idx)
        train_dataset = train_dataset.batch(hyperparams.batch_size, drop_remainder=drop_remainder)
        val_dataset = val_dataset.batch(hyperparams.batch_size)
        train_cardinality = train_validation_split_idx
        logger.info(
            f"Split train set into training set of size {train_cardinality} "
            f"and validation set of size {cardinality - train_cardinality}"
        )
    else:
        train_dataset = dataset.batch(hyperparams.batch_size, drop_remainder=drop_remainder)
        train_cardinality = dataset.cardinality()
        val_dataset = None

    privacy_guarantees: Optional[LocalPrivacyGuarantees] = None
    if hyperparams.local_privacy:
        logger.info(f"Creating LDP variant of {str(hyperparams.optimizer)} with parameters {hyperparams.local_privacy}")
        privacy_params = hyperparams.local_privacy
        opt_config = optimizer.get_config()
        opt_name = opt_config.get("name", "unknown")
        if opt_name == "Adam":
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        elif opt_name == "Adagrad":
            optimizer = DPKerasAdagradOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        elif opt_name == "SGD":
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        else:
            raise ValueError(f"No DP variant for optimizer {opt_name} found in TF Privacy...")

        delta = 1.0 / (int(train_cardinality) ** 1.1)
        q = hyperparams.batch_size / train_cardinality  # q - the sampling ratio.
        if q > 1:
            raise app.UsageError("n must be larger than the batch size.")
        orders = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5] + list(range(5, 64)) + [128, 256, 512]
        steps = int(math.ceil(hyperparams.epochs * train_cardinality / hyperparams.batch_size))
        eps, opt_order = apply_dp_sgd_analysis(q, privacy_params.noise_multiplier, steps, orders, delta)
        rdp = compute_rdp(
            q,
            noise_multiplier=privacy_params.noise_multiplier,
            steps=steps,
            orders=orders,
        )
        privacy_guarantees = LocalPrivacyGuarantees(eps=eps, delta=delta, rdp=rdp.tolist(), orders=orders, steps=steps)
        f"Calculated privacy guarantees: {str(privacy_guarantees)}"

        # Manually set loss' reduction method to None to support per-example loss calculation
        # Required to enable different microbatch sizes
        loss_serialized = keras.losses.serialize(keras.losses.get(loss))
        loss_name = (
            loss_serialized
            if isinstance(loss_serialized, str)
            else loss_serialized.get("config", dict()).get("name", "unknown")
        )
        if loss_name == "sparse_categorical_crossentropy":
            loss = keras.losses.SparseCategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)
        elif loss_name == "categorical_crossentropy":
            loss = keras.losses.CategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)
        elif loss_name == "mean_squared_error":
            loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
        else:
            raise ValueError(f"Unkown loss type {loss_name}")

    logger.info(f"Compiling model")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(f"Running training")
    # Train Model
    # RuntimeError, ValueError
    history: History = model.fit(
        train_dataset,
        epochs=hyperparams.epochs,
        shuffle=hyperparams.shuffle_data,
        validation_data=val_dataset,
        verbose=verbose,
    )

    test_metrics = None
    if test_data_loader:
        logger.info(f"Test data loader found, loading it now...")
        test_dataset = test_data_loader.load()
        logger.info(f"Running evaluation for updated model")
        metrics = model.evaluate(test_dataset.batch(hyperparams.batch_size), return_dict=True)
        test_metrics = TestMetrics(cardinality=test_dataset.cardinality(), metrics=metrics)
        logger.info(f"Test Metrics: {str(test_metrics)}")

    # serialization error
    logger.info(f"Serializing model parameters")
    weights_serialized = weights_serializer.serialize(model.get_weights())
    if string_serializer:
        weights_serialized = string_serializer.to_str(weights_serialized)
    logger.info(f"Finished serializing model parameters of size {sys.getsizeof(weights_serialized)} bytes")

    return ClientResult(
        parameters=SerializedParameters(
            blob=weights_serialized,
            serializer=weights_serializer.get_config(),
            string_format=(string_serializer.get_format() if string_serializer else BinaryStringFormat.NONE),
        ),
        history=history.history,
        test_metrics=test_metrics,
        cardinality=train_cardinality,
        privacy_guarantees=privacy_guarantees,
    )


def feddf_mongodb_handler(
    session_id: str,
    round_id: int,
    client_id: str,
    database: MongodbConnectionConfig,
    evaluate_only: bool = False,
    invocation_delay: int = 0,
):
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """

    db = pymongo.MongoClient(database.url)

    try:
        # Create daos to access database
        config_dao = ClientConfigDao(db=db)
        client_model_dao = ClientModelDao(db=db)
        client_parameter_dao = ClientParameterDao(db=db)
        results_dao = ClientResultDao(db=db)
        client_history_dao = ClientHistoryDao(db=db)

        start_time = timer()
        logger.info(f"Loading model from database")
        # Load model and latest weights
        model = client_model_dao.load(session_id=session_id, client_id=client_id)
        latest_params = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)
        model = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=latest_params,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )
        logger.info(
            f"Model successfully loaded from database. Serialized parameters: {sys.getsizeof(latest_params.blob)} bytes"
        )

        # Load client configuration and prepare call statements
        client_config = config_dao.load(client_id=client_id)
        client_params = ClientInvocationParams(
            data=client_config.data,
            model=model,
            hyperparams=client_config.hyperparams,
        )

        test_data_loader = (
            DatasetLoaderBuilder.from_config(client_params.data.test_data) if client_params.data.test_data else None
        )
        model_loader = ModelLoaderBuilder.from_config(client_params.model)
        if evaluate_only:
            test_data = test_data_loader.load()
            model = model_loader.load()
            cardinality = test_data.cardinality()

            test_data = test_data.batch(client_config.hyperparams.batch_size)

            evaluation_result = model.evaluate(test_data, return_dict=True)
            test_metrics = TestMetrics(cardinality=cardinality, metrics=evaluation_result)
            return InvocationResult(
                session_id=session_id,
                round_id=round_id,
                client_id=client_id,
                test_metrics=test_metrics,
            )
        ### happens only in training
        data_loader = DatasetLoaderBuilder.from_config(client_params.data.train_data)
        weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)
        verbose: bool = True
        logger.info(f"Successfully loaded configs and model")
        client_result = run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=client_params.hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=None,
            test_data_loader=None,
            verbose=verbose,
        )

        end_time = timer()
        ## add invocation delay if applicable
        elapsed_time = (end_time - start_time + (invocation_delay if invocation_delay > 0 else 0)) / 60.0

        logger.info(f"Storing client results in database. Starting now...")
        results_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            result=client_result,
        )
        logger.info(f"Storing client history in database...")
        client_history = client_history_dao.load(client_id)
        client_history.training_times.append(elapsed_time)
        # save the dataset cardinality if
        if not (
            (client_result.cardinality == tf.data.INFINITE_CARDINALITY)
            or (client_result.cardinality == tf.data.UNKNOWN_CARDINALITY)
        ):
            client_history.train_cardinality = client_result.cardinality
        # remove the round if declared missed round
        # the missed round is not removed in case the client did not finish and save the results
        if round_id in client_history.missed_rounds:
            client_history.missed_rounds.remove(round_id)
        client_history_dao.save(client_history)

        logger.info(f"Finished writing to database")

        return InvocationResult(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
        )

    except (
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
        PersistenceError,
    ) as e:
        raise ClientError(e) from e
    finally:
        db.close()


def fedless_mongodb_handler(
    session_id: str,
    round_id: int,
    client_id: str,
    database: MongodbConnectionConfig,
    evaluate_only: bool = False,
    invocation_delay: int = 0,
):
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """

    db = pymongo.MongoClient(database.url)

    try:
        # Create daos to access database
        config_dao = ClientConfigDao(db=db)
        model_dao = ModelDao(db=db)
        parameter_dao = ParameterDao(db=db)
        results_dao = ClientResultDao(db=db)
        client_history_dao = ClientHistoryDao(db=db)

        start_time = timer()
        logger.info(f"Loading model from database")
        # Load model and latest weights
        model = model_dao.load(session_id=session_id)
        latest_params = parameter_dao.load_latest(session_id)
        model = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=latest_params,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )
        logger.info(
            f"Model successfully loaded from database. Serialized parameters: {sys.getsizeof(latest_params.blob)} bytes"
        )

        # Load client configuration and prepare call statements
        client_config = config_dao.load(client_id=client_id)
        client_params = ClientInvocationParams(
            data=client_config.data,
            model=model,
            hyperparams=client_config.hyperparams,
            test_data=client_config.test_data,
        )

        test_data_loader = (
            DatasetLoaderBuilder.from_config(client_params.test_data) if client_params.test_data else None
        )
        model_loader = ModelLoaderBuilder.from_config(client_params.model)
        if evaluate_only:
            test_data = test_data_loader.load()
            model = model_loader.load()
            cardinality = test_data.cardinality()

            test_data = test_data.batch(client_config.hyperparams.batch_size)

            evaluation_result = model.evaluate(test_data, return_dict=True)
            test_metrics = TestMetrics(cardinality=cardinality, metrics=evaluation_result)
            return InvocationResult(
                session_id=session_id,
                round_id=round_id,
                client_id=client_id,
                test_metrics=test_metrics,
            )
        ### happens only in training
        data_loader = DatasetLoaderBuilder.from_config(client_params.data)
        weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)
        verbose: bool = True
        logger.info(f"Successfully loaded configs and model")
        client_result = run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=client_params.hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=None,
            test_data_loader=None,
            verbose=verbose,
        )

        end_time = timer()
        ## add invocation delay if applicable
        elapsed_time = (end_time - start_time + (invocation_delay if invocation_delay > 0 else 0)) / 60.0

        logger.info(f"Storing client results in database. Starting now...")
        results_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            result=client_result,
        )
        logger.info(f"Storing client history in database...")
        client_history = client_history_dao.load(client_id)
        client_history.training_times.append(elapsed_time)
        # save the dataset cardinality if
        if not (
            (client_result.cardinality == tf.data.INFINITE_CARDINALITY)
            or (client_result.cardinality == tf.data.UNKNOWN_CARDINALITY)
        ):
            client_history.train_cardinality = client_result.cardinality
        # remove the round if declared missed round
        # the missed round is not removed in case the client did not finish and save the results
        if round_id in client_history.missed_rounds:
            client_history.missed_rounds.remove(round_id)
        client_history_dao.save(client_history)

        logger.info(f"Finished writing to database")

        return InvocationResult(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
        )

    except (
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
        PersistenceError,
    ) as e:
        raise ClientError(e) from e
    finally:
        db.close()


def default_handler(
    data_config: DatasetLoaderConfig,
    model_config: ModelLoaderConfig,
    hyperparams: Hyperparams,
    test_data_config: DatasetLoaderConfig = None,
    weights_serializer: WeightsSerializer = NpzWeightsSerializer(),
    string_serializer: Optional[StringSerializer] = Base64StringConverter,
    verbose: bool = False,
) -> ClientResult:
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """
    logger.info(f"handler called with hyperparams={str(hyperparams)}")
    data_loader = DatasetLoaderBuilder.from_config(data_config)
    model_loader = ModelLoaderBuilder.from_config(model_config)
    test_data_loader = DatasetLoaderBuilder.from_config(test_data_config) if test_data_config else None

    try:
        return run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
            test_data_loader=test_data_loader,
            verbose=verbose,
        )
    except (
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
    ) as e:
        raise ClientError(e) from e


def master_handler(
    session_id: str,
    round_id: int,
    client_id: str,
    database: MongodbConnectionConfig,
    evaluate_only: bool = False,
    invocation_delay: int = 0,
    algorithm: str = "feddf",
    action: str = None,
):

    logger.info(
        f"handler called for session_id={session_id} round_id={round_id} client_id={client_id} algorithm={algorithm}"
    )

    logger.info(f"invocation delay {invocation_delay} sec for client_id={client_id}")

    # delayed execution only for training
    if invocation_delay == -1:
        # fail in both
        raise ClientError("client invoked with -1 delay: simulating failed client in training and test")
    elif invocation_delay == -2 and not evaluate_only:
        # fail training only
        raise ClientError("client invoked with -1 delay: simulating failed client in training only")
    elif not evaluate_only:
        if invocation_delay > 0:
            # sleep in training only
            time.sleep(invocation_delay)
        elif invocation_delay < -2:
            # for any number less than -1
            prob = random.uniform(0, 1)
            logger.info(f"genrating failure prob: {prob}")
            if prob < 0.5:
                raise ClientError("client failed with prob {prob}: simulating failed client")

    if algorithm == "fedmd":
        return fedmd_mongodb_handler(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            database=database,
            evaluate_only=evaluate_only,
            action=action,
            invocation_delay=invocation_delay,
        )

    elif algorithm == "feddf":
        return feddf_mongodb_handler(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            database=database,
            evaluate_only=evaluate_only,
            invocation_delay=invocation_delay,
        )

    elif algorithm == "fedless":
        return fedless_mongodb_handler(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            database=database,
            evaluate_only=evaluate_only,
            invocation_delay=invocation_delay,
        )

    else:
        raise NotImplementedError("Provided algorithm has not yet been implemented")
