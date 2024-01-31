import logging
import math
import random
import sys
from typing import Optional
import time
import pymongo
import tensorflow as tf
from tensorflow import nest, square, linalg, multiply

# from tensorflow import print as tfprint
from tensorflow import keras as keras
from absl import app
from tensorflow.python.keras.callbacks import History
from tensorflow_privacy import (
    VectorizedDPKerasAdamOptimizer as DPKerasAdamOptimizer,
    VectorizedDPKerasAdagradOptimizer as DPKerasAdagradOptimizer,
    VectorizedDPKerasSGDOptimizer as DPKerasSGDOptimizer,
    compute_rdp,
)
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    apply_dp_sgd_analysis,
)

from timeit import default_timer as timer
from fedless.common.models.aggregation_models import AggregationStrategy
from fedless.common.models.models import (
    ClientConfig,
    ClientPersistentHistory,
    SerializedModel,
)

from fedless.datasets.dataset_loaders import (
    DatasetLoader,
    DatasetNotLoadedError,
)
from fedless.datasets.benchmark_configurator import DatasetLoaderBuilder
from fedless.common.models import (
    DatasetLoaderConfig,
    ModelLoaderConfig,
    Hyperparams,
    ClientResult,
    ClientScore,
    SerializedParameters,
    TestMetrics,
    LocalPrivacyGuarantees,
    MongodbConnectionConfig,
    SimpleModelLoaderConfig,
    ClientInvocationParams,
    InvocationResult,
    BinaryStringFormat,
)
from fedless.common.persistence.client_daos import (
    ClientConfigDao,
    ClientControlDao,
    ClientHistoryDao,
    ClientScoreDao,
    InvocationHistoryDao,
    ModelDao,
    ParameterDao,
    ClientResultDao,
)
from fedless.common.persistence.mongodb_base_connector import (
    DocumentNotLoadedException,
    PersistenceError,
)
from fedless.common.serialization import (
    ModelLoadError,
    ModelLoader,
    ModelLoaderBuilder,
    WeightsSerializer,
    StringSerializer,
    Base64StringConverter,
    NpzWeightsSerializer,
    SerializationError,
    deserialize_parameters,
)

from fedless.client.optimizers.fednova import FedNova
from fedless.client.optimizers.scaffold import Scaffold

logger = logging.getLogger(__name__)

# tensorflow eager execution if debugger is active
# if hasattr(sys, "gettrace") and sys.gettrace() is not None:
#     print("** Debugger on: enable tensorflow eager execution")
#     tf.config.run_functions_eagerly(True)
#     tf.data.experimental.enable_debug_mode()
# else:
#     print("** Debugger off: graph execution")


class ClientError(Exception):
    """Error in client code"""


def fedless_mongodb_handler(
    session_id: str,
    round_id: int,
    client_id: str,
    invocation_id: str,
    database: MongodbConnectionConfig,
    evaluate_only: bool = False,
    invocation_delay: int = 0,
) -> InvocationResult:
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """

    logger.info(
        f"handler called for session_id={session_id} round_id={round_id} client_id={client_id}"
    )

    logger.info(f"invocation delay {invocation_delay} sec for client_id={client_id}")

    # delayed execution only for training
    if invocation_delay == -1:
        # fail in both
        raise ClientError(
            "client invoked with -1 delay: simulating failed client in training and test"
        )
    elif invocation_delay == -2 and not evaluate_only:
        # fail training only
        raise ClientError(
            "client invoked with -1 delay: simulating failed client in training only"
        )
    elif not evaluate_only:
        if invocation_delay > 0:
            # sleep in training only
            time.sleep(invocation_delay)
        elif invocation_delay < -2:
            # for any number less than -1
            prob = random.uniform(0, 1)
            logger.info(f"genrating failure prob: {prob}")
            if prob < 0.5:
                raise ClientError(
                    "client failed with prob {prob}: simulating failed client"
                )

    db = pymongo.MongoClient(database.url)

    try:
        # Create daos to access database
        config_dao = ClientConfigDao(db=db)
        model_dao = ModelDao(db=db)
        parameter_dao = ParameterDao(db=db)
        controls_dao = ClientControlDao(db=db)
        result_dao = ClientResultDao(db=db)
        client_history_dao = ClientHistoryDao(db=db)
        client_score_dao = ClientScoreDao(db=db)
        invocation_history_dao = InvocationHistoryDao(db=db)

        # ---------------------- START TIMER
        start_time = timer()
        logger.debug(f"Loading model from database")
        # Load model and latest weights
        model: SerializedModel = model_dao.load(session_id=session_id)
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
        logger.debug(
            f"Model successfully loaded from database. Serialized parameters: {sys.getsizeof(latest_params.blob)} bytes"
        )

        # Load client configuration and prepare call statements
        client_config: ClientConfig = config_dao.load(client_id=client_id)

        client_params = ClientInvocationParams(
            data=client_config.data,
            model=model,
            hyperparams=client_config.hyperparams,
            test_data=client_config.test_data,
        )

        score_mode = True

        test_data_loader = (
            DatasetLoaderBuilder.from_config(client_params.test_data)
            if client_params.test_data
            else None
        )
        model_loader = ModelLoaderBuilder.from_config(client_params.model)
        if evaluate_only:
            test_data = test_data_loader.load()
            model = model_loader.load()
            cardinality = test_data.cardinality()

            # batch size for SCAFFOLD
            # batch_size = (
            #     cardinality // 5
            #     if client_config.hyperparams.strategy == AggregationStrategy.SCAFFOLD
            #     else client_config.hyperparams.batch_size
            # )
            batch_size = client_config.hyperparams.batch_size

            test_data = test_data.batch(batch_size)

            evaluation_result = model.evaluate(test_data, return_dict=True)
            test_metrics = TestMetrics(
                cardinality=cardinality, metrics=evaluation_result
            )
            return InvocationResult(
                session_id=session_id,
                round_id=round_id,
                invocation_id=invocation_id,
                client_id=client_id,
                test_metrics=test_metrics,
            )

        ### happens only in training

        if (
            client_config.hyperparams.strategy == AggregationStrategy.SCAFFOLD
        ):  # SCAFFOLD
            server_control = parameter_dao.load_controls(
                session_id=session_id, round_id=round_id
            )
            client_params.hyperparams.scaffold.server_controls = deserialize_parameters(
                server_control
            )
            try:
                local_controls = controls_dao.load(
                    session_id=session_id, client_id=client_id
                )
                client_params.hyperparams.scaffold.local_controls = (
                    deserialize_parameters(local_controls)
                )
            except DocumentNotLoadedException:  # first round
                client_params.hyperparams.scaffold.local_controls = None

        data_loader = DatasetLoaderBuilder.from_config(client_params.data)
        weights_serializer: WeightsSerializer = NpzWeightsSerializer(
            compressed=client_config.compress_model
        )
        verbose: bool = True
        logger.debug(f"Successfully loaded configs and model")
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
        # ---------------------- END TIMER

        ## add invocation delay if applicable
        elapsed_time = (
            end_time - start_time + (invocation_delay if invocation_delay > 0 else 0)
        ) / 60.0

        # invocation_history_dao
        invocation_history_dao.exec_done(invocation_id, end_time - start_time)

        if client_result.local_controls:
            logger.debug("Storing local control to database.")
            controls_dao.save(
                session_id=session_id,
                client_id=client_id,
                local_controls=client_result.local_controls,
                overwrite=True,
            )
            # clear local controls to prevent extra save in result collection
            client_result.local_controls = None

        logger.debug(f"Storing client results in database. Starting now...")
        result_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            result=client_result,
        )

        if client_params.hyperparams.strategy == "fedlesscan":
            logger.debug(f"Storing client history in database...")
            client_history: ClientPersistentHistory = client_history_dao.load(client_id)
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

        if score_mode:
            training_time = client_result.training_time + (
                invocation_delay if invocation_delay > 0 else 0
            )
            logger.debug(f"Storing client spec scores in database...")
            if client_score_dao.stats_exists(client_id):
                client_score_dao.update_stats(client_id, training_time)
            else:
                client_score = ClientScore(
                    session_id=session_id,
                    client_id=client_id,
                    training_times=[training_time],
                    cardinality=client_result.cardinality,
                    epochs=client_params.hyperparams.epochs,
                    batch_size=client_params.hyperparams.batch_size,
                )
                client_score_dao.save(client_score)

        logger.debug(f"Finished writing to database")

        return InvocationResult(
            session_id=session_id,
            round_id=round_id,
            invocation_id=invocation_id,
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
    verbose: bool = True,
) -> ClientResult:
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """
    logger.info(f"handler called with hyperparams={str(hyperparams)}")
    data_loader = DatasetLoaderBuilder.from_config(data_config)
    model_loader = ModelLoaderBuilder.from_config(model_config)
    test_data_loader = (
        DatasetLoaderBuilder.from_config(test_data_config) if test_data_config else None
    )

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


# fedprox/fednova loss function
def penalty_loss_func(local_model, global_model, mu, loss_func):
    def loss_function(y_true, y_pred):
        model_difference = nest.map_structure(
            lambda a, b: a - b, local_model.weights, global_model.weights
        )
        squared_norm = square(linalg.global_norm(model_difference))
        return loss_func(y_true, y_pred) + multiply(multiply(mu, 0.5), squared_norm)

    return loss_function


def run(
    data_loader: DatasetLoader,
    model_loader: ModelLoader,
    hyperparams: Hyperparams,
    weights_serializer: WeightsSerializer,
    string_serializer: Optional[StringSerializer] = None,
    validation_split: float = None,
    test_data_loader: DatasetLoader = None,
    verbose: bool = True,
) -> ClientResult:
    """
    Loads model and data, trains the model and returns serialized parameters wrapped as :class:`ClientResult`

    :raises DatasetNotLoadedError, ModelLoadError, RuntimeError if the model was never compiled,
     ValueError if input data is invalid or shape does not match the one expected by the model, SerializationError
    """
    # Load data and model
    logger.debug(f"Loading dataset...")
    dataset = data_loader.load()
    logger.debug(f"Finished loading dataset. Loading model...")
    model = model_loader.load()
    global_model = model_loader.load()

    strategy = hyperparams.strategy

    # check whether extra param is present
    isFedProx = strategy == AggregationStrategy.FedProx
    isFedNova = strategy == AggregationStrategy.FedNova
    isScaffold = strategy == AggregationStrategy.SCAFFOLD
    isFedlesScan = strategy == "fedlesscan"

    # Set configured optimizer if specified
    loss = keras.losses.get(hyperparams.loss) if hyperparams.loss else model.loss

    # add the custom fedprox compiler
    if isFedProx and hyperparams.fedprox.mu != 0:
        logger.info(f"using fedprox @ client loss")
        loss = penalty_loss_func(model, global_model, hyperparams.fedprox.mu, loss)

    # if isFedNova or isScaffold:
    #     hyperparams.optimizer = {
    #         "class_name": strategy,
    #         "config": {
    #             "name": strategy,
    #             "learning_rate": hyperparams.SGD_learning_rate,
    #         },
    #     }

    # with keras.utils.custom_object_scope(
    #     {AggregationStrategy.FedNova: FedNova, AggregationStrategy.SCAFFOLD: Scaffold}
    # ):
    optimizer = (
        keras.optimizers.get(hyperparams.optimizer)
        if hyperparams.optimizer
        else model.optimizer
    )

    if isFedNova:
        optimizer = FedNova(
            learning_rate=hyperparams.SGD_learning_rate, mu=hyperparams.fednova.mu
        )
        optimizer.set_global_weights(global_model.weights)

    if isScaffold:
        # optimizer = Scaffold(learning_rate=hyperparams.SGD_learning_rate)
        logger.info(
            f"using SCAFFOLD optimizer with parent optimizer: {optimizer._name}"
        )
        optimizer = Scaffold(optimizer._name, optimizer.learning_rate)
        optimizer.set_controls(global_model.weights, hyperparams.scaffold)

    logger.info(f"using optimizer {optimizer._name} @ lr={str(optimizer.lr.numpy())}")

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

        # batch size = 0.2 * data size
        train_batch_size = (
            train_dataset.cardinality() // 5 if isScaffold else hyperparams.batch_size
        )

        val_batch_size = (
            val_dataset.cardinality() // 5 if isScaffold else hyperparams.batch_size
        )

        train_dataset = train_dataset.batch(
            train_batch_size, drop_remainder=drop_remainder
        )
        val_dataset = val_dataset.batch(val_batch_size)
        train_cardinality = train_validation_split_idx
        logger.debug(
            f"Split train set into training set of size {train_cardinality} "
            f"and validation set of size {cardinality - train_cardinality}"
        )
    else:
        train_cardinality = dataset.cardinality()
        train_batch_size = (
            train_cardinality // 5 if isScaffold else hyperparams.batch_size
        )

        train_dataset = dataset.batch(train_batch_size, drop_remainder=drop_remainder)
        val_dataset = None

    privacy_guarantees: Optional[LocalPrivacyGuarantees] = None
    if hyperparams.local_privacy:
        logger.debug(
            f"Creating LDP variant of {str(hyperparams.optimizer)} with parameters {hyperparams.local_privacy}"
        )
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
            raise ValueError(
                f"No DP variant for optimizer {opt_name} found in TF Privacy..."
            )

        delta = 1.0 / (int(train_cardinality) ** 1.1)
        q = hyperparams.batch_size / train_cardinality  # q - the sampling ratio.
        if q > 1:
            raise app.UsageError("n must be larger than the batch size.")
        orders = (
            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            + list(range(5, 64))
            + [128, 256, 512]
        )
        steps = int(
            math.ceil(hyperparams.epochs * train_cardinality / hyperparams.batch_size)
        )
        eps, opt_order = apply_dp_sgd_analysis(
            q, privacy_params.noise_multiplier, steps, orders, delta
        )
        rdp = compute_rdp(
            q,
            noise_multiplier=privacy_params.noise_multiplier,
            steps=steps,
            orders=orders,
        )
        privacy_guarantees = LocalPrivacyGuarantees(
            eps=eps, delta=delta, rdp=rdp.tolist(), orders=orders, steps=steps
        )
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
            loss = keras.losses.SparseCategoricalCrossentropy(
                reduction=keras.losses.Reduction.NONE
            )
        elif loss_name == "categorical_crossentropy":
            loss = keras.losses.CategoricalCrossentropy(
                reduction=keras.losses.Reduction.NONE
            )
        elif loss_name == "mean_squared_error":
            loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
        else:
            raise ValueError(f"Unkown loss type {loss_name}")

    logger.info(f"Compiling model")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info(f"Running training")
    # Train Model
    # RuntimeError, ValueError

    start_time = timer()
    history: History = model.fit(
        train_dataset,
        epochs=hyperparams.epochs,
        shuffle=hyperparams.shuffle_data,
        validation_data=val_dataset,
        verbose=verbose,
    )
    end_time = timer()
    training_time = end_time - start_time

    test_metrics = None
    if test_data_loader:
        logger.debug(f"Test data loader found, loading it now...")
        test_dataset = test_data_loader.load()
        logger.debug(f"Running evaluation for updated model")
        metrics = model.evaluate(
            test_dataset.batch(hyperparams.batch_size), return_dict=True
        )
        test_metrics = TestMetrics(
            cardinality=test_dataset.cardinality(), metrics=metrics
        )
        logger.debug(f"Test Metrics: {str(test_metrics)}")

    weights = model.get_weights()

    logger.debug(f"Serializing model parameters")
    weights_serialized = weights_serializer.serialize(weights)
    if string_serializer:
        weights_serialized = string_serializer.to_str(weights_serialized)
    logger.info(
        f"Finished serializing model parameters of size {sys.getsizeof(weights_serialized)} bytes"
    )

    # strategy specific variable
    local_counters = optimizer.get_config() if isFedNova else None

    # Update local control for SCAFFOLD (2 option provided)
    if isScaffold:
        option = hyperparams.scaffold.option
        if option == 1:
            data = train_dataset.unbatch()
            data_size = len(list(data))
            _x, _y = data.batch(data_size).get_single_element()
            with tf.GradientTape() as tape:
                _y_pred = global_model(_x, training=False)
                _loss = global_model.compute_loss(_x, _y, _y_pred)
            local_controls = tape.gradient(_loss, global_model.weights)
            local_controls = [optimizer.lr * layer for layer in local_controls]
        else:
            local_controls = optimizer.get_new_client_controls(
                global_model.get_weights(),
                model.get_weights(),
                option=option,
            )

        old_local_controls = hyperparams.scaffold.local_controls
        local_controls_diff = (
            [new - old for new, old in zip(local_controls, old_local_controls)]
            if old_local_controls
            else local_controls
        )

        # new local control -> for client further usage
        local_controls_serialized = weights_serializer.serialize(local_controls)
        if string_serializer:
            local_controls_serialized = string_serializer.to_str(
                local_controls_serialized
            )

        local_controls = SerializedParameters(  # scaffold
            blob=local_controls_serialized,
            serializer=weights_serializer.get_config(),
            string_format=(
                string_serializer.get_format()
                if string_serializer
                else BinaryStringFormat.NONE
            ),
        )

        # local controls difference (new - old) -> for aggregator
        local_controls_diff_serialized = weights_serializer.serialize(
            local_controls_diff
        )
        if string_serializer:
            local_controls_serialized = string_serializer.to_str(local_controls_diff)

        local_controls_diff = SerializedParameters(  # scaffold
            blob=local_controls_diff_serialized,
            serializer=weights_serializer.get_config(),
            string_format=(
                string_serializer.get_format()
                if string_serializer
                else BinaryStringFormat.NONE
            ),
        )

        logger.info(
            f"Finished serializing local controls parameters of size {sys.getsizeof(local_controls_serialized)} bytes"
        )
    else:
        local_controls = None
        local_controls_diff = None

    return ClientResult(
        parameters=SerializedParameters(
            blob=weights_serialized,
            serializer=weights_serializer.get_config(),
            string_format=(
                string_serializer.get_format()
                if string_serializer
                else BinaryStringFormat.NONE
            ),
        ),
        history=history.history,
        local_counters=local_counters,  # fednova
        local_controls=local_controls,
        local_controls_diff=local_controls_diff,  # scaffold
        test_metrics=test_metrics,
        cardinality=train_cardinality,
        training_time=training_time,
        privacy_guarantees=privacy_guarantees,
    )
