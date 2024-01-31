import logging
from typing import Optional

import pymongo
import tensorflow as tf
from fedless.aggregator.stall_aware_parameter_aggregator import (
    StallAwareParameterAggregator,
)

from fedless.datasets.benchmark_configurator import DatasetLoaderBuilder

from fedless.common.models import (
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    AggregatorFunctionResult,
    SerializedParameters,
    TestMetrics,
    DatasetLoaderConfig,
    SerializedModel,
)
from fedless.common.models.aggregation_models import (
    AggregationHyperParams,
    AggregationStrategy,
)
from fedless.common.persistence import (
    ClientResultDao,
    ParameterDao,
    PersistenceError,
    ModelDao,
)
from fedless.common.serialization import (
    WeightsSerializerBuilder,
    SerializationError,
)

from fedless.aggregator.parameter_aggregator import ParameterAggregator
from fedless.aggregator.exceptions import AggregationError

from fedless.aggregator.fed_avg_aggregator import (
    FedAvgAggregator,
    StreamFedAvgAggregator,
)

from fedless.aggregator.fedlesscan_aggregator import (
    FedLesScanAggregator,
    StreamFedLesScanAggregator,
)

from fedless.aggregator.fedscore_aggregator import (
    FedScoreAggregator,
    StreamFedScoreAggregator,
)

from fedless.aggregator.fednova_aggregator import (
    FedNovaAggregator,
    StreamFedNovaAggregator,
)

from fedless.aggregator.scaffold_aggregator import (
    ScaffoldAggregator,
    StreamScaffoldAggregator,
)

# from fedless.aggregator import *

logger = logging.getLogger(__name__)


def get_aggregator(
    strategy: str,
    round_id: int,
    aggregation_hyper_params: AggregationHyperParams = None,
) -> ParameterAggregator:
    switcher = {
        "fedavg": FedAvgAggregator,
        "fedprox": FedAvgAggregator,
        "fedlesscan": FedLesScanAggregator,
        "fednova": FedNovaAggregator,
        "scaffold": ScaffoldAggregator,
        "fedscore": FedScoreAggregator,
    }

    switcher_online = {
        "fedavg": StreamFedAvgAggregator,
        "fedprox": StreamFedAvgAggregator,
        "fedlesscan": StreamFedLesScanAggregator,
        "fednova": StreamFedNovaAggregator,
        "scaffold": StreamScaffoldAggregator,
        "fedscore": StreamFedScoreAggregator,
    }

    aggregator_class = (
        switcher_online.get(strategy, switcher_online["fedavg"])
        if aggregation_hyper_params.aggregate_online
        else switcher.get(strategy, switcher["fedavg"])
    )

    aggregator = (
        aggregator_class(
            current_round=round_id, aggregation_hyper_params=aggregation_hyper_params
        )
        if issubclass(aggregator_class, StallAwareParameterAggregator)
        or aggregator_class == ScaffoldAggregator
        or aggregator_class == FedNovaAggregator
        else aggregator_class()
    )
    return aggregator


def default_aggregation_handler(
    session_id: str,
    round_id: int,
    database: MongodbConnectionConfig,
    serializer: WeightsSerializerConfig,
    test_data: Optional[DatasetLoaderConfig] = None,
    delete_results_after_finish: bool = True,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FedAvg,
    aggregation_hyper_params: AggregationHyperParams = None,
) -> AggregatorFunctionResult:

    mongo_client = pymongo.MongoClient(database.url)
    logger.info(
        f"Aggregator ({aggregation_strategy}) invoked for session {session_id} and round {round_id}"
    )
    try:

        result_dao = ClientResultDao(mongo_client)
        parameter_dao = ParameterDao(mongo_client)

        aggregator = get_aggregator(
            aggregation_strategy, round_id, aggregation_hyper_params
        )

        previous_dic, previous_results = aggregator.select_aggregation_candidates(
            mongo_client, session_id, round_id
        )

        logger.info(f"Aggregator got {len(previous_dic)} results for aggregation...")

        previous_results = (
            list(previous_results)
            if not isinstance(previous_results, list)
            else previous_results
        )

        logger.debug(f"Starting aggregation...")
        aggregation_result = aggregator.aggregate(previous_results, previous_dic)

        new_parameters, test_results = (
            aggregation_result.new_global_weights,
            aggregation_result.client_metrics,
        )

        logger.debug(f"Aggregation finished")

        if (
            aggregation_strategy == AggregationStrategy.SCAFFOLD
            and aggregation_result.new_global_controls
        ):
            global_controls = aggregation_result.new_global_controls
            logger.debug(f"Serializing global controls (SCAFFOLD)")
            serialized_global_controls_str = WeightsSerializerBuilder.from_config(
                serializer
            ).serialize(global_controls)

            serialized_global_controls = SerializedParameters(
                blob=serialized_global_controls_str, serializer=serializer
            )

        else:
            serialized_global_controls = None

        # aggregator
        # cleanup and count results
        count_func = result_dao.count_results_for_round
        delete_func = result_dao.delete_results_for_round
        cleanup_params = {"session_id": session_id, "round_id": round_id}

        tolerance = aggregation_hyper_params.tolerance
        if isinstance(aggregator, StallAwareParameterAggregator) and tolerance > 0:
            count_func = result_dao.count_results_for_session
            delete_func = result_dao.delete_results_for_session
            cleanup_params.pop("round_id", None)

        # only clean up process if asyn
        if not aggregation_hyper_params.is_synchronous:
            cleanup_params["tolerance_round_id"] = max(0, round_id - tolerance)
            cleanup_params["file_ids"] = [result["file_id"] for result in previous_dic]
            cleanup_params.pop("round_id", None)
            count_func = lambda *args, **kwargs: len(previous_dic)
            delete_func = result_dao.delete_results_by_file_ids

        results_processed = count_func(**cleanup_params)
        if delete_results_after_finish:
            logger.debug(f"Deleting individual results...")
            delete_func(**cleanup_params)

        global_test_metrics = None
        if test_data:
            logger.debug(f"Evaluating model")
            model_dao = ModelDao(mongo_client)
            # Load model and latest weights
            serialized_model: SerializedModel = model_dao.load(session_id=session_id)
            test_data = DatasetLoaderBuilder.from_config(test_data).load()
            cardinality = test_data.cardinality()
            test_data = test_data.batch(aggregation_hyper_params.test_batch_size)
            model: tf.keras.Model = tf.keras.models.model_from_json(
                serialized_model.model_json
            )
            model.set_weights(new_parameters)
            if not serialized_model.loss or not serialized_model.optimizer:
                raise AggregationError("If compiled=True, a loss has to be specified")
            model.compile(
                optimizer=tf.keras.optimizers.get(serialized_model.optimizer),
                loss=tf.keras.losses.get(serialized_model.loss),
                metrics=serialized_model.metrics or [],
            )
            evaluation_result = model.evaluate(test_data, return_dict=True)
            global_test_metrics = TestMetrics(
                cardinality=cardinality, metrics=evaluation_result
            )

        logger.debug(f"Serializing model")
        serialized_params_str = WeightsSerializerBuilder.from_config(
            serializer
        ).serialize(new_parameters)

        serialized_params = SerializedParameters(
            blob=serialized_params_str, serializer=serializer
        )

        new_round_id = round_id + 1
        logger.debug(f"Saving model to database")
        parameter_dao.save(
            session_id=session_id,
            round_id=new_round_id,
            params=serialized_params,
            global_controls=serialized_global_controls,
        )
        logger.debug(f"Finished...")

        return AggregatorFunctionResult(
            new_round_id=new_round_id,
            num_clients=results_processed,
            test_results=test_results,
            global_test_results=global_test_metrics,
        )
    except (SerializationError, PersistenceError) as e:
        raise AggregationError(e) from e
    finally:
        mongo_client.close()
