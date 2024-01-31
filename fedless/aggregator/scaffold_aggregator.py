import logging
from functools import reduce
from typing import Iterator, Optional, List, Tuple

import numpy as np
import tensorflow as tf

from fedless.aggregator.exceptions import (
    InsufficientClientResults,
    UnknownCardinalityError,
)
from fedless.common.models import (
    Parameters,
    ClientResult,
    AggregationResult,
    TestMetrics,
)
from fedless.common.models.aggregation_models import AggregationHyperParams
from fedless.common.persistence.client_daos import ClientResultDao, ParameterDao

from fedless.common.serialization import deserialize_parameters

from fedless.aggregator.parameter_aggregator import (
    ParameterAggregator,
)

logger = logging.getLogger(__name__)

# SCAFFOLD:
# Stochastic Controlled Averaging for Federated Learning
# https://arxiv.org/abs/1910.06378


class ScaffoldAggregator(ParameterAggregator):
    def __init__(
        self, current_round: int, aggregation_hyper_params: AggregationHyperParams
    ):
        super().__init__()
        self.total_client_count = aggregation_hyper_params.total_client_count
        self.global_lr = aggregation_hyper_params.lr

    def _aggregate(
        self,
        client_parameters: List[List[np.ndarray]],
        client_cardinalities: List[float],
        client_contorl_diffs: List[List[np.ndarray]],
    ) -> List[np.ndarray]:

        total_client_count = self.total_client_count
        selected_clients_count = len(client_cardinalities)
        global_params = self.global_weights
        global_controls = self.global_controls
        global_lr = 1  # self.global_lr

        weighted_parameters = [
            [layer * num_examples for layer in weights]
            for weights, num_examples in zip(client_parameters, client_cardinalities)
        ]

        total_cardinality = sum(client_cardinalities)

        # get average parameter
        avg_parameters: List[np.ndarray] = [
            reduce(np.add, layer_updates) / total_cardinality
            for layer_updates in zip(*weighted_parameters)
        ]

        # delta_weights = [
        #     [(c_layer - g_layer) for g_layer, c_layer in zip(global_params, client_i)]
        #     for client_i in client_parameters
        # ]

        # delta_avg_weights = [
        #     reduce(np.add, layer_updates) / selected_clients_count
        #     for layer_updates in zip(*delta_weights)
        # ]

        avg_controls = [
            reduce(np.add, layer_updates) / selected_clients_count
            for layer_updates in zip(*client_contorl_diffs)
        ]

        # calc new global weights
        # x = x + lr_g * delta_x
        # new_global_weights = [
        #     global_layer + global_lr * delta_avg
        #     for global_layer, delta_avg in zip(global_params, delta_avg_weights)
        # ]

        # clac new global controls
        # c = c + |S|/N * delta_ci
        new_global_controls = [
            global_layer + (selected_clients_count / total_client_count) * delta_avg
            for global_layer, delta_avg in zip(global_controls, avg_controls)
        ]

        return (avg_parameters, new_global_controls)

    def select_aggregation_candidates(self, mongo_client, session_id, round_id):
        # load global weights/controls
        logger.debug(f"[ParameterDao] Loading global weights/controls")
        parameter_dao = ParameterDao(mongo_client)
        global_weights = parameter_dao.load(session_id=session_id, round_id=round_id)
        global_controls = parameter_dao.load_controls(
            session_id=session_id, round_id=round_id
        )
        self.global_weights = deserialize_parameters(global_weights)
        self.global_controls = deserialize_parameters(global_controls)

        # load client results
        logger.debug(f"[ClientResultDao] Loading client results..")
        result_dao = ClientResultDao(mongo_client)

        round_dicts, round_candidates = result_dao.load_results_for_round(
            session_id=session_id, round_id=round_id
        )

        if not round_candidates:
            raise InsufficientClientResults(
                f"Found no client results for session {session_id} and round {round_id}"
            )
        return round_dicts, round_candidates

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        client_stats: List[dict],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        client_parameters: List[List[np.ndarray]] = []
        client_control_diffs: List[List[np.ndarray]] = []
        client_cardinalities: List[int] = []
        client_metrics: List[TestMetrics] = []

        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
            local_controls_diff = deserialize_parameters(
                client_result.local_controls_diff
            )
            del client_result.parameters, client_result.local_controls_diff
            cardinality = client_result.cardinality

            # Check if cardinality is valid and handle accordingly
            if cardinality in [
                tf.data.UNKNOWN_CARDINALITY,
                tf.data.INFINITE_CARDINALITY,
            ]:
                if not default_cardinality:
                    raise UnknownCardinalityError(
                        f"Cardinality for client result invalid. "
                    )
                else:
                    cardinality = default_cardinality

            client_parameters.append(params)
            client_control_diffs.append(local_controls_diff)
            client_cardinalities.append(cardinality)
            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        new_global_weights, new_global_controls = self._aggregate(
            client_parameters, client_cardinalities, client_control_diffs
        )
        return AggregationResult(
            new_global_weights=new_global_weights,
            new_global_controls=new_global_controls,
            client_metrics=client_metrics or None,
        )


class StreamScaffoldAggregator(ScaffoldAggregator):
    def __init__(self, **kwargs):
        raise NotImplementedError()
