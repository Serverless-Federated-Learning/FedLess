import logging
from functools import reduce
from typing import Iterator, Optional, List, Tuple

import numpy as np

from fedless.aggregator.exceptions import (
    InsufficientClientResults,
)
from fedless.common.models import Parameters, ClientResult, TestMetrics
from fedless.common.models.aggregation_models import (
    AggregationHyperParams,
    AggregationResult,
)
from fedless.common.persistence.client_daos import (
    ClientResultDao,
    ParameterDao,
)

from fedless.common.serialization import deserialize_parameters

from fedless.aggregator.parameter_aggregator import (
    ParameterAggregator,
)

logger = logging.getLogger(__name__)

# FedNova
# Tackling the Objective Inconsistency Problem
# in Heterogeneous Federated Optimization
# https://arxiv.org/abs/2007.07481


class FedNovaAggregator(ParameterAggregator):
    def __init__(
        self, current_round: int, aggregation_hyper_params: AggregationHyperParams
    ):
        super().__init__(current_round, aggregation_hyper_params)
        self.mu = aggregation_hyper_params.mu
        self.gmf = aggregation_hyper_params.gmf
        self.lr = aggregation_hyper_params.lr
        self.buf = None

    def _aggregate(
        self,
        client_parameters: List[List[np.ndarray]],
        client_cardinalities: List[float],
        client_steps: List[int],
        client_norm_vecs: List[float],
    ) -> List[np.ndarray]:

        global_weights = self.global_model
        total_sample_size = sum(client_cardinalities)

        # get accumulated gradient for each client
        def _get_cum_grads(weights):
            return [
                g_layer - l_layer for g_layer, l_layer in zip(global_weights, weights)
            ]

        client_cum_grads = [
            _get_cum_grads(client_weight) for client_weight in client_parameters
        ]

        if self.mu != 0:
            tau_eff = sum(
                [
                    (size / total_sample_size) * steps
                    for size, steps in zip(client_cardinalities, client_steps)
                ]
            )
        else:
            tau_eff = sum(
                [
                    (size / total_sample_size) * local_norm_vec
                    for size, local_norm_vec in zip(
                        client_cardinalities, client_norm_vecs
                    )
                ]
            )

        def norm_gradient(grads, tau_eff, client_norm_vec, ratio):
            scale = tau_eff / client_norm_vec * ratio
            return [scale * layer for layer in grads]

        # Normalized gradient = gradient * (tau_eff / local_norm_vec * relative sample size)
        # normalized gradients for all clients
        normalized_gradients = [
            norm_gradient(
                cum_grads,
                tau_eff,
                client_norm_vec,
                (sample_size / total_sample_size),
            )
            for cum_grads, client_norm_vec, sample_size in zip(
                client_cum_grads, client_norm_vecs, client_cardinalities
            )
        ]

        # sum normalized gradients from all clients
        total_normalized_gradients: List[np.ndarray] = [
            reduce(np.add, layer_updates)
            for layer_updates in zip(*normalized_gradients)
        ]

        if self.gmf != 0:
            # TODO: global_momentum_buffer
            if self.buf is None:
                self.buf = [layer / self.lr for layer in total_normalized_gradients]
            else:
                self.buf = [
                    buf_layer * self.gmf + (grad_layer / self.lr)
                    for buf_layer, grad_layer in zip(
                        self.buf, total_normalized_gradients
                    )
                ]

            new_global_weights = [
                np.add(layer_weights, layer_gradients)
                for layer_weights, layer_gradients in zip(
                    global_weights, (self.buf * self.lr)
                )
            ]

        else:
            # old - norm grad
            new_global_weights = [
                np.subtract(layer_weights, layer_gradients)
                for layer_weights, layer_gradients in zip(
                    global_weights, total_normalized_gradients
                )
            ]

        # apply normalized gradients to global model
        return new_global_weights

    def select_aggregation_candidates(self, mongo_client, session_id, round_id):
        parameter_dao = ParameterDao(mongo_client)
        self.global_model = deserialize_parameters(
            parameter_dao.load(session_id=session_id, round_id=round_id)
        )
        result_dao = ClientResultDao(mongo_client)
        logger.debug(f"Establishing database connection")

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
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        client_parameters: List[List[np.ndarray]] = []

        client_cardinalities: List[int] = []
        client_norm_vecs: List[float] = []
        client_steps: List[int] = []

        default_step = 0
        total_sample_size: int = 0
        tau_eff: float = 0.0

        client_metrics: List[TestMetrics] = []
        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
            del client_result.parameters
            cardinality = client_result.cardinality

            client_parameters.append(params)
            client_cardinalities.append(cardinality)
            client_norm_vecs.append(
                client_result.local_counters.get("local_normalizing_vec", 0)
            )
            client_steps.append(
                client_result.local_counters.get("local_steps", default_step)
            )

            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        return AggregationResult(
            new_global_weights=self._aggregate(
                client_parameters,
                client_cardinalities,
                client_steps,
                client_norm_vecs,
            ),
            client_metrics=client_metrics or None,
        )


class StreamFedNovaAggregator(FedNovaAggregator):
    def __init__(self, **kwargs):
        raise NotImplementedError()
