import logging
from functools import reduce
from typing import Iterator, Optional, List, Tuple

import numpy as np
import tensorflow as tf

from fedless.aggregator.exceptions import (
    InsufficientClientResults,
    UnknownCardinalityError,
)
from fedless.common.models import Parameters, ClientResult, TestMetrics
from fedless.common.models.aggregation_models import AggregationResult
from fedless.common.persistence.client_daos import ClientResultDao, ParameterDao

from fedless.common.serialization import deserialize_parameters

from fedless.aggregator.stall_aware_parameter_aggregator import (
    StallAwareParameterAggregator,
)

logger = logging.getLogger(__name__)


class FedLesScanAggregator(StallAwareParameterAggregator):
    def _aggregate(
        self,
        client_stats: List[dict],
        client_parameters: List[List[np.ndarray]],
        client_cardinalities: List[float],
    ) -> List[np.ndarray]:

        # Default tolerance = 0 => default score = 1 for all
        client_staleness_scores = self._score_clients(client_stats)
        weighted_parameters = [
            [score * layer * num_examples for layer in weights]
            for score, weights, num_examples in zip(
                client_staleness_scores, client_parameters, client_cardinalities
            )
        ]

        num_examples_total = sum(client_cardinalities)

        # get average parameter
        avg_parameters: List[np.ndarray] = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_parameters)
        ]
        return avg_parameters

    def select_aggregation_candidates(self, mongo_client, session_id, round_id):
        result_dao = ClientResultDao(mongo_client)
        logger.debug(f"Establishing database connection")
        round_dicts, round_candidates = result_dao.load_results_for_session(
            session_id=session_id, round_id=round_id, tolerance=self.tolerance
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
        client_cardinalities: List[int] = []
        client_metrics: List[TestMetrics] = []
        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
            del client_result.parameters
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
            client_cardinalities.append(cardinality)
            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        return AggregationResult(
            new_global_weights=self._aggregate(
                client_stats, client_parameters, client_cardinalities
            ),
            client_metrics=client_metrics or None,
        )


class StreamFedLesScanAggregator(FedLesScanAggregator):
    def __init__(self, **kwargs):
        raise NotImplementedError()
