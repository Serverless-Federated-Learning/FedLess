import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np

from fedless.aggregator.exceptions import InsufficientClientResults
from fedless.aggregator.parameter_aggregator import ParameterAggregator
from fedless.common.models import Parameters, SerializedParameters, TestMetrics
from fedless.common.persistence.client_daos import (
    ClientLogitPredictionsDao,
    ClientResultDao,
    ParameterDao,
)
from fedless.common.serialization import deserialize_parameters

logger = logging.getLogger(__name__)


class FedMDAggregator(ParameterAggregator):
    def _aggregate(self, logits: List[List[np.ndarray]]) -> List[np.ndarray]:
        return np.mean(logits, axis=0)

    def select_aggregation_candidates(self, mongo_client, session_id, round_id):

        client_logit_predictions_dao = ClientLogitPredictionsDao(mongo_client)
        logger.debug(f"Establishing database connection")
        round_dicts, round_candidates = client_logit_predictions_dao.load_results_for_round(
            session_id=session_id, round_id=round_id
        )
        if not round_candidates:
            raise InsufficientClientResults(f"Found no client results for session {session_id} and round {round_id}")
        return round_dicts, round_candidates

    def aggregate(
        self, client_logits: Iterator[SerializedParameters], *args
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        client_parameters: List[List[np.ndarray]] = []

        for client_logit in client_logits:
            params = deserialize_parameters(client_logit)
            del client_logit

            client_parameters.append(params)

        return (self._aggregate(client_parameters), None)

    def perform_round_db_cleanup(self, mongo_client, session_id, round_id):
        client_logit_predictions_dao = ClientLogitPredictionsDao(mongo_client)
        client_logit_predictions_dao.delete_results_for_round(session_id, round_id)

        client_results_dao = ClientResultDao(mongo_client)
        client_results_dao.delete_results_for_round(session_id, round_id)

        global_logits_dao = ParameterDao(mongo_client)
        global_logits_dao.delete_results_for_round(session_id, round_id)
