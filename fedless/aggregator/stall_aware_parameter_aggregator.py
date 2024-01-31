import logging
from typing import List


from fedless.common.models.aggregation_models import (
    AggregationHyperParams,
)


from fedless.aggregator.parameter_aggregator import ParameterAggregator

logger = logging.getLogger(__name__)


class StallAwareParameterAggregator(ParameterAggregator):
    def __init__(
        self, current_round: int, aggregation_hyper_params: AggregationHyperParams
    ):
        self.is_synchronous = aggregation_hyper_params.is_synchronous
        self.current_round = current_round
        self.tolerance = (
            aggregation_hyper_params.tolerance
            if aggregation_hyper_params is not None
            else 0
        )
        super().__init__()

    # stale score => [ 0.5, 0.33, 0.25, 0.2, 0.16, ... ]
    def _score_clients(self, client_results: List[dict]):
        scores = map(
            lambda client_dict: (client_dict["round_id"] + 1)
            / (self.current_round + 1),
            client_results,
        )
        return list(scores)

    # stale score => [ 0.7, 0.57, 0.5, 0.44, 0.4, ... ]
    def _asyn_staleness_score(self, client_stats: List[dict]):

        scores = map(
            lambda client_dict: 1
            / (self.current_round - client_dict["round_id"] + 1) ** 0.5,
            client_stats,
        )
        return list(scores)
