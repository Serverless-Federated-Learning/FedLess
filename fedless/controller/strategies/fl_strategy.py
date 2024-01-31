import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from fedless.common.models.aggregation_models import AggregationStrategy
from fedless.common.models.models import TestMetrics
from fedless.controller.strategies.Intelligent_selection import ClientSelectionScheme

logger = logging.getLogger(__name__)


class FLStrategy(ABC):
    def __init__(
        self,
        clients,
        selectionStrategy: ClientSelectionScheme,
        aggregation_strategy: AggregationStrategy,  # Per round/Per Session,
    ):
        self.clients = clients
        self.selectionStrategy = selectionStrategy
        self.aggregation_strategy = aggregation_strategy

    def aggregate_metrics(self, metrics: List[TestMetrics], metric_names: Optional[List[str]] = None) -> Dict:
        if metric_names is None:
            metric_names = ["loss"]

        cardinalities, metrics = zip(*((metric.cardinality, metric.metrics) for metric in metrics))
        result_dict = {}
        for metric_name in metric_names:
            values = [metric[metric_name] for metric in metrics]
            mean = np.average(values, weights=cardinalities)
            result_dict.update(
                {
                    f"mean_{metric_name}": mean,
                    f"all_{metric_name}": values,
                    f"median_{metric_name}": np.median(values),
                }
            )
        return result_dict

    @abstractmethod
    async def fit_round(self, round: int, clients: List) -> Tuple[float, float, Dict]:
        """
        :return: (loss, accuracy, metrics) tuple
        """

    async def fit(
        self,
        n_clients_in_round: int,
        max_rounds: int,
        max_accuracy: Optional[float] = None,
    ):
        for round in range(max_rounds):
            # clients = self.sample_clients(n_clients_in_round, self.clients)
            clients = self.selectionStrategy.select_clients(n_clients_in_round, self.clients, round, max_rounds)
            logger.info(f"Sampled {len(clients)} for round {round}")
            loss, accuracy, metrics = await self.fit_round(round, clients)
            logger.info(f"Round {round} finished. Global loss={loss}, accuracy={accuracy}")

            if max_accuracy and accuracy >= max_accuracy:
                logger.info(f"Reached accuracy {accuracy} after {round + 1} rounds. Aborting...")
                break
