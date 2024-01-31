import abc
from typing import Iterator, List

from fedless.common.models import (
    ClientResult,
    AggregationResult,
)


class ParameterAggregator(abc.ABC):

    """
    return the dict containing client sepcs and clientresults containing the files to agregate
    """

    @abc.abstractmethod
    def select_aggregation_candidates(self, **kwargs) -> Iterator:
        pass

    @abc.abstractmethod
    def aggregate(
        self, client_results: Iterator[ClientResult], client_stats: List[dict]
    ) -> AggregationResult:
        pass
