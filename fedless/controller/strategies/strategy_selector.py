from fedless.common.models import AggregationStrategy
from fedless.controller.strategies.feddf_strategy import FedDFStrategy
from fedless.controller.strategies.fedless_strategy import FedlessStrategy
from fedless.controller.strategies.fedmd_strategy import FedMDStrategy
from fedless.controller.strategies.Intelligent_selection import (
    DBScanClientSelection,
    DBScanModelClientSelection,
    RandomClientSelection,
)


def select_strategy(strategy: str, invocation_attrs: dict):
    switcher = {
        "fedlesscan": FedlessStrategy(
            selection_strategy=DBScanClientSelection(
                invocation_attrs["mongodb_config"],
                invocation_attrs["session"],
                invocation_attrs["save_dir"],
            ),
            aggregation_strategy=AggregationStrategy.PER_SESSION,
            **invocation_attrs,
        ),
        "fedavg": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.PER_ROUND,
            **invocation_attrs,
        ),
        "fedprox": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.PER_ROUND,
            **invocation_attrs,
        ),
        "fedmd": FedMDStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.FED_MD,
            **invocation_attrs,
        ),
        "feddf": FedDFStrategy(
            selection_strategy=DBScanModelClientSelection(
                invocation_attrs["mongodb_config"],
                invocation_attrs["session"],
                invocation_attrs["save_dir"],
            ),
            aggregation_strategy=AggregationStrategy.FED_DF,
            **invocation_attrs,
        ),
    }

    # default to fedless strategy
    return switcher.get(strategy, switcher["fedmd"])
