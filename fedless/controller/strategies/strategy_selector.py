from fedless.common.models import AggregationStrategy
from fedless.controller.strategies.client_selection.random_selection import (
    AsyncRandomClientSelection,
)
from fedless.controller.strategies.fedless_strategy import FedlessStrategy

from fedless.controller.strategies.client_selection import (
    RandomClientSelection,
    DBScanClientSelection,
    ScoreBasedClientSelection,
)


def select_strategy(strategy: str, invocation_attrs: dict):

    switcher = {
        "fedavg": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.FedAvg,
            **invocation_attrs,
        ),
        "fedprox": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.FedProx,
            **invocation_attrs,
        ),
        "fedlesscan": FedlessStrategy(
            selection_strategy=DBScanClientSelection(
                invocation_attrs["mongodb_config"],
                invocation_attrs["session"],
                invocation_attrs["save_dir"],
            ),
            aggregation_strategy=AggregationStrategy.FedLesScan,
            **invocation_attrs,
        ),
        "fednova": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.FedNova,
            **invocation_attrs,
        ),
        "scaffold": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.SCAFFOLD,
            **invocation_attrs,
        ),
        "fedasync": FedlessStrategy(
            selection_strategy=AsyncRandomClientSelection(
                session_id=invocation_attrs["session"],
                db_config=invocation_attrs["mongodb_config"],
            ),
            aggregation_strategy=AggregationStrategy.FedScore,
            **invocation_attrs,
        ),
        "fedscore": FedlessStrategy(
            selection_strategy=ScoreBasedClientSelection(
                session_id=invocation_attrs["session"],
                buffer_ratio=invocation_attrs["buffer_ratio"],
                db_config=invocation_attrs["mongodb_config"],
                log_dir=invocation_attrs["save_dir"],
            ),
            aggregation_strategy=AggregationStrategy.FedScore,
            **invocation_attrs,
        ),
    }

    # default to fedless strategy
    return switcher.get(strategy, switcher["fedlesscan"])
