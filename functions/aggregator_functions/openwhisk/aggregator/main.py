import logging

from pydantic import ValidationError

from fedless.aggregator.aggregation import default_aggregation_handler, AggregationError
from fedless.common.providers import openwhisk_action_handler
from fedless.common.models import AggregatorFunctionParams

logging.basicConfig(level=logging.DEBUG)


@openwhisk_action_handler((ValidationError, AggregationError))
def main(request):
    config = AggregatorFunctionParams.parse_obj(request["body"])

    return default_aggregation_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        database=config.database,
        serializer=config.serializer,
        test_data=config.test_data,
        aggregation_strategy=config.aggregation_strategy,
        aggregation_hyper_params=config.aggregation_hyper_params,
    )
