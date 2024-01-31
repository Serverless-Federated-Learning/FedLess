import logging

from pydantic import ValidationError
from fedless.controller.evaluation import EvaluationError, default_evaluation_handler


from fedless.common.models import EvaluatorParams
from fedless.common.providers import openfaas_action_handler

logging.basicConfig(level=logging.DEBUG)


@openfaas_action_handler(caught_exceptions=(ValidationError, EvaluationError))
def handle(event, context):
    config = EvaluatorParams.parse_raw(event.body)

    return default_evaluation_handler(
        database=config.database,
        session_id=config.session_id,
        round_id=config.round_id,
        test_data=config.test_data,
        batch_size=config.batch_size,
    )
