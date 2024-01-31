import logging

from pydantic import ValidationError

from fedless.evaluation import default_evaluation_handler, EvaluationError
from fedless.common.models import EvaluatorParams
from fedless.common.providers import openwhisk_action_handler

logging.basicConfig(level=logging.DEBUG)


@openwhisk_action_handler((ValidationError, EvaluationError))
def main(request):
    config = EvaluatorParams.parse_obj(request["body"])

    return default_evaluation_handler(
        database=config.database,
        session_id=config.session_id,
        round_id=config.round_id,
        test_data=config.test_data,
        batch_size=config.batch_size,
    )
