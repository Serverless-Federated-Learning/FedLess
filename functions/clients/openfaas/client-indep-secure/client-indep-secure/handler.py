import logging

from pydantic import ValidationError

from fedless.client import (
    fedless_mongodb_handler,
    ClientError,
)
from fedless.common.models import InvokerParams
from fedless.common.providers import openfaas_action_handler

logging.basicConfig(level=logging.DEBUG)


@openfaas_action_handler(caught_exceptions=(ValidationError, ClientError))
def handle(event, context):
    config = InvokerParams.parse_raw(event.body)

    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
        invocation_delay=config.invocation_delay,
    )
