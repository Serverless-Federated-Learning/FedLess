import logging

import azure.functions
from pydantic import ValidationError

from fedless.client import ClientError, fedless_mongodb_handler
from fedless.common.models import InvokerParams
from fedless.common.providers import azure_handler

logging.basicConfig(level=logging.DEBUG)


@azure_handler(caught_exceptions=(ValidationError, ValueError, ClientError))
def main(req: azure.functions.HttpRequest):
    config = InvokerParams.parse_obj(req.get_json())

    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
        invocation_delay=config.invocation_delay,
    )
