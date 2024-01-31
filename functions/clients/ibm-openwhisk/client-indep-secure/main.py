import logging
from pydantic import ValidationError

from fedless.client import (
    fedless_mongodb_handler,
    ClientError,
)
from fedless.common.models import InvokerParams
from fedless.common.providers import openwhisk_action_handler

logging.basicConfig(level=logging.DEBUG)


@openwhisk_action_handler((ValidationError, ClientError))
def main(request):
    """
    Train client on given data and model and return :class:`fedless.client.ClientResult`.
    Relies on :meth:`fedless.client.openwhisk_action_handler` decorator
    for return object conversion and error handling
    :return Response dictionary containing http response
    """
    config = InvokerParams.parse_obj(request["body"])

    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
        invocation_delay=config.invocation_delay,
    )
