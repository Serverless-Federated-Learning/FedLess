import logging

from pydantic import ValidationError

from fedless.controller.invocation import (
    InvocationError,
    function_invoker_handler,
)
from fedless.common.providers import openwhisk_action_handler
from fedless.common.models import InvokerParams

logging.basicConfig(level=logging.DEBUG)


@openwhisk_action_handler((ValidationError, InvocationError))
def main(request):
    config = InvokerParams.parse_obj(request["body"])

    return function_invoker_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        invocation_id=config.invocation_id,
        client_id=config.client_id,
        database=config.database,
        http_headers=config.http_headers,
        http_proxies=config.http_proxies,
    )
