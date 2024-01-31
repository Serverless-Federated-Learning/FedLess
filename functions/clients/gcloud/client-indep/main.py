import logging

from pydantic import ValidationError

from fedless.client import ClientError, fedless_mongodb_handler
from fedless.common.models import InvokerParams
from fedless.common.providers import gcloud_http_error_handler

logging.basicConfig(level=logging.DEBUG)


@gcloud_http_error_handler((ValidationError, ClientError))
def http(request):
    """Example function running training on client and returning result (weights, history, ...)

    :param request: flask.Request object. Body has to be a serialized ClientInvocationParams
    :returns :class:`fedless.client.ClientResult` that gets wrapped by the decorator as a tuple with HTTP response
        infos. In case of an exception this tuple comprises an error code and information about the exception.
    """
    body: bytes = request.get_data()
    config = InvokerParams.parse_raw(body)
    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
        invocation_delay=config.invocation_delay,
    )
