import logging

from pydantic import ValidationError

from fedless.client import (
    default_handler,
    ClientError,
)
from fedless.common.models import ClientInvocationParams
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
    config = ClientInvocationParams.parse_raw(body)
    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
        test_data_config=config.test_data,
    )
