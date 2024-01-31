import logging

from pydantic import ValidationError

from fedless.client import ClientError, default_handler
from fedless.common.models import ClientInvocationParams
from fedless.common.providers import lambda_proxy_handler

logging.basicConfig(level=logging.DEBUG)


@lambda_proxy_handler(caught_exceptions=(ValidationError, ClientError))
def handler(event, context):
    """
    Train client on given data and model and return :class:`fedless.client.ClientResult`.
    Relies on :meth:`fedless.client.lambda_proxy_handler` decorator
    for return object conversion and error handling
    :return Response dictionary compatible with API gateway's lambda-proxy integration
    """
    config = ClientInvocationParams.parse_obj(event["body"])

    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
        test_data_config=config.test_data,
    )
