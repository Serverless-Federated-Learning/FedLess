import logging

from pydantic import ValidationError

from fedless.client import (
    default_handler,
    ClientError,
)
from fedless.common.models import ClientInvocationParams
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
    config = ClientInvocationParams.parse_obj(request["body"])

    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
        test_data_config=config.test_data,
    )
