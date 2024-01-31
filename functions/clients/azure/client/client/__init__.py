import logging

import azure.functions
from pydantic import ValidationError

from fedless.client import ClientError, default_handler
from fedless.common.models import ClientInvocationParams
from fedless.common.providers import azure_handler

logging.basicConfig(level=logging.DEBUG)


@azure_handler(caught_exceptions=(ValidationError, ValueError, ClientError))
def main(req: azure.functions.HttpRequest):
    config = ClientInvocationParams.parse_obj(req.get_json())

    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
        test_data_config=config.test_data,
    )
