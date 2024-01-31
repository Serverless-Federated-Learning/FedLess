import os
import logging

import azure.functions
from pydantic import ValidationError

from fedless.common.auth import (
    AuthenticationError,
    fetch_cognito_public_keys,
    verify_invoker_token,
)
from fedless.common.providers import azure_handler
from fedless.common.models import InvokerParams
from fedless.client import fedless_mongodb_handler, ClientError

logging.basicConfig(level=logging.DEBUG)

cached_public_keys = None


@azure_handler(
    caught_exceptions=(ValidationError, ValueError, ClientError, AuthenticationError)
)
def main(req: azure.functions.HttpRequest):
    try:
        region = os.environ.get("COGNITO_USER_POOL_REGION")
        userpool_id = os.environ.get("COGNITO_USER_POOL_ID")
        expected_client_id = os.environ.get("COGNITO_INVOKER_CLIENT_ID")
        required_scope = os.environ.get("COGNITO_REQUIRED_SCOPE")
        auth_header = req.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer"):
            raise AuthenticationError(f"Auth header is not a bearer token")
        token = auth_header.split(" ")[1]
        global cached_public_keys
        if not cached_public_keys:
            print("Did not find public keys, fetching from server")
            cached_public_keys = fetch_cognito_public_keys(
                region=region, userpool_id=userpool_id
            )

        if not verify_invoker_token(
            token=token,
            public_keys=cached_public_keys,
            expected_client_id=expected_client_id,
            required_scope=required_scope,
        ):
            raise AuthenticationError(f"Token invalid")
    except KeyError as e:
        print(e)
        raise AuthenticationError(e) from e

    config = InvokerParams.parse_obj(req.get_json())

    return fedless_mongodb_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        evaluate_only=config.evaluate_only,
        invocation_delay=config.invocation_delay,
    )
