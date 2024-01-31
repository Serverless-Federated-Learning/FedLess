import logging

from fedless.common.auth import (
    verify_invoker_token,
    fetch_cognito_public_keys,
    AuthenticationError,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

cached_public_keys = None


def main(request):
    print(f"Got Request: {request}")
    region = request["region"]
    userpool_id = request["userpool_id"]
    expected_client_id = request["expected_client_id"]
    required_scope = request["required_scope"]

    # Get token from authorization header
    try:
        headers = request["__ow_headers"]
        auth_header = headers["authorization"]  # Should be "Bearer xxx"
        if not auth_header.startswith("Bearer"):
            raise AuthenticationError(f"Auth header is not a bearer token")
        token = auth_header.split(" ")[1]
    except KeyError as e:
        print(e)
        raise AuthenticationError(e) from e

    global cached_public_keys
    if not cached_public_keys:
        logger.info("Did not find public keys, fetching from server")
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

    return request
