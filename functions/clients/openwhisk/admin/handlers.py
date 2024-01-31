import json
import logging
import uuid

import boto3

from fedless.common.auth import CognitoClient

from fedless.common.auth import (
    verify_invoker_token,
    fetch_cognito_public_keys,
    AuthenticationError,
    ResourceServerScope,
)
from fedless.common.providers import openwhisk_action_handler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

cached_public_keys = None


def get_token_from_request(request) -> str:
    try:
        headers = request["headers"]
        auth_header = headers["authorization"]  # Should be "Bearer xxx"
        if not auth_header.startswith("Bearer"):
            raise AuthenticationError(f"Auth header is not a bearer token")
        return auth_header.split(" ")[1]
    except KeyError as e:
        logger.error(e)
        raise AuthenticationError(e) from e


def get_public_keys(request):
    region = request["region"]
    userpool_id = request["userpool_id"]
    global cached_public_keys
    if not cached_public_keys:
        logger.info("Did not find public keys, fetching from server")
        cached_public_keys = fetch_cognito_public_keys(
            region=region, userpool_id=userpool_id
        )
    return cached_public_keys


@openwhisk_action_handler((AuthenticationError,))
def create_resource_server(request):
    print(f"Got Request: {request}")
    region = request["region"]
    userpool_id = request["userpool_id"]
    expected_client_id = request["expected_client_id"]
    aws_access_key_id = request["aws_access_key_id"]
    aws_access_key_secret = request["aws_access_key_secret"]
    invoker_client_id = request["invoker_client_id"]

    body = request["body"]
    resource_server_identifier = body.get("identifier", str(uuid.uuid4()))
    resource_server_name = body.get("name", resource_server_identifier)

    logger.info(
        {
            "region": region,
            "expected_client_id": expected_client_id,
        }
    )

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_access_key_secret,
        region_name=region,
    )
    client = session.client("cognito-idp")
    cognito = CognitoClient(
        client=client,
        user_pool_id=userpool_id,
        region_name=region,
    )

    token = get_token_from_request(request)
    public_keys = get_public_keys(request)
    logger.info(public_keys, token)
    if not verify_invoker_token(
        token=token,
        public_keys=public_keys,
        expected_client_id=expected_client_id,
        required_scope=None,
    ):
        raise AuthenticationError(f"Token invalid")

    scope_identifiers = cognito.create_resource_server(
        name=resource_server_name,
        identifier=resource_server_identifier,
        scopes=[
            ResourceServerScope(
                name="invoke", description="Invoke this resource server"
            )
        ],
    )

    # Add scope to invoker client
    for scope in scope_identifiers:
        cognito.add_scope_to_client(client_id=invoker_client_id, scope=scope)

    return json.dumps(
        {
            "name": resource_server_name,
            "identifier": resource_server_identifier,
            "region": region,
            "user_pool_id": userpool_id,
            "invoker_id": invoker_client_id,
            "scopes": scope_identifiers,
        }
    )
