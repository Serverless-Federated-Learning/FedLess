import logging
import time
from typing import Dict, List, NamedTuple

import boto3
import requests
from botocore.client import BaseClient
from jose import JWTError, jwk, jwt
from jose.utils import base64url_decode

logger = logging.getLogger(__name__)


class AuthenticationError(BaseException):
    pass


class NotAuthorizedException(AuthenticationError):
    pass


def fetch_cognito_public_keys(region: str, userpool_id: str, session: requests.Session = None) -> List[Dict]:
    session = session or requests.Session()
    keys_url = f"https://cognito-idp.{region}.amazonaws.com/{userpool_id}/.well-known/jwks.json"
    try:
        response = session.get(keys_url).json()
        return response["keys"]
    except (requests.RequestException, ValueError, KeyError) as e:
        raise AuthenticationError(e) from e


def verify_invoker_token(
    token: str,
    public_keys: List[Dict],
    expected_client_id: str = None,
    required_scope: str = None,
):
    try:
        # In large parts based on jwt authorization template from AWS
        headers = jwt.get_unverified_headers(token)
        kid = headers["kid"]

        for public_key in public_keys:
            if kid == public_key.get("kid", ""):
                matching_public_key = public_key
                break
        else:
            logger.error("Public key unknown")
            return False
        logger.info(f"Found matching public key {matching_public_key}")
        public_key = jwk.construct(matching_public_key)
        message, encoded_signature = str(token).rsplit(".", 1)
        decoded_signature = base64url_decode(encoded_signature.encode("utf-8"))

        # Verify signature
        if not public_key.verify(message.encode("utf8"), decoded_signature):
            logging.error("Signature verification failed")
            return False

        claims = jwt.get_unverified_claims(token)
        if time.time() > claims["exp"]:
            logger.error("Token is expired")
            return False

        if expected_client_id and claims["client_id"] != expected_client_id:
            logger.error("Token was not issued for this audience")
            return False

        if required_scope and required_scope not in claims["scope"]:
            logger.error(f"Token does not include required scope {required_scope}")
            return False
    except (JWTError, ValueError, KeyError) as e:
        raise AuthenticationError(e) from e

    logger.info(f"Successfully verified token")
    return True


class ResourceServerScope(NamedTuple):
    name: str
    description: str


class CognitoClient:
    def __init__(
        self,
        user_pool_id: str,
        region_name: str,
        client: BaseClient = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
    ):
        self.user_pool_id = user_pool_id

        # Get client
        if client:
            self.client = client
        elif aws_access_key_id and aws_secret_access_key and aws_session_token:
            self.client = boto3.client(
                "cognito-idp",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
        else:
            self.client = boto3.client("cognito-idp", region_name=region_name)

    def fetch_token_for_client(
        self,
        client_id: str,
        client_secret: str,
        auth_endpoint: str,
        required_scopes: List[str],
        session: requests.Session = None,
    ) -> str:
        session = session or requests.Session()

        body = {"grant_type": "client_credentials", "scope": " ".join(required_scopes)}
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        logger.info(f"Request body for token creation: {body}")

        try:
            response = session.post(
                url=auth_endpoint,
                data=body,
                auth=(client_id, client_secret),
                headers=headers,
            ).json()
            return response["access_token"]
        except (requests.RequestException, ValueError) as e:
            raise AuthenticationError(e) from e
        except KeyError as e:
            if "error" in response:
                raise AuthenticationError(f"Auth server returned error: {response['error']}")
            raise AuthenticationError(f"Auth server did not return access token") from e

    def add_scope_to_client(self, client_id: str, scope: str):

        # "update_user_pool_client" requires previously set values to explicitely be included
        # We therefore need to fetch the current information first
        try:
            client_description = self.client.describe_user_pool_client(
                UserPoolId=self.user_pool_id,
                ClientId=client_id,
            )["UserPoolClient"]

            required_fields = [
                "ClientName",
                "RefreshTokenValidity",
                "AccessTokenValidity",
                "IdTokenValidity",
                "TokenValidityUnits",
                "ReadAttributes",
                "WriteAttributes",
                "ExplicitAuthFlows",
                "SupportedIdentityProviders",
                "CallbackURLs",
                "LogoutURLs",
                "DefaultRedirectURI",
                "AllowedOAuthFlows",
                "AllowedOAuthFlowsUserPoolClient",
                "AnalyticsConfiguration",
                "PreventUserExistenceErrors",
            ]

            # Copy over existing arguments
            custom_args = {}
            for field in required_fields:
                if field in client_description:
                    custom_args[field] = client_description[field]

            # Add new scope
            custom_args["AllowedOAuthScopes"] = client_description.get("AllowedOAuthScopes", [])
            custom_args["AllowedOAuthScopes"].append(scope)

            # Remove scopes from previously deleted resource servers, weirdly cognito doesn't remove those automatically
            resource_servers = self.client.list_resource_servers(UserPoolId=self.user_pool_id, MaxResults=50)[
                "ResourceServers"
            ]
            resource_server_identifiers = list(server["Identifier"] for server in resource_servers)
            custom_args["AllowedOAuthScopes"] = list(
                scope
                for scope in custom_args["AllowedOAuthScopes"]
                if scope.split("/")[0] in resource_server_identifiers
            )

            # Update credentials to allow invoker to also call new protected resource
            self.client.update_user_pool_client(UserPoolId=self.user_pool_id, ClientId=client_id, **custom_args)
            logger.info(
                f"Succesfully added scope {scope} to client with name "
                f"{client_description.get('ClientName')} and id {client_id}"
            )
        except (
            self.client.exceptions.ResourceNotFoundException,
            self.client.exceptions.InvalidParameterException,
            self.client.exceptions.TooManyRequestsException,
            self.client.exceptions.ResourceNotFoundException,
            self.client.exceptions.InternalErrorException,
            KeyError,
        ) as e:
            raise AuthenticationError(e) from e
        except self.client.exceptions.NotAuthorizedException as e:
            raise NotAuthorizedException(e) from e

    def create_resource_server(self, identifier: str, name: str, scopes: List[ResourceServerScope]) -> List[str]:
        """
        :return: List of strings representing the unique identifiers for the scopes of this resource-server
        """
        try:
            scopes_formatted = list(
                {"ScopeName": scope.name, "ScopeDescription": scope.description} for scope in scopes
            )
            response = self.client.create_resource_server(
                UserPoolId=self.user_pool_id,
                Identifier=identifier,
                Name=name,
                Scopes=scopes_formatted,
            )

            scope_identifiers = (
                f"{response['ResourceServer']['Identifier']}/{scope['ScopeName']}"
                for scope in response["ResourceServer"]["Scopes"]
            )
            return list(scope_identifiers)

        except (
            self.client.exceptions.ResourceNotFoundException,
            self.client.exceptions.InvalidParameterException,
            self.client.exceptions.TooManyRequestsException,
            self.client.exceptions.ResourceNotFoundException,
            self.client.exceptions.LimitExceededException,
            self.client.exceptions.InternalErrorException,
            KeyError,
        ) as e:
            raise AuthenticationError(e) from e
        except self.client.exceptions.NotAuthorizedException as e:
            raise NotAuthorizedException(e) from e
