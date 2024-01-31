import base64
import binascii
import json
import logging
import subprocess
import traceback
from json.decoder import JSONDecodeError
from typing import Callable, Dict, Iterable, List, Tuple, Type, Union

import azure.functions
import pydantic

logger = logging.getLogger(__name__)


def create_http_success_response(body: str, status: int = 200):
    """Creates successful response compatible with API gateway lambda-proxy integration"""
    return {
        "statusCode": status,
        "body": body,
        "headers": {"Content-Type": "application/json"},
    }


def format_exception_for_user(exception: Exception) -> Dict:
    """Create dictionary with information about the exception to be returned to a user"""
    return {
        "errorMessage": str(exception),
        "errorType": str(exception.__class__.__name__),
        "details": traceback.format_exc(),
    }


def create_http_user_error_response(exception: Exception, status: int = 400):
    """Create error response for given exception. Compatible with API gateway lambda-proxy integration"""
    return {
        "statusCode": status,
        "body": json.dumps(format_exception_for_user(exception)),
        "headers": {"Content-Type": "application/json"},
    }


def create_gcloud_http_success_response(
    body: str, status: int = 200
) -> Tuple[Union[str, bytes, dict], int, Union[Dict, List]]:
    """
    Create object that can be converted into Flask response object by Google Cloud API
    See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.make_response for more info
    """
    return body, status, {"Content-Type": "application/json"}


def create_gcloud_http_user_error_response(
    exception: Exception, status: int = 400
) -> Tuple[Union[str, bytes, dict], int, Union[Dict, List]]:
    """
    Create object from exception that can be converted into Flask response object by Google Cloud API
    See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.make_response for more info
    """
    return (
        json.dumps(format_exception_for_user(exception)),
        status,
        {"Content-Type": "application/json"},
    )


def create_azure_success_response(body: str, status: int = 200) -> azure.functions.HttpResponse:
    return azure.functions.HttpResponse(body=body, status_code=status, headers={"Content-Type": "application/json"})


def create_azure_user_error_response(exception: Exception, status: int = 400) -> azure.functions.HttpResponse:
    return azure.functions.HttpResponse(
        body=json.dumps(format_exception_for_user(exception)),
        status_code=status,
        headers={"Content-Type": "application/json"},
    )


def azure_handler(
    caught_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[Callable], Callable]:
    """Azure function compatible decorator to parse input, catch certain exceptions and respond to them with 400 errors."""

    def decorator(func):
        def patched_func(req: azure.functions.HttpRequest):
            try:
                # try:
                #     body = req.get_json()
                # except ValueError as e:
                #     return create_azure_user_error_response(e)
                result: Union[pydantic.BaseModel, str] = func(req)
                if isinstance(result, str):
                    return create_azure_success_response(result)
                return create_azure_success_response(result.json())
            except caught_exceptions as e:
                return create_azure_user_error_response(e)

        return patched_func

    return decorator


def lambda_proxy_handler(
    caught_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[Callable], Callable]:
    """API Gateway's lambda proxy integration compatible
    decorator to parse input, catch certain exceptions and respond to them with 400 errors."""

    def decorator(func):
        def patched_func(event, context):
            try:
                if "body" in event and isinstance(event["body"], str):
                    event["body"] = json.loads(event["body"])
                result: Union[pydantic.BaseModel, str] = func(event, context)
                if isinstance(result, str):
                    return create_http_success_response(result)
                return create_http_success_response(result.json())
            except caught_exceptions as e:
                return create_http_user_error_response(e)

        return patched_func

    return decorator


def gcloud_http_error_handler(
    caught_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[Callable[["flask.Request"], Union[pydantic.BaseModel, str]]], Callable[["flask.Request"], Dict],]:
    """Decorator for Google Cloud Function handlers to,
    catch certain exceptions and respond to them with 400 errors."""

    def decorator(func):
        def patched_func(*args, **kwargs):
            try:
                result: Union[pydantic.BaseModel, str] = func(*args, **kwargs)
                if isinstance(result, str):
                    return create_gcloud_http_success_response(result)
                return create_gcloud_http_success_response(result.json())
            except caught_exceptions as e:
                return create_gcloud_http_user_error_response(e)

        return patched_func

    return decorator


def openfaas_action_handler(
    caught_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[Callable[["flask.Request"], Union[pydantic.BaseModel, str]]], Callable[["flask.Request"], Dict],]:
    """Decorator for OpenFaas Function handlers to,
    catch certain exceptions and respond to them with 400 errors."""

    def decorator(func):
        def patched_func(*args, **kwargs):
            try:
                result: Union[pydantic.BaseModel, str] = func(*args, **kwargs)
                if isinstance(result, str):
                    return create_http_success_response(result)
                return create_http_success_response(result.json())
            except caught_exceptions as e:
                return create_http_user_error_response(e)

        return patched_func

    return decorator


def openwhisk_action_handler(
    caught_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[Callable[[Dict], Union[pydantic.BaseModel, str]]], Callable[[Dict], Dict],]:
    """Decorator for Openwhisk action handlers to parse input,
    catch certain exceptions and respond to them with 400 errors.
    Can deal with normal actions as well as web actions"""

    def decorator(func):
        def patched_func(params: Dict):
            # Put parameters or request body under key "body". Strip __ow_ prefix for web action support
            # See https://github.com/apache/openwhisk/blob/master/docs/webactions.md#http-context for more info
            if any(map(lambda name: name.startswith("__ow_"), params.keys())):
                params = {
                    (key[len("__ow_") :] if key.startswith("__ow_") else key): value for key, value in params.items()
                }
            else:
                params = {"body": params}

            try:
                if isinstance(params["body"], (str, bytes)):
                    # Openwhisk sometimes base64 encodes the body
                    try:
                        params["body"] = json.loads(params["body"])
                    except JSONDecodeError:
                        params["body"] = json.loads(base64.b64decode(params["body"]))

                result: Union[pydantic.BaseModel, str] = func(params)
                if isinstance(result, str):
                    return create_http_success_response(result)
                return create_http_success_response(result.json())
            except caught_exceptions + (JSONDecodeError, binascii.Error) as e:
                return create_http_user_error_response(e)

        return patched_func

    return decorator


async def check_program_installed(name: str):
    # process = await asyncio.create_subprocess_exec(
    #    "hash", name, stdout=asyncio.subprocess.PIPE
    # )
    exitcode, _ = subprocess.getstatusoutput(f"hash {name}")
    return exitcode == 0
