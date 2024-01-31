import json
import logging
from json import JSONDecodeError
from typing import Iterable, Optional, Dict, Union

import backoff
import pymongo
import requests
from requests.adapters import HTTPAdapter

# from requests.packages.urllib3.util.retry import Retry
from urllib3.util import Retry


from fedless.common.models import (
    OpenwhiskActionConfig,
    ApiGatewayLambdaFunctionConfig,
    FunctionInvocationConfig,
    ClientInvocationParams,
    InvocationResult,
    MongodbConnectionConfig,
    ModelLoaderConfig,
    SimpleModelLoaderConfig,
    GCloudFunctionConfig,
    OpenwhiskWebActionConfig,
    AzureFunctionHTTPConfig,
    SerializedParameters,
    BinaryStringFormat,
    OpenFaasFunctionConfig,
)
from fedless.common.serialization import Base64StringConverter
from fedless.common.persistence.client_daos import (
    ClientConfigDao,
    ParameterDao,
    ModelDao,
    ClientResultDao,
)
from fedless.common.persistence.mongodb_base_connector import PersistenceError

logger = logging.getLogger(__name__)


class InvocationError(Exception):
    """Error during function invokation"""


class InvalidInvocationResponse(InvocationError):
    """Response of the invoked function is malformed or otherwise invalid"""


class UnauthorizedInvocationError(InvocationError):
    """Authorization unsuccessful"""


class InvocationTimeOut(InvocationError):
    """Result not found in expected time"""


def function_invoker_handler(
    session_id: str,
    round_id: int,
    invocation_id: str,
    client_id: str,
    database: MongodbConnectionConfig,
    http_headers: Optional[Dict] = None,
    http_proxies: Optional[Dict] = None,
) -> InvocationResult:
    logger.debug(
        f"Invoker called for session {session_id} and client {client_id} for round {round_id}"
    )
    db = pymongo.MongoClient(
        host=database.host,
        port=database.port,
        username=database.username,
        password=database.password,
    )

    try:
        # Create daos to access database
        config_dao = ClientConfigDao(db=db)
        model_dao = ModelDao(db=db)
        parameter_dao = ParameterDao(db=db)
        result_dao = ClientResultDao(db=db)

        # Load model and latest weights
        logger.debug(f"Loading model from database")
        model = model_dao.load(session_id=session_id)
        latest_params: SerializedParameters = parameter_dao.load_latest(session_id)
        if isinstance(latest_params.blob, bytes):
            logger.debug(f"Making model parameters serializable")
            latest_params.blob = Base64StringConverter.to_str(latest_params.blob)
            latest_params.string_format = BinaryStringFormat.BASE64
        model = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=latest_params,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )

        logger.debug(f"Load client config from db")
        # Load client configuration and prepare call statements
        client_config = config_dao.load(client_id=client_id)
        client_params = ClientInvocationParams(
            data=client_config.data,
            model=model,
            hyperparams=client_config.hyperparams,
            test_data=client_config.test_data,
        )

        # Call client
        logger.debug(f"Calling function")
        session = retry_session(backoff_factor=1.0)
        session.headers.update(http_headers or {})
        session.proxies.update(http_proxies or {})
        client_result = invoke_sync(
            function_config=client_config.function,
            data=client_params.dict(),
            session=session,
        )

        logger.debug(f"Finished calling function")
        if isinstance(client_result, dict) and "parameters" in client_result:
            logger.debug(f"Storing results to db")
            result_dao.save(
                session_id=session_id,
                round_id=round_id,
                client_id=client_id,
                result=client_result,
            )
        else:
            logger.error(f"Client invocation failed with response {client_result}")
            raise InvocationError(
                f"Client invocation failed with response {client_result}"
            )

        return InvocationResult(
            session_id=session_id,
            round_id=round_id,
            invocation_id=invocation_id,
            client_id=client_id,
        )

    except PersistenceError as e:
        raise InvocationError(e) from e
    finally:
        db.close()


def invoke_sync(
    function_config: FunctionInvocationConfig,
    data: Union[str, Dict],
    timeout: float = 600,
    session: Optional[requests.Session] = None,
):
    """Convenience method to invoke the given function and abstract different FaaS platforms"""
    session = session or requests.Session()

    if function_config.type == "openwhisk":
        params: OpenwhiskActionConfig = function_config.params
        return invoke_wsk_action_sync(
            data=data,
            action_name=params.name,
            api_host=params.api_host,
            auth_token=params.auth_token,
            session=session,
            verify_certificate=not params.self_signed_cert,
            max_poll_time=timeout,
        )
    elif function_config.type == "openwhisk-web":
        params: OpenwhiskWebActionConfig = function_config.params
        if params.token:
            session.headers.update({"X-require-whisk-auth": params.token})
        response_dict = invoke_http_function_sync(
            url=params.endpoint,
            data=data,
            session=session,
            timeout=timeout,
            verify_certificate=not params.self_signed_cert,
        )
        response_dict = response_dict.get("body", response_dict)

        if not isinstance(response_dict, dict):
            return json.loads(response_dict)
        return response_dict
    elif function_config.type == "lambda":
        params: ApiGatewayLambdaFunctionConfig = function_config.params
        if params.api_key:
            session.headers.update({"X-api-key": params.api_key})
        return invoke_http_function_sync(
            url=params.apigateway, data=data, session=session, timeout=timeout
        )
    elif function_config.type == "gcloud":
        params: GCloudFunctionConfig = function_config.params
        return invoke_http_function_sync(
            url=params.url, data=data, session=session, timeout=timeout
        )
    elif function_config.type == "azure":
        params: AzureFunctionHTTPConfig = function_config.params
        return invoke_http_function_sync(
            url=params.trigger_url, data=data, session=session, timeout=timeout
        )
    elif function_config.type == "openfaas":
        params: OpenFaasFunctionConfig = function_config.params
        return invoke_http_function_sync(
            url=params.url, data=data, session=session, timeout=timeout
        )
    else:
        raise NotImplementedError(
            f"Function of type {function_config.type} not supported"
        )


def invoke_http_function_sync(
    url: str,
    data: Union[Dict, str],
    method: str = "POST",
    session: requests.Session = None,
    verify_certificate: bool = True,
    timeout: float = None,
) -> Dict:
    """Make generic HTTP request for the given function endpoint and wait until result is returned (blocking)"""
    if isinstance(data, dict):
        data = json.dumps(data)

    session = session or requests.Session()
    try:
        response = session.request(
            method=method,
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            verify=verify_certificate,
            proxies=session.proxies,
            timeout=timeout,
        )

        if (
            response.ok
            or response.status_code == 400
            and "errorMessage" in response.text
        ):
            return response.json()
        elif response.status_code == 504:
            raise InvocationTimeOut(
                f"504 Server Error: Gateway Timeout for url: {response.url}"
            )
        response.raise_for_status()
    except requests.exceptions.ReadTimeout as e:
        raise InvocationTimeOut(f"Socket read timeout for url: {url}")
    except requests.exceptions.RequestException as e:
        raise InvocationError(e) from e
    except ValueError as e:
        raise InvalidInvocationResponse(e) from e


def invoke_wsk_action_async(
    action_name: str,
    api_host: str,
    auth_token: str,
    data: Union[Dict, str],
    session: requests.Session = None,
    verify_certificate: bool = True,
) -> str:
    """Invoke function and immediately return activation id"""
    session = session or requests.Session()

    if isinstance(data, dict):
        data = json.dumps(data)

    action_url = (
        f"https://{auth_token}@{api_host}/api/v1/"
        f"namespaces/_/actions/{action_name}?blocking=false"
    )

    try:
        response = session.post(
            url=action_url,
            data=data,
            headers={"Content-Type": "application/json"},
            verify=verify_certificate,
            proxies=session.proxies,
        )
        response.raise_for_status()
        response_dict = response.json()
        return response_dict["activationId"]
    except requests.exceptions.RequestException as e:
        raise InvocationError(e) from e
    except (KeyError, ValueError) as e:
        raise InvalidInvocationResponse(e) from e


def _fetch_openwhisk_activation_result(
    activation_id: str,
    api_host: str,
    auth_token: str,
    session: requests.Session = None,
    verify_certificate: bool = True,
) -> Optional[Dict]:
    """Check if result for given activation_id exists. Return None if not found"""

    session = session or requests.Session()

    try:
        response = session.get(
            url=f"https://{auth_token}@{api_host}/api/v1/namespaces/_/activations/{activation_id}/result",
            verify=verify_certificate,
            proxies=session.proxies,
        )

        if (
            response.status_code == 404
            and "The requested resource does not exist." in response.text
        ):
            return None

        response.raise_for_status()
        response_dict = response.json()["result"]
        response_body = response_dict.get("body")

        # Sometimes the body is still a string because the body was escaped
        if isinstance(response_body, str):
            return json.loads(response_body)

        return response_body
    except requests.exceptions.RequestException as e:
        raise InvocationError(e) from e
    except (KeyError, ValueError, JSONDecodeError) as e:
        raise InvalidInvocationResponse(e) from e


def poll_openwhisk_activation_result(
    activation_id: str,
    api_host: str,
    auth_token: str,
    session: requests.Session = None,
    verify_certificate: bool = True,
    max_time: float = 500,
    interval: float = 1,
):
    """Poll result for given activation id (blocking)"""

    _fetch_and_retry = backoff.on_predicate(
        backoff.constant, max_time=max_time, interval=interval, logger=None
    )(_fetch_openwhisk_activation_result)
    result = _fetch_and_retry(
        activation_id=activation_id,
        api_host=api_host,
        auth_token=auth_token,
        session=session,
        verify_certificate=verify_certificate,
    )
    if result is None:
        raise InvocationTimeOut(
            f"Result for activation with id {activation_id} not found in time. "
            f"Either the function did not run correctly or it timed out. "
            f"Check logs and consider increasing the maximum polling time or maximum function run time."
        )
    return result


def invoke_wsk_action_sync(
    action_name: str,
    api_host: str,
    auth_token: str,
    data: Union[Dict, str],
    session: requests.Session = None,
    verify_certificate: bool = True,
    max_poll_time: float = 600,
    poll_interval: float = 1,
) -> Dict:
    """Invoke a synchronous openwhisk action and continuously poll until it returns a result (blocking)"""
    session = session or requests.Session()

    activation_id = invoke_wsk_action_async(
        action_name=action_name,
        api_host=api_host,
        auth_token=auth_token,
        data=data,
        session=session,
        verify_certificate=verify_certificate,
    )

    return poll_openwhisk_activation_result(
        activation_id=activation_id,
        api_host=api_host,
        auth_token=auth_token,
        session=session,
        verify_certificate=verify_certificate,
        max_time=max_poll_time,
        interval=poll_interval,
    )


def retry_session(
    retries: int = 3,
    allowed_methods: Iterable[str] = None,
    status_list: Iterable[int] = None,
    session: requests.Session = None,
    prefixes: Iterable[str] = None,
    backoff_factor: float = 0.0,
) -> requests.Session:
    """Creates a custom :class:`requests.Session` that automatically retries HTTP Requests"""
    if allowed_methods is None:
        allowed_methods = {"POST", *Retry.DEFAULT_METHOD_WHITELIST}
    if prefixes is None:
        prefixes = ["http://", "https://"]
    if status_list is None:
        status_list = {
            413,
            421,
            423,
            429,
            500,
            502,
            503,
        }  # Optionally include 504 for Gateway Timeout

    session = session or requests.Session()

    retry = Retry(
        status=retries,
        read=0,
        connect=0,
        total=None,
        status_forcelist=status_list,
        allowed_methods=allowed_methods,
        backoff_factor=backoff_factor,
    )
    adapter = HTTPAdapter(max_retries=retry)

    # Register adapters
    for prefix in prefixes:
        session.mount(prefix, adapter)
    return session
