import httpretty
import pytest
import requests

from fedless.controller.invocation import (
    retry_session,
    invoke_wsk_action_async,
    InvocationError,
    InvalidInvocationResponse,
    poll_openwhisk_activation_result,
    InvocationTimeOut,
    _fetch_openwhisk_activation_result,
    invoke_wsk_action_sync,
    invoke_http_function_sync,
)
from fedless.common.models import OpenwhiskActionConfig
from .common import get_error_function


@pytest.fixture
def dummy_wsk_action_config():
    return OpenwhiskActionConfig(
        name="hello",
        auth_token="abc123",
        api_host="my-cluster:3131",
        self_signed_cert=True,
    )


@pytest.fixture
def dummy_wsk_action_url(dummy_wsk_action_config):
    return (
        f"https://{dummy_wsk_action_config.auth_token}@{dummy_wsk_action_config.api_host}/api/v1/"
        f"namespaces/_/actions/{dummy_wsk_action_config.name}?blocking=false"
    )


@pytest.fixture
def dummy_wsk_activation_id():
    return "123asdb"


@pytest.fixture
def dummy_wsk_activation_url(dummy_wsk_action_config, dummy_wsk_activation_id):
    return (
        f"https://{dummy_wsk_action_config.auth_token}@{dummy_wsk_action_config.api_host}"
        f"/api/v1/namespaces/_/activations/{dummy_wsk_activation_id}/result"
    )


@httpretty.activate
def test_retry_session_does_retry_with_default_parameters():
    session_under_test = retry_session()

    httpretty.register_uri(
        httpretty.POST,
        "https://test-url.com",
        responses=[
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=502,
            ),
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=502,
            ),
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=502,
            ),
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=200,
            ),
        ],
    )

    response = session_under_test.post("https://test-url.com")
    assert response.status_code == 200


@httpretty.activate
def test_retry_session_does_retry_with_custom_parameters():
    session = requests.Session()
    session_under_test = retry_session(
        retries=2,
        allowed_methods=["GET"],
        status_list=[400],
        session=session,
        prefixes=["http://"],
    )

    httpretty.register_uri(
        httpretty.GET,
        "http://test-url.com",
        responses=[
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=400,
            ),
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=200,
            ),
        ],
    )

    response = session_under_test.get("http://test-url.com")
    assert response.status_code == 200


@httpretty.activate
def test_retry_fails_on_invalid_status():
    session_under_test = retry_session()
    httpretty.register_uri(
        httpretty.GET,
        "http://test-url.com",
        responses=[
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
                status=403,
            ),
            httpretty.Response(
                body='{"message": "HTTPretty :)"}',
            ),
        ],
    )
    response = session_under_test.get("http://test-url.com")
    assert response.status_code == 403


@httpretty.activate
def test_async_wsk_invocation_returns_activation_id(
    dummy_wsk_action_config, dummy_wsk_action_url
):
    activation_id = "m14ctivation"

    httpretty.register_uri(
        httpretty.POST,
        dummy_wsk_action_url,
        body='{"activationId": "' + str(activation_id) + '"}',
    )

    returned_activation_id = invoke_wsk_action_async(
        action_name=dummy_wsk_action_config.name,
        api_host=dummy_wsk_action_config.api_host,
        auth_token=dummy_wsk_action_config.auth_token,
        data="",
        verify_certificate=not dummy_wsk_action_config.self_signed_cert,
    )

    assert returned_activation_id == activation_id


@httpretty.activate
def test_async_wsk_invocation_wraps_errors(
    dummy_wsk_action_config, dummy_wsk_action_url
):
    # Call returns 401
    httpretty.register_uri(
        httpretty.POST,
        dummy_wsk_action_url,
        status=401,
        body='{"error": "123","code": "123"}',
    )

    with pytest.raises(InvocationError):
        invoke_wsk_action_async(
            action_name=dummy_wsk_action_config.name,
            api_host=dummy_wsk_action_config.api_host,
            auth_token=dummy_wsk_action_config.auth_token,
            data={},
            verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        )

    # Requests throws timeout error
    httpretty.register_uri(
        httpretty.POST,
        dummy_wsk_action_url,
        status=403,
        body=get_error_function(requests.Timeout),
    )

    with pytest.raises(InvocationError):
        invoke_wsk_action_async(
            action_name=dummy_wsk_action_config.name,
            api_host=dummy_wsk_action_config.api_host,
            auth_token=dummy_wsk_action_config.auth_token,
            data={},
            verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        )

    # Result is malformed
    httpretty.register_uri(
        httpretty.POST,
        dummy_wsk_action_url,
        body='{"no-activation-id-here": "123"}',
    )

    with pytest.raises(InvalidInvocationResponse):
        invoke_wsk_action_async(
            action_name=dummy_wsk_action_config.name,
            api_host=dummy_wsk_action_config.api_host,
            auth_token=dummy_wsk_action_config.auth_token,
            data="",
            verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        )


@httpretty.activate
def test_poll_wsk_result_returns_result(
    dummy_wsk_action_config,
    dummy_wsk_activation_url,
    dummy_wsk_activation_id,
):
    httpretty.register_uri(
        httpretty.GET,
        dummy_wsk_activation_url,
        responses=[
            httpretty.Response(
                body='{"error": "The requested resource does not exist."}',
                status=404,
            ),
            httpretty.Response(
                body='{"status": "string","success": true,"size": 0, "result": {"body": {"value": "My Result"}}}',
            ),
        ],
    )

    result = poll_openwhisk_activation_result(
        activation_id=dummy_wsk_activation_id,
        api_host=dummy_wsk_action_config.api_host,
        auth_token=dummy_wsk_action_config.auth_token,
        verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        max_time=10,
        interval=0.01,
    )
    assert result == {"value": "My Result"}


@httpretty.activate
def test_poll_wsk_result_raises_error_on_no_result(
    dummy_wsk_action_config,
    dummy_wsk_activation_url,
    dummy_wsk_activation_id,
):
    httpretty.register_uri(
        httpretty.GET,
        dummy_wsk_activation_url,
        responses=[
            httpretty.Response(
                body='{"error": "The requested resource does not exist."}',
                status=404,
            ),
            httpretty.Response(
                body='{"error": "The requested resource does not exist."}',
                status=404,
            ),
        ],
    )

    with pytest.raises(InvocationTimeOut):
        poll_openwhisk_activation_result(
            activation_id=dummy_wsk_activation_id,
            api_host=dummy_wsk_action_config.api_host,
            auth_token=dummy_wsk_action_config.auth_token,
            verify_certificate=not dummy_wsk_action_config.self_signed_cert,
            max_time=0.1,
            interval=0.01,
        )


@httpretty.activate
def test_fetch_wsk_result_wraps_errors(
    dummy_wsk_action_config,
    dummy_wsk_activation_url,
    dummy_wsk_activation_id,
):
    httpretty.register_uri(
        httpretty.GET,
        dummy_wsk_activation_url,
        body=get_error_function(requests.HTTPError),
    )

    with pytest.raises(InvocationError):
        _fetch_openwhisk_activation_result(
            activation_id=dummy_wsk_activation_id,
            api_host=dummy_wsk_action_config.api_host,
            auth_token=dummy_wsk_action_config.auth_token,
            verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        )

    httpretty.register_uri(
        httpretty.GET,
        dummy_wsk_activation_url,
        body='{"no-result-here":"sorry"}',
    )
    with pytest.raises(InvalidInvocationResponse):
        _fetch_openwhisk_activation_result(
            activation_id=dummy_wsk_activation_id,
            api_host=dummy_wsk_action_config.api_host,
            auth_token=dummy_wsk_action_config.auth_token,
            verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        )


@httpretty.activate
def test_invoke_wsk_action_sync_returns_result(
    dummy_wsk_action_config,
    dummy_wsk_activation_url,
    dummy_wsk_activation_id,
    dummy_wsk_action_url,
):
    # Async invocation returns activation id
    httpretty.register_uri(
        httpretty.POST,
        dummy_wsk_action_url,
        body='{"activationId": "' + dummy_wsk_activation_id + '"}',
    )

    # Result for this activation id
    httpretty.register_uri(
        httpretty.GET,
        dummy_wsk_activation_url,
        responses=[
            httpretty.Response(
                body='{"error": "The requested resource does not exist."}',
                status=404,
            ),
            httpretty.Response(
                body='{"error": "The requested resource does not exist."}',
                status=404,
            ),
            httpretty.Response(
                body='{"status": "string","success": true,"size": 0, "result": {"body": {"Value": "My Result"}}}',
            ),
        ],
    )

    result = invoke_wsk_action_sync(
        data={},
        action_name=dummy_wsk_action_config.name,
        api_host=dummy_wsk_action_config.api_host,
        auth_token=dummy_wsk_action_config.auth_token,
        verify_certificate=not dummy_wsk_action_config.self_signed_cert,
        max_poll_time=10,
        poll_interval=0.01,
    )
    assert result == {"Value": "My Result"}


@httpretty.activate
def test_invoke_function_sync_returns_correct_result():
    # Successful invocation
    url = "https://test-api-gateway.com/dev/funtion"
    httpretty.register_uri(
        httpretty.POST,
        url,
        body='{"key123": "value123"}',
    )

    response = invoke_http_function_sync(url, data={})
    assert response == {"key123": "value123"}

    response = invoke_http_function_sync(url, data='{"param1": "value1"}')
    assert response == {"key123": "value123"}

    # Custom error was returned
    httpretty.register_uri(
        httpretty.POST,
        url,
        status=400,
        body='{"errorMessage": "message", "errorType": "ValidationError", "details": "details"}',
    )
    response = invoke_http_function_sync(url, data='{"param1": "value1"}')
    assert response == {
        "errorMessage": "message",
        "errorType": "ValidationError",
        "details": "details",
    }


@httpretty.activate
def test_invoke_function_sync_raises_correct_errors():
    url = "https://test-api-gateway.com/dev/funtion"

    # Gateway Timeout
    httpretty.register_uri(httpretty.POST, url, body="{}", status=504)
    with pytest.raises(InvocationTimeOut):
        invoke_http_function_sync(url, data={})

    # General Timeout
    httpretty.register_uri(
        httpretty.POST, url, body=get_error_function(requests.Timeout)
    )
    with pytest.raises(InvocationError):
        invoke_http_function_sync(url, data={})

    # 400 Error
    httpretty.register_uri(httpretty.POST, url, status=400, body="{}")
    with pytest.raises(InvocationError):
        invoke_http_function_sync(url, data={})

    # Response not valid json
    httpretty.register_uri(httpretty.POST, url, body="{Not valid json}")
    with pytest.raises(InvalidInvocationResponse):
        invoke_http_function_sync(url, data={})
