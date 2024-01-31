# import asyncio
# import base64
# import json
# from typing import Optional, Dict
# from unittest.mock import patch, MagicMock

# import pydantic
# from pydantic import ValidationError

# from fedless.client import ClientError
# from fedless.data import DatasetNotLoadedError
# from fedless.common.providers import (
#     create_http_success_response,
#     create_http_user_error_response,
#     lambda_proxy_handler,
#     create_gcloud_http_success_response,
#     create_gcloud_http_user_error_response,
#     gcloud_http_error_handler,
#     openwhisk_action_handler,
#     check_program_installed,
#     OpenwhiskCluster,
# )
# from fedless.common.models import OpenwhiskFunctionDeploymentConfig
# from .fixtures import *


# class DummyModel(pydantic.BaseModel):
#     parameters: str
#     history: Optional[Dict]
#     cardinality: int


# def default_handler(*args, **kwargs):
#     return DummyModel(parameters="abc", history={"accuracy": [1.0]}, cardinality=4)


# def test_http_client_result_reponse():
#     result = DummyModel(parameters="1234", history={"loss": [1.0]}, cardinality=12)
#     response = create_http_success_response(result.json())

#     assert response == {
#         "statusCode": 200,
#         "body": json.dumps(
#             {"parameters": "1234", "history": {"loss": [1.0]}, "cardinality": 12}
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# @patch("traceback.format_exc")
# def test_http_user_error_response(mock_format_exc):
#     format_exc_mock_value = "Formatted exception information"
#     mock_format_exc.return_value = format_exc_mock_value

#     message = "Dataset did not load!"
#     exception = ClientError(DatasetNotLoadedError(message))

#     response = create_http_user_error_response(exception)

#     assert response == {
#         "statusCode": 400,
#         "body": json.dumps(
#             {
#                 "errorMessage": message,
#                 "errorType": str(ClientError.__name__),
#                 "details": format_exc_mock_value,
#             }
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# def test_lambda_proxy_decorator_returns_valid_response():
#     def dummy_handler(event, context):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = lambda_proxy_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function({}, {})
#     assert result_object == {
#         "statusCode": 200,
#         "body": json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# def test_lambda_proxy_decorator_accepts_json():
#     @lambda_proxy_handler((ValidationError, ClientError))
#     def dummy_handler(event, context):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         ).json()

#     result_object = dummy_handler({}, {})
#     assert result_object == {
#         "statusCode": 200,
#         "body": json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# @patch("traceback.format_exc")
# def test_lambda_proxy_decorator_returns_valid_error_dict(mock_format_exc):
#     exception_message = "Dataset did not load!"
#     format_exc_mock_value = "Formatted exception information"
#     mock_format_exc.return_value = format_exc_mock_value

#     def dummy_handler(event, context):
#         raise ClientError(DatasetNotLoadedError(exception_message))

#     patched_function = lambda_proxy_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function({}, {})
#     assert result_object == {
#         "statusCode": 400,
#         "body": json.dumps(
#             {
#                 "errorMessage": exception_message,
#                 "errorType": str(ClientError.__name__),
#                 "details": format_exc_mock_value,
#             }
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# def test_lambda_proxy_decorator_does_not_wrap_unknown_exception():
#     def dummy_handler(event, context):
#         raise MemoryError("fake memory error")

#     patched_function = lambda_proxy_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     with pytest.raises(MemoryError, match="fake memory error"):
#         patched_function({}, {})


# def test_lambda_proxy_decorator_parses_body():
#     def dummy_handler(event, context):
#         assert isinstance(event, dict)
#         assert isinstance(event["body"], dict)
#         return DummyModel(parameters="1234", history={"loss": [0.0, 1.0]})

#     patched_function = lambda_proxy_handler((ValidationError, ClientError))(
#         dummy_handler
#     )
#     patched_function({"body": '{"key": "value"}'}, {})


# def test_gcloud_http_reponse():
#     result = DummyModel(parameters="1234", history={"loss": [1.0]}, cardinality=12)
#     response = create_gcloud_http_success_response(result.json())

#     assert response == (
#         json.dumps(
#             {"parameters": "1234", "history": {"loss": [1.0]}, "cardinality": 12}
#         ),
#         200,
#         {"Content-Type": "application/json"},
#     )


# @patch("traceback.format_exc")
# def test_glcoud_http_user_error_response(mock_format_exc):
#     format_exc_mock_value = "Formatted exception information"
#     mock_format_exc.return_value = format_exc_mock_value

#     message = "Dataset did not load!"
#     exception = DatasetNotLoadedError(message)

#     response = create_gcloud_http_user_error_response(exception)

#     assert response == (
#         json.dumps(
#             {
#                 "errorMessage": message,
#                 "errorType": str(DatasetNotLoadedError.__name__),
#                 "details": format_exc_mock_value,
#             }
#         ),
#         400,
#         {"Content-Type": "application/json"},
#     )


# def test_gcloud_http_error_handler_decorator_returns_valid_response():
#     def dummy_handler(request):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = gcloud_http_error_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function(None)
#     assert result_object == (
#         json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         200,
#         {"Content-Type": "application/json"},
#     )


# def test_gcloud_http_error_handler_decorator_accepts_json():
#     @gcloud_http_error_handler((ValidationError, ClientError))
#     def dummy_handler(request):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         ).json()

#     result_object = dummy_handler(None)
#     assert result_object == (
#         json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         200,
#         {"Content-Type": "application/json"},
#     )


# @patch("traceback.format_exc")
# def test_gcloud_http_error_handler_decorator_returns_valid_error_dict(mock_format_exc):
#     exception_message = "Dataset did not load!"
#     format_exc_mock_value = "Formatted exception information"
#     mock_format_exc.return_value = format_exc_mock_value

#     def dummy_handler(request):
#         raise ClientError(DatasetNotLoadedError(exception_message))

#     patched_function = gcloud_http_error_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function(None)
#     assert result_object == (
#         json.dumps(
#             {
#                 "errorMessage": exception_message,
#                 "errorType": str(ClientError.__name__),
#                 "details": format_exc_mock_value,
#             }
#         ),
#         400,
#         {"Content-Type": "application/json"},
#     )


# def test_gcloud_http_error_handler_decorator_does_not_wrap_unknown_exception():
#     def dummy_handler(request):
#         raise MemoryError("fake memory error")

#     patched_function = gcloud_http_error_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     with pytest.raises(MemoryError, match="fake memory error"):
#         patched_function(None)


# def test_openwhisk_http_error_handler_decorator_returns_valid_response():
#     def dummy_handler(params):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function({})
#     assert result_object == {
#         "statusCode": 200,
#         "body": json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# def test_openwhisk_http_error_handler_decorator_returns_valid_response_on_str_return():
#     def dummy_handler(params):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         ).json()

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function({})
#     assert result_object == {
#         "statusCode": 200,
#         "body": json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# def test_openwhisk_http_error_handler_decorator_accepts_json():
#     @openwhisk_action_handler((ValidationError, ClientError))
#     def dummy_handler(params):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     result_object = dummy_handler({})
#     assert result_object == {
#         "statusCode": 200,
#         "body": json.dumps(
#             {"parameters": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# @patch("traceback.format_exc")
# def test_openwhisk_http_error_handler_decorator_returns_valid_error_dict(
#     mock_format_exc,
# ):
#     exception_message = "Dataset did not load!"
#     format_exc_mock_value = "Formatted exception information"
#     mock_format_exc.return_value = format_exc_mock_value

#     def dummy_handler(request):
#         raise ClientError(DatasetNotLoadedError(exception_message))

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     result_object = patched_function({})
#     assert result_object == {
#         "statusCode": 400,
#         "body": json.dumps(
#             {
#                 "errorMessage": exception_message,
#                 "errorType": str(ClientError.__name__),
#                 "details": format_exc_mock_value,
#             }
#         ),
#         "headers": {"Content-Type": "application/json"},
#     }


# def test_openwhisk_http_error_handler_decorator_does_not_wrap_unknown_exception():
#     def dummy_handler(request):
#         raise MemoryError("fake memory error")

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )

#     with pytest.raises(MemoryError, match="fake memory error"):
#         patched_function({})


# def test_openwhisk_web_action_handler_accepts_direct_invocation():
#     def dummy_handler(params):
#         assert isinstance(params, dict)
#         assert params["body"] == {"key": "value"}
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )
#     patched_function({"key": "value"})


# def test_openwhisk_web_action_handler_accepts_empty_body():
#     def dummy_handler(params):
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )
#     patched_function({})


# def test_openwhisk_web_action_handler_converts_web_request_body():
#     def dummy_handler(params):
#         assert isinstance(params, dict)
#         assert params["body"] == {"key": "value"}
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )
#     patched_function({"__ow_body": json.dumps({"key": "value"})})


# def test_openwhisk_web_action_handler_converts_base64_encoded_web_request_body():
#     def dummy_handler(params):
#         assert isinstance(params, dict)
#         assert params["body"] == {"key": "value"}
#         return DummyModel(
#             parameters="1234", history={"loss": [0.0, 1.0]}, cardinality=12
#         )

#     patched_function = openwhisk_action_handler((ValidationError, ClientError))(
#         dummy_handler
#     )
#     patched_function(
#         {
#             "__ow_body": base64.b64encode(
#                 bytes(json.dumps({"key": "value"}), encoding="utf-8")
#             )
#         }
#     )


# @pytest.mark.asyncio
# async def test_check_program_installed_with_valid_program():
#     assert await check_program_installed("ls")
#     assert await check_program_installed("pwd")
#     assert not await check_program_installed("this-program-does-certainly-not-exist")


# @pytest.mark.asyncio
# async def test_openwhisk_deploy():
#     # Fix for Python 3.7
#     # https://stackoverflow.com/questions/51394411/python-object-magicmock-cant-be-used-in-await-expression
#     class AsyncMock(MagicMock):
#         async def __call__(self, *args, **kwargs):
#             return super(AsyncMock, self).__call__(*args, **kwargs)

#     cluster = OpenwhiskCluster(apihost="localhost:3141", auth="myauthstring")
#     run_cmd_mock = AsyncMock(cluster._run_command)

#     with patch.object(cluster, "_run_command", run_cmd_mock):
#         await cluster.deploy(
#             OpenwhiskFunctionDeploymentConfig(
#                 name="client",
#                 file="main.py",
#                 image="Dockerfile",
#                 memory=2048,
#                 timeout=60,
#             )
#         )
#         run_cmd_mock.assert_called_with(
#             "wsk action update client main.py --docker Dockerfile "
#             "--memory 2048 --timeout 60 --web raw --web-secure false"
#         )
