from typing import Optional, Union
from pydantic import (
    Field,
    BaseModel,
    validator,
)
from fedless.common.models.validation_func import params_validate_types_match


class OpenwhiskActionConfig(BaseModel):
    """Info to describe different functions deployed in an openwhisk cluster"""

    type: str = Field("openwhisk", const=True)

    namespace: str = "guest"
    package: str = "default"
    name: str
    auth_token: str
    api_host: str
    self_signed_cert: bool


class OpenwhiskWebActionConfig(BaseModel):
    type: str = Field("openwhisk-web", const=True)
    self_signed_cert: bool = True
    endpoint: str
    token: Optional[str]


class ApiGatewayLambdaFunctionConfig(BaseModel):
    """Lambda function deployed via Api Gateway. All requests time out after 30 seconds due to fixed limit"""

    type: str = Field("lambda", const=True)
    apigateway: str
    api_key: Optional[str]


class GCloudFunctionConfig(BaseModel):
    """Google cloud function"""

    type: str = Field("gcloud", const=True)
    url: str


class OpenFaasFunctionConfig(BaseModel):
    """OpenFaas function"""

    type: str = Field("openfaas", const=True)
    url: str


class AzureFunctionHTTPConfig(BaseModel):
    """Azure function"""

    type: str = Field("azure", const=True)
    trigger_url: str


class FaaSConfig(BaseModel):
    auto_scaling: int = Field(default=0)
    cool_off_time: int = Field(default=0)
    exec_timeout: int = Field(default=300)


class FunctionInvocationConfig(BaseModel):
    """Necessary information to invoke a function"""

    type: str
    params: Union[
        OpenwhiskActionConfig,
        ApiGatewayLambdaFunctionConfig,
        GCloudFunctionConfig,
        OpenwhiskWebActionConfig,
        AzureFunctionHTTPConfig,
        OpenFaasFunctionConfig,
    ]
    invocation_delay: Optional[int] = 0
    cool_start: bool = False
    auto_scaling: int = Field(default=1)

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class GCloudProjectConfig(BaseModel):
    type: str = Field("gcloud", const=True)
    account: str
    project: str


class OpenwhiskFunctionDeploymentConfig(BaseModel):
    type: str = Field("openwhisk", const=True)
    name: str
    main: Optional[str]
    file: str
    image: str
    memory: int
    timeout: int
    web: str = "raw"
    web_secure: bool = False


class GCloudFunctionDeploymentConfig(BaseModel):
    type: str = Field("gcloud", const=True)
    name: str
    directory: str
    memory: int
    timeout: int
    wheel_url: str
    entry_point: Optional[str] = None
    runtime: str = "python38"
    max_instances: int = 100
    trigger_http: bool = True
    allow_unauthenticated: bool = True


class FunctionDeploymentConfig(BaseModel):
    type: str
    params: OpenwhiskFunctionDeploymentConfig

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )
