from typing import Optional, List

from pydantic import BaseModel

from fedless.common.models import (
    FunctionDeploymentConfig,
    FunctionInvocationConfig,
    Hyperparams,
    MongodbConnectionConfig,
    AggregationHyperParams,
)
from fedless.common.models.models import FileServerConfig


class ClientFunctionConfig(BaseModel):
    function: FunctionInvocationConfig
    hyperparams: Optional[Hyperparams]
    replicas: int = 1


class ClientFunctionConfigList(BaseModel):
    # cold start duration for mocking cold start
    functions: List[ClientFunctionConfig]
    hyperparams: Optional[Hyperparams]


class AggregationFunctionConfig(BaseModel):
    function: FunctionInvocationConfig
    # configure aggregation hyper params has default values
    hyperparams: AggregationHyperParams


class CognitoConfig(BaseModel):
    user_pool_id: str
    region_name: str
    auth_endpoint: str
    invoker_client_id: str
    invoker_client_secret: str
    required_scopes: List[str] = ["client-functions/invoke"]


class ExperimentConfig(BaseModel):
    cognito: Optional[CognitoConfig] = None
    file_server: Optional[FileServerConfig] = None
    database: MongodbConnectionConfig
    evaluator: FunctionInvocationConfig
    aggregator: AggregationFunctionConfig
    clients: ClientFunctionConfigList
