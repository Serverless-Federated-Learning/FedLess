from typing import Dict, List, Optional, Union

import pydantic

from fedless.common.models import (
    AggregationHyperParams,
    FunctionInvocationConfig,
    Hyperparams,
    MongodbConnectionConfig,
)


class ClientFunctionConfig(pydantic.BaseModel):
    function: FunctionInvocationConfig
    hyperparams: Optional[Hyperparams]
    replicas: int = 1


class ClientFunctionConfigList(pydantic.BaseModel):
    functions: List[ClientFunctionConfig]
    hyperparams: Optional[Hyperparams]


class AggregationFunctionConfig(pydantic.BaseModel):
    function: FunctionInvocationConfig
    # configure aggregation hyper params has default values
    hyperparams: AggregationHyperParams


class AggregationFunctionConfigList(pydantic.BaseModel):
    agg_functions: List[AggregationFunctionConfig]


class CognitoConfig(pydantic.BaseModel):
    user_pool_id: str
    region_name: str
    auth_endpoint: str
    invoker_client_id: str
    invoker_client_secret: str
    required_scopes: List[str] = ["client-functions/invoke"]


class ClassDistributionConfig(pydantic.BaseModel):
    private_class_mapping: Dict[str, str] = None
    public_class_mapping: Dict[str, str] = None


# class FedMDConfig(pydantic.BaseModel):
#     models: List[ModelConfig]
#     class_distribution: Optional[ClassDistributionConfig]


# class FedDFConfig(pydantic.BaseModel):
#     models: List[ModelConfig]
#     class_distribution: Optional[ClassDistributionConfig]


class ExperimentConfig(pydantic.BaseModel):
    cognito: Optional[CognitoConfig] = None
    database: MongodbConnectionConfig
    evaluator: FunctionInvocationConfig
    aggregator: Union[AggregationFunctionConfigList, AggregationFunctionConfig]
    clients: ClientFunctionConfigList
    # fedmd_config: Optional[FedMDConfig]
    # feddf_config: Optional[FedDFConfig]
    class_distribution: Optional[ClassDistributionConfig] = None
