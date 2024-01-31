from enum import Enum

from fedless.common.models.models import (
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    DatasetLoaderConfig,
    TestMetrics,
    Parameters,
)
from pydantic import BaseModel
from typing import Optional, List


class AggregationStrategy(str, Enum):
    FedAvg = "fedavg"
    FedProx = "fedprox"
    FedLesScan = "fedlesscan"
    FedNova = "fednova"
    SCAFFOLD = "scaffold"
    FedScore = "fedscore"


class AggregationHyperParams(BaseModel):
    total_client_count: int = 0
    tolerance: int = 0
    is_synchronous: bool = True
    aggregate_online: bool = False
    test_batch_size: int = 10
    buffer_ratio: float = 0.5
    mu: float = 0.001  # fednova
    gmf: float = 0.0  # fednova
    lr: float = 0.01  # fednova/SCAFOLD


class AggregatorFunctionParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    serializer: WeightsSerializerConfig = WeightsSerializerConfig(
        type="npz", params=NpzWeightsSerializerConfig(compressed=False)
    )
    test_data: Optional[DatasetLoaderConfig]
    aggregation_hyper_params: AggregationHyperParams
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FedAvg


class AggregatorFunctionResult(BaseModel):
    new_round_id: int
    num_clients: int
    test_results: Optional[List[TestMetrics]]
    global_test_results: Optional[TestMetrics]


# for numpy list support in BaseModel
class BaseModelNumpay(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class AggregationResult(BaseModelNumpay):
    new_global_weights: Parameters = None
    new_global_controls: Optional[Parameters]
    client_metrics: Optional[List[TestMetrics]]
