from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel

from fedless.common.models.models import (
    DatasetLoaderConfig,
    MongodbConnectionConfig,
    NpzWeightsSerializerConfig,
    TestMetrics,
    WeightsSerializerConfig,
)


class AggregationStrategy(str, Enum):
    PER_ROUND = "per_round"
    PER_SESSION = "per_session"  # enhanced with staleness aware aggregation
    FED_MD = "fed_md"
    FED_DF = "fed_df"


class FedDFAggregatorHyperParams(BaseModel):
    # n_fusion_iterations: int
    KL_temperature: int
    n_pseudo_batches: int
    pseudo_batch_size: int
    eval_batches_frequency: int
    n_distillation_data: int
    patience: int


class AggregationHyperParams(BaseModel):
    tolerance: int = 0
    aggregate_online: bool = False
    test_batch_size: int = 10
    feddf_hyperparams: Optional[FedDFAggregatorHyperParams]


class AggregatorFunctionParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    serializer: WeightsSerializerConfig = WeightsSerializerConfig(
        type="npz", params=NpzWeightsSerializerConfig(compressed=False)
    )
    test_data: Optional[DatasetLoaderConfig]
    aggregation_hyper_params: AggregationHyperParams
    aggregation_strategy: AggregationStrategy = AggregationStrategy.PER_ROUND
    model_type: Optional[str] = None
    action: Optional[str] = None


class AggregatorFunctionResult(BaseModel):
    new_round_id: int
    num_clients: int
    test_results: Union[Optional[List[TestMetrics]], Optional[TestMetrics]]
    global_test_results: Optional[TestMetrics]
