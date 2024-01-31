from enum import Enum
from typing import Optional, Union, Dict, List
from urllib import parse

from datetime import datetime

import numpy as np
from pydantic import (
    Field,
    BaseModel,
    validator,
    BaseSettings,
    PositiveInt,
    StrictBytes,
)

from fedless.common.models.function_config_models import FunctionInvocationConfig
from fedless.common.models.validation_func import params_validate_types_match
from fedless.datasets.fedscale.google_speech.dataset_loader import FedScaleConfig

from fedless.datasets.leaf.dataset_loader import LEAFConfig
from fedless.datasets.mnist.dataset_loader import MNISTConfig

Parameters = List[np.ndarray]


class SerializedModel(BaseModel):
    model_json: str
    optimizer: Union[str, Dict]
    loss: Union[str, Dict]
    metrics: List[str]


class TestMetrics(BaseModel):
    cardinality: int = Field(
        description="tf.data.INFINITE_CARDINALITY if the dataset contains an infinite number of elements or "
        "tf.data.UNKNOWN_CARDINALITY if the analysis fails to determine the number of elements in the dataset "
        "Source: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cardinality"
    )
    metrics: Dict = Field(description="Dictionary mapping from metric name to value")


class H5FullModelSerializerConfig(BaseModel):
    """Configuration parameters for this serializer"""

    type: str = Field("h5", const=True)
    save_traces: bool = True


class NpzWeightsSerializerConfig(BaseModel):
    """Configuration parameters for this serializer"""

    type: str = Field("npz", const=True)
    compressed: bool = False


class BinaryStringFormat(str, Enum):
    BASE64 = "base64"
    NONE = "none"


class LocalDifferentialPrivacyParams(BaseModel):
    l2_norm_clip: float
    noise_multiplier: float
    num_microbatches: Optional[int]


# class DatasetConfig(BaseModel):
#     """configuration for arbitary dataset"""
#     location: Union[AnyHttpUrl, Path]


class DatasetLoaderConfig(BaseModel):
    """Configuration for arbitrary dataset loaders"""

    type: str
    params: Union[LEAFConfig, MNISTConfig, FedScaleConfig]
    # params: DatasetConfig

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class InvocationResult(BaseModel):
    """Returned by invoker functions"""

    session_id: str
    round_id: int
    client_id: str
    invocation_id: Optional[str]
    test_metrics: Optional[TestMetrics] = None


class MongodbConnectionConfig(BaseSettings):
    """
    Data class holding connection info for a MongoDB instance
    Automatically tries to fill in missing values from environment variables
    """

    host: str = Field(...)
    port: Optional[int] = Field(default=27017)
    username: str = Field(...)
    password: str = Field(...)
    is_srv: Optional[bool] = False

    @property
    def url(self) -> str:
        """Return url representation"""
        if self.is_srv:
            return f"mongodb+srv://{parse.quote(self.username)}:{parse.quote(self.password)}@{self.host}/?retryWrites=true&w=majority"
        else:
            return f"mongodb://{parse.quote(self.username)}:{parse.quote(self.password)}@{self.host}:{self.port}/?retryWrites=true&w=majority"

    class Config:
        env_prefix = "fedless_mongodb_"


class FileServerConfig(BaseSettings):
    host: str = Field(...)
    port: Optional[int] = Field(default=80)
    basename: Optional[str] = Field(default="")
    https: Optional[bool] = Field(default=False)

    @property
    def url(self) -> str:
        return (
            ("https://" if self.https else "http://")
            + f"{self.host}:{self.port}"
            + (f"/{self.basename}" if self.basename else "")
        )


class ModelSerializerConfig(BaseModel):
    """Configuration object for arbitrary model serializers of type :class:`ModelSerializer`"""

    type: str
    params: Optional[H5FullModelSerializerConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class WeightsSerializerConfig(BaseModel):
    """Configuration for parameters serializers of type :class:`WeightsSerializer`"""

    type: str
    params: NpzWeightsSerializerConfig

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class SerializedParameters(BaseModel):
    """Parameters as serialized blob with information on how to deserialize it"""

    blob: Union[StrictBytes, str]
    serializer: WeightsSerializerConfig
    string_format: BinaryStringFormat = BinaryStringFormat.NONE


class LocalPrivacyGuarantees(BaseModel):
    eps: float
    delta: float
    rdp: Optional[List]
    orders: Optional[List]
    steps: Optional[int]


class FedProxParams(BaseModel):
    mu: float = 0.001


class FedNovaParams(BaseModel):
    mu: float = 0.001


class ScaffoldParams(BaseModel):
    option: int = 2  # Option to update client controls [1, 2]
    server_controls: Optional[SerializedParameters]
    local_controls: Optional[SerializedParameters]
    local_controls_diff: Optional[SerializedParameters]  # ci^(t+1) - ci^t


class Hyperparams(BaseModel):
    from fedless.common.models.aggregation_models import AggregationStrategy

    """Parameters for training and some data processing"""

    cold_start: bool = False
    cold_start_duration = 0.5  # mean of norm dist for random sample value
    batch_size: PositiveInt
    epochs: PositiveInt
    shuffle_data: bool = True
    optimizer: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Optimizer, either string with name of optimizer or "
        "a config dictionary retrieved via tf.keras.optimizers.serialize. ",
    )
    SGD_learning_rate: float = 0.001  # for fednova and scaffold
    strategy: str = None
    fedprox: FedProxParams = FedProxParams()
    fednova: FedNovaParams = FedNovaParams()
    scaffold: ScaffoldParams = ScaffoldParams()

    loss: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Name of loss function, see https://www.tensorflow.org/api_docs/python/tf/keras/losses, or "
        "a config dictionary retrieved via tf.keras.losses.serialize. ",
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of metrics to be evaluated by the model",
    )
    local_privacy: Optional[LocalDifferentialPrivacyParams]


class ClientResult(BaseModel):
    """Result of client function execution"""

    parameters: SerializedParameters
    history: Optional[Dict]
    training_time: Optional[float]
    local_counters: Optional[Dict]  # for fednova only
    local_controls: Optional[SerializedParameters]  # For Scaffold (client)
    local_controls_diff: Optional[SerializedParameters]  # For Scaffold (aggregator)
    test_metrics: Optional[TestMetrics]
    cardinality: int = Field(
        description="tf.data.INFINITE_CARDINALITY if the dataset contains an infinite number of elements or "
        "tf.data.UNKNOWN_CARDINALITY if the analysis fails to determine the number of elements in the dataset "
        "(e.g. when the dataset source is a file). "
        "Source: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cardinality"
    )
    privacy_guarantees: Optional[LocalPrivacyGuarantees]


class ClientResultStorageObject(BaseModel):
    """Client Result persisted in database with corresponding client identifier"""

    key: str
    result: ClientResult


class PayloadModelLoaderConfig(BaseModel):
    """Configuration parameters required for :class:`PayloadModelLoader`"""

    type: str = Field("payload", const=True)
    payload: Union[StrictBytes, str]
    serializer: ModelSerializerConfig = ModelSerializerConfig(type="h5")


class SimpleModelLoaderConfig(BaseModel):
    """Configuration parameters required for :class:`SimpleModelLoader`"""

    type: str = Field("simple", const=True)

    params: SerializedParameters
    model: str = Field(
        description="Json representation of model architecture. "
        "Created via tf.keras.Model.to_json()"
    )
    compiled: bool = False
    optimizer: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Optimizer, either string with name of optimizer or "
        "a config dictionary retrieved via tf.keras.optimizers.serialize.",
    )
    loss: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Loss, either string with name of loss or "
        "a config dictionary retrieved via tf.keras.losses.serialize.",
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of metrics to be evaluated by the model",
    )


class ModelLoaderConfig(BaseModel):
    """Configuration for arbitrary :class:`ModelLoader`'s"""

    type: str
    params: Union[PayloadModelLoaderConfig, SimpleModelLoaderConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class ClientConfig(BaseModel):
    client_id: str
    session_id: str
    invocation_id: Optional[str]
    function: FunctionInvocationConfig
    data: DatasetLoaderConfig
    hyperparams: Hyperparams
    test_data: Optional[DatasetLoaderConfig]
    compress_model: bool = False


class InvocationStatus(int, Enum):
    completed = 1
    running = 0
    failed = -1
    inv_timeout = -2
    exec_timeout = -3


class InvocationHistory(BaseModel):
    invocation_id: str
    client_id: str
    round_id: int
    session_id: str
    status: InvocationStatus = InvocationStatus.running  # running | completed | failed
    invocation_delay: Optional[float]
    cold_start: bool = False
    invocation_time: Optional[float]  # inv
    complete_time: Optional[float]  # return result
    response_time: Optional[float]  # complete - inv
    function_duration: Optional[float]  # elapsed time / in function time


class ClientScore(BaseModel):
    session_id: str
    client_id: str
    training_times: List[float] = []
    booster_value: float = 1.0  # scale score
    cardinality: PositiveInt = 0  # training data
    epochs: PositiveInt
    batch_size: PositiveInt


class ClientPersistentHistory(BaseModel):
    client_id: str
    session_id: str
    training_times: list = []
    ema: float = 0
    latest_updated: int = -1
    # id of the missed rounds
    missed_rounds: list = []
    # rounds to be skipped from the last failed round this increase exponentially and reset if the client succeded once
    client_backoff: float = 0
    train_cardinality: int = -1  # client cardinality if not inf or not unknown


class InvokerParams(BaseModel):
    """Parameters to run invoker function similarly as proposed by FedKeeper"""

    session_id: str
    round_id: int
    invocation_id: str
    client_id: str
    database: MongodbConnectionConfig
    evaluate_only: bool = False
    http_headers: Optional[Dict] = None
    http_proxies: Optional[Dict] = None
    invocation_delay: Optional[
        int
    ] = 0  # values can be -1 for failure, 0 for no delay, number in secs to delay running the client


class ClientInvocationParams(BaseModel):
    """Convenience class to directly parse and serialize loaders and hyperparameters"""

    data: DatasetLoaderConfig
    model: ModelLoaderConfig
    hyperparams: Hyperparams
    test_data: Optional[DatasetLoaderConfig]


class EvaluatorParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    test_data: DatasetLoaderConfig
    batch_size: int = 128
    metrics: List[str] = ["accuracy"]


class EvaluatorResult(BaseModel):
    metrics: TestMetrics
