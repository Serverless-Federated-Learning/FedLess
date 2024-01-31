import base64
import random
from itertools import zip_longest
from typing import Tuple, Dict
from unittest.mock import patch, MagicMock

import h5py
import pydantic
from _pytest.monkeypatch import MonkeyPatch
from keras.utils.losses_utils import ReductionV2
from pydantic import ValidationError

from fedless.common.models import (
    WeightsSerializerConfig,
    H5FullModelSerializerConfig,
    SerializedParameters,
    NpzWeightsSerializerConfig,
    ModelLoaderConfig,
    PayloadModelLoaderConfig,
    ModelSerializerConfig,
    BinaryStringFormat,
)
from fedless.common.serialization import (
    H5FullModelSerializer,
    WeightsSerializerBuilder,
    deserialize_parameters,
    ModelSerializer,
    ModelLoaderBuilder,
    ModelSerializerBuilder,
    PayloadModelLoader,
    ModelLoadError,
    SerializationError,
    SimpleModelLoader,
    serialize_model,
)
from .common import (
    get_error_function,
    are_weights_equal,
    is_optimizer_state_preserved,
    is_model_trainable,
)
from .fixtures import *


def assert_models_equal(model: tf.keras.Model, model_re: tf.keras.Model):
    assert isinstance(model_re, tf.keras.Model)
    assert model_re.get_config() == model.get_config()
    assert are_weights_equal(model.get_weights(), model_re.get_weights())
    assert is_optimizer_state_preserved(model.optimizer, model_re.optimizer)


@pytest.fixture
def dummy_data() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 10
    return np.random.randn(n_samples, 3), np.random.randn(n_samples, 4)


def model_serializer_test_suite(
    model: tf.keras.Model,
    serializer: ModelSerializer,
    dummy_data: Tuple[np.ndarray, np.ndarray],
):
    blob = serializer.serialize(model)
    model_re = serializer.deserialize(blob)

    assert_models_equal(model, model_re)
    assert is_model_trainable(model, dummy_data)


def test_h5_serializer(simple_model, dummy_data: Tuple[np.ndarray, np.ndarray]):
    s = H5FullModelSerializer()
    model_serializer_test_suite(simple_model, s, dummy_data)


def test_h5_serializer_rethrows_exception(simple_model, monkeypatch: MonkeyPatch):
    s = H5FullModelSerializer()

    with pytest.raises(SerializationError):
        monkeypatch.setattr(simple_model, "save", get_error_function(ImportError))
        s.serialize(simple_model)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(h5py.File, "__enter__", get_error_function(IOError))
        s.serialize(simple_model)


def test_h5_serializer_does_not_wrap_memory_error(
    simple_model, monkeypatch: MonkeyPatch
):
    s = H5FullModelSerializer()
    blob = s.serialize(simple_model)

    with pytest.raises(MemoryError):
        monkeypatch.setattr(h5py.File, "__enter__", get_error_function(MemoryError))
        s.deserialize(blob)


def test_h5_serializer_fails_on_invalid_blob(simple_model, monkeypatch: MonkeyPatch):
    s = H5FullModelSerializer()

    blob = s.serialize(simple_model)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(tf.keras.models, "load_model", get_error_function(IOError))
        s.deserialize(blob)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(
            tf.keras.models, "load_model", get_error_function(ImportError)
        )
        s.deserialize(blob)


def test_h5_serializer_config_type():
    config = H5FullModelSerializerConfig(save_traces=False)
    assert not config.save_traces

    config = H5FullModelSerializerConfig(type="h5", save_traces=True)
    assert config.save_traces and config.type == "h5"

    with pytest.raises(pydantic.ValidationError):
        H5FullModelSerializerConfig(type="s3")

    with pytest.raises(pydantic.ValidationError):
        # noinspection PyTypeChecker
        H5FullModelSerializerConfig(save_traces="Yes, please")


@pytest.mark.parametrize("save_traces", [True, False])
def test_h5_serializer_can_be_constructed_from_config(save_traces):
    config = H5FullModelSerializerConfig(save_traces=save_traces)
    serializer = H5FullModelSerializer.from_config(config)
    assert serializer is not None
    assert serializer.save_traces == save_traces


def test_model_serializer_config_types_must_match():
    with pytest.raises(ValidationError):
        config_dict = {"type": "s3", "params": {"type": "h5"}}
        ModelSerializerConfig.parse_obj(config_dict)


@pytest.mark.parametrize("serializer_config", [H5FullModelSerializerConfig()])
def test_model_serializer_config_types_can_be_created(serializer_config):
    config = ModelSerializerConfig(
        type=serializer_config.type, params=serializer_config
    )
    assert config is not None
    assert config.type == serializer_config.type
    assert config.params == serializer_config


def test_model_serializer_builder_fails_on_unknown_type():
    config = ModelSerializerConfig(type="invalid_type")
    with pytest.raises(NotImplementedError):
        ModelSerializerBuilder.from_config(config)


@patch("fedless.serialization.H5FullModelSerializer")
def test_model_serializer_builder_creates_object(serializer_mock):
    serializer_config = H5FullModelSerializerConfig()
    config = ModelSerializerConfig(type="h5", params=serializer_config)
    serializer = ModelSerializerBuilder.from_config(config)
    assert serializer is not None
    assert serializer_mock.from_config.called_with(serializer_config)


@patch("fedless.serialization.H5FullModelSerializer")
def test_model_serializer_builder_creates_object_without_params_specified(
    serializer_mock,
):
    config = ModelSerializerConfig(type="h5")
    serializer = ModelSerializerBuilder.from_config(config)
    assert serializer is not None
    assert serializer_mock.called_with()


@pytest.mark.parametrize("bytes_length", [0, 1, 16, 512])
def test_base64_converter_on_random_bytes(bytes_length):
    random_bytes = bytes([random.randrange(0, 256) for _ in range(bytes_length)])
    json_str = Base64StringConverter.to_str(random_bytes)
    assert Base64StringConverter.from_str(json_str) == random_bytes


def test_base64_converter_raises_error_on_internal_error(monkeypatch):
    monkeypatch.setattr(base64, "b64decode", get_error_function(ValueError))
    with pytest.raises(ValueError):
        valid_b64_str = "SSBoYXZlIHRoZSBoaWdoIGdyb3VuZCE"
        Base64StringConverter.from_str(valid_b64_str)


def test_base64_converter_throws_error_on_invalid_string():
    with pytest.raises(ValueError):
        invalid_b64_str = "!nv4l!D!"
        Base64StringConverter.from_str(invalid_b64_str)


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_serializer_restores_weights(
    simple_model: tf.keras.Model, compressed
):
    s = NpzWeightsSerializer(compressed=compressed)

    weights = simple_model.get_weights()
    reconstructured_weights = s.deserialize(s.serialize(weights))

    assert all(
        (np.allclose(a, b) for a, b in zip_longest(weights, reconstructured_weights))
    )


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_serialier_does_not_fail_on_empty_input(compressed):
    s = NpzWeightsSerializer(compressed=compressed)
    assert s.deserialize(s.serialize([])) == []


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_restores_types(compressed):
    weights = [
        np.random.randn(10, 15).astype(np.float32),
        np.random.randint(0, 32, (8, 5)),
        np.random.randint(0, 1, (5, 6)).astype(np.int8),
    ]

    s = NpzWeightsSerializer(compressed=compressed)
    reconstructured_weights = s.deserialize(s.serialize(weights))
    assert all(
        (
            np.allclose(a, b) and a.dtype == a.dtype
            for a, b in zip_longest(weights, reconstructured_weights)
        )
    )


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_serializer_wraps_loading_errors(
    compressed, monkeypatch: MonkeyPatch
):
    s = NpzWeightsSerializer(compressed=compressed)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(np, "load", get_error_function(ValueError))
        s.deserialize(b"")

    with pytest.raises(SerializationError):
        monkeypatch.setattr(np, "load", get_error_function(IOError))
        s.deserialize(b"")


def test_weights_serializer_builder_returns_npz_serializer():
    config = WeightsSerializerConfig.parse_obj(
        {"type": "npz", "params": {"compressed": False}}
    )

    serializer = WeightsSerializerBuilder.from_config(config)
    assert isinstance(serializer, NpzWeightsSerializer)
    assert serializer.compressed is False


def test_weights_serializer_builder_throws_not_implemented_error():
    class FakeConfig(pydantic.BaseModel):
        type: str
        params: Dict

    fake_config = FakeConfig(type="does-not-exist", params={})
    with pytest.raises(NotImplementedError):
        WeightsSerializerBuilder.from_config(fake_config)


def test_deserialize_parameters_correct():
    weights = [np.random.rand(10, 10, 15), np.random.rand(99, 4, 3)]
    serializer = NpzWeightsSerializer(compressed=True)
    blob_bytes = serializer.serialize(weights)
    blob_string = Base64StringConverter.to_str(blob_bytes)
    params_serialized = SerializedParameters(
        blob=blob_string,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig(compressed=True)
        ),
        string_format=BinaryStringFormat.BASE64,
    )

    parameters = deserialize_parameters(params_serialized)
    assert are_weights_equal(parameters, weights)


def test_deserialize_parameters_works_without_string_format_by_default():
    weights = [np.random.rand(10, 10, 15), np.random.rand(99, 4, 3)]
    serializer = NpzWeightsSerializer()
    blob_bytes = serializer.serialize(weights)
    params_serialized = SerializedParameters(
        blob=blob_bytes,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig()
        ),
    )
    parameters = deserialize_parameters(params_serialized)
    assert are_weights_equal(parameters, weights)


class SerializerStub(ModelSerializer):
    def __init__(self):
        self.model = None

    def serialize(self, model):
        self.model = model
        return b""

    def deserialize(self, blob: bytes):
        return self.model


def test_model_loader_builder_raises_not_implemented_error():
    config = MagicMock(ModelLoaderConfig)
    config.type = "does-not-exist"

    with pytest.raises(NotImplementedError):
        ModelLoaderBuilder.from_config(config)


@patch("fedless.serialization.PayloadModelLoader")
def test_model_loader_builder_returns_payload_loader(payload_model_loader_mock):
    class LoaderStub:
        pass

    config_mock = ModelLoaderConfig(
        type="payload", params=PayloadModelLoaderConfig(payload="abc123")
    )

    payload_model_loader_mock.from_config.return_value = loader_stub = LoaderStub()

    model_loader = ModelLoaderBuilder.from_config(config_mock)
    assert model_loader == loader_stub
    payload_model_loader_mock.from_config.assert_called_with(config_mock.params)


@patch("fedless.serialization.SimpleModelLoader")
def test_model_loader_builder_returns_simple_loader(simple_model_loader_mock):
    class LoaderStub:
        pass

    class ConfigStub(pydantic.BaseModel):
        type: str
        params: Dict

    config_mock = ConfigStub(type="simple", params={})

    simple_model_loader_mock.from_config.return_value = loader_stub = LoaderStub()

    model_loader = ModelLoaderBuilder.from_config(config_mock)
    assert model_loader == loader_stub
    simple_model_loader_mock.from_config.assert_called_with(config_mock.params)


@patch("fedless.common.models.params_validate_types_match")
def test_model_loader_config_types_match(params_validate_types_match):
    payload_config = MagicMock(PayloadModelLoaderConfig)
    with pytest.raises(pydantic.ValidationError):
        ModelLoaderConfig(type="other", params=payload_config)
    assert params_validate_types_match.called_at_least_once


def test_model_loader_config_only_accepts_valid_configs():
    class FakeConfig(pydantic.BaseModel):
        type: str = "does-not-exist"
        attr: int

    with pytest.raises(pydantic.ValidationError):
        # noinspection PyTypeChecker
        ModelLoaderConfig(type="does-not-exist", params=FakeConfig(attr=2))


def test_payload_model_loader_config_type_fixed():
    with pytest.raises(pydantic.ValidationError):
        PayloadModelLoaderConfig(type="something-else", payload="")


def test_payload_model_loader_config_from_dict():
    config_dict = {
        "payload": "abc",
    }
    config = PayloadModelLoaderConfig.parse_obj(config_dict)
    assert config is not None
    assert config.payload == "abc"


def test_payload_model_loader_from_config_correct():
    with patch.object(ModelSerializerBuilder, "from_config") as from_config_mock:
        from_config_mock.return_value = serializer_stub = SerializerStub()

        config = PayloadModelLoaderConfig(
            payload="abc", serializer=ModelSerializerConfig(type="h5")
        )
        loader = PayloadModelLoader.from_config(config)
    assert loader is not None
    assert loader.payload == "abc"
    assert loader.serializer == serializer_stub


def test_payload_model_loader_fails_on_invalid_serializer():
    with pytest.raises(NotImplementedError):
        with patch.object(ModelSerializerBuilder, "from_config") as from_config_mock:
            from_config_mock.side_effect = NotImplementedError()

            PayloadModelLoader.from_config(
                PayloadModelLoaderConfig(
                    payload="abc", serializer=ModelSerializerConfig(type="h5")
                )
            )


@patch("fedless.serialization.Base64StringConverter")
@patch("fedless.serialization.ModelSerializerBuilder")
def test_payload_model_loader_works_correctly(
    serializer_builder_mock, string_converter_mock, simple_model
):
    serializer_stub = MagicMock(SerializerStub())
    serializer_stub.deserialize.return_value = simple_model
    serializer_builder_mock.from_config.return_value = serializer_stub
    string_converter_mock.from_str.return_value = b"0123abc"

    loader = PayloadModelLoader.from_config(
        PayloadModelLoaderConfig(
            payload="abc", serializer=ModelSerializerConfig(type="h5")
        )
    )

    model: tf.keras.Model = loader.load()

    string_converter_mock.from_str.assert_called_with("abc")
    serializer_stub.deserialize.assert_called_with(b"0123abc")
    assert model == simple_model


@patch("fedless.serialization.ModelSerializerBuilder")
def test_payload_model_loader_works_for_raw_bytes(
    serializer_builder_mock, simple_model
):
    serializer_stub = MagicMock(SerializerStub())
    serializer_stub.deserialize.return_value = simple_model
    serializer_builder_mock.from_config.return_value = serializer_stub

    loader = PayloadModelLoader.from_config(
        PayloadModelLoaderConfig(
            payload=b"abc-test", serializer=ModelSerializerConfig(type="h5")
        )
    )

    model: tf.keras.Model = loader.load()

    serializer_stub.deserialize.assert_called_with(b"abc-test")
    assert model == simple_model


@patch("fedless.serialization.Base64StringConverter")
@patch("fedless.serialization.ModelSerializerBuilder")
def test_payload_model_loader_throws_model_error_when_serializer_fails(
    serializer_builder_mock, string_converter_mock
):
    with pytest.raises(ModelLoadError):
        string_converter_mock.from_str.return_value = b"0123abc"
        serializer_stub = MagicMock(SerializerStub())
        serializer_stub.deserialize.side_effect = SerializationError()
        serializer_builder_mock.from_config.return_value = serializer_stub

        loader = PayloadModelLoader.from_config(
            PayloadModelLoaderConfig(
                payload="abc", serializer=ModelSerializerConfig(type="h5")
            )
        )

        tf.keras.Model = loader.load()


@patch("fedless.serialization.Base64StringConverter")
def test_payload_model_loader_throws_model_error_for_invalid_payload(
    string_converter_mock,
):
    with pytest.raises(ModelLoadError):
        string_converter_mock.from_str.side_effect = ValueError()

        loader = PayloadModelLoader.from_config(
            PayloadModelLoaderConfig(
                payload="abc", serializer=ModelSerializerConfig(type="h5")
            )
        )

        tf.keras.Model = loader.load()


@patch("fedless.serialization.deserialize_parameters")
def test_simple_model_loader_works_works_and_compiles_model(
    deserialize_parameters_mock,
    simple_model: tf.keras.Model,
):
    deserialize_parameters_mock.return_value = simple_model.get_weights()

    loader = SimpleModelLoader(
        parameters=None,
        model=simple_model.to_json(),
        compiled=True,
        optimizer=tf.keras.optimizers.serialize(simple_model.optimizer),
        loss="mse",
        # metrics=simple_model.compiled_metrics.metrics,
    )

    model = loader.load()
    assert are_weights_equal(model.get_weights(), simple_model.get_weights())
    assert type(model.optimizer) == type(simple_model.optimizer)


@patch("fedless.serialization.deserialize_parameters")
def test_simple_model_loader_works_without_compiling_model(
    deserialize_parameters_mock, simple_model: tf.keras.Model
):
    deserialize_parameters_mock.return_value = simple_model.get_weights()

    loader = SimpleModelLoader(
        parameters=None,
        model=simple_model.to_json(),
        compiled=False,
    )

    model = loader.load()
    assert are_weights_equal(model.get_weights(), simple_model.get_weights())


@patch("fedless.serialization.deserialize_parameters")
def test_simple_model_loader_throws_correct_errors(
    deserialize_parameters_mock, simple_model: tf.keras.Model
):

    with pytest.raises(ModelLoadError):
        deserialize_parameters_mock.return_value = simple_model.get_weights()
        SimpleModelLoader(
            parameters=None,
            model="invalid-json:'",
            compiled=False,
        ).load()

    with pytest.raises(ModelLoadError):
        deserialize_parameters_mock.side_effect = SerializationError
        SimpleModelLoader(
            parameters=None,
            model=simple_model.to_json(),
            compiled=False,
        ).load()


def test_serialize_model_throws_error():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(5, input_shape=(3,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Softmax(),
        ]
    )
    with pytest.raises(SerializationError):
        serialize_model(model)


def test_serialize_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(5, input_shape=(3,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Softmax(),
        ]
    )
    opt = tf.keras.optimizers.Adam(lr=0.2)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=ReductionV2.NONE)
    model.compile(loss=loss, optimizer=opt, metrics=["Accuracy"])
    result = serialize_model(model)
    assert result.model_json == model.to_json()
    assert tf.keras.optimizers.get(result.optimizer).lr == 0.2
    assert tf.keras.losses.get(result.loss).reduction == ReductionV2.NONE
