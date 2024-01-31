from __future__ import annotations

import abc
import base64
import binascii
import io
from json.decoder import JSONDecodeError
from typing import TypeVar, Callable, Union, Optional, List, Dict

import h5py
import numpy as np
import tensorflow as tf

from fedless.common.cache import cache
from fedless.common.models import (
    ModelSerializerConfig,
    WeightsSerializerConfig,
    PayloadModelLoaderConfig,
    SimpleModelLoaderConfig,
    ModelLoaderConfig,
    H5FullModelSerializerConfig,
    SerializedParameters,
    Parameters,
    BinaryStringFormat,
    NpzWeightsSerializerConfig,
    SerializedModel,
)


class SerializationError(Exception):
    """Object could not be (de)serialized"""


_RetT = TypeVar("_RetT")


def serialize_model(model: tf.keras.Model) -> SerializedModel:
    if not model.optimizer or not model.loss:
        raise SerializationError(f"Cannot serialize model as it is not compiled")

    user_metrics = getattr(model.compiled_metrics, "_user_metrics", [])

    return SerializedModel(
        model_json=model.to_json(),
        optimizer=tf.keras.optimizers.serialize(model.optimizer),
        loss=(
            tf.keras.losses.serialize(model.loss)
            if not isinstance(model.loss, str)
            else model.loss
        ),
        metrics=user_metrics,
    )


def h5py_serialization_error_handler(
    func: Callable[..., _RetT]
) -> Callable[..., _RetT]:
    """
    Executes the function and wraps and rethrows any unhandled exception as a SerializationError
    :param func: Any serialization function
    :return: wrapped function
    """

    # noinspection PyMissingOrEmptyDocstring
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            raise
        except Exception as e:
            # Catching general exceptions is bad practice but according to
            # https://docs.h5py.org/en/stable/faq.html#exceptions, we do not know which exceptions can be thrown
            # when opening/closing h5py files. As we're only dealing with in-memory representations, this should
            # hopefully be acceptable with the exception clause above
            raise SerializationError(e) from e

    return new_func


def deserialize_parameters(serialized_params: SerializedParameters) -> Parameters:
    """Deserialize parameters with specified serialization method"""
    serializer = WeightsSerializerBuilder.from_config(serialized_params.serializer)

    if serialized_params.string_format == BinaryStringFormat.BASE64:
        blob_bytes = Base64StringConverter.from_str(serialized_params.blob)
    elif serialized_params.string_format == BinaryStringFormat.NONE:
        blob_bytes = serialized_params.blob
    else:
        raise SerializationError(
            f"Binary string format {serialized_params.string_format} not known"
        )

    return serializer.deserialize(blob_bytes)


def wrap_exceptions_as_serialization_error(
    *exceptions: Exception.__class__,
) -> Callable[[Callable[..., _RetT]], Callable[..., _RetT]]:
    """
    Wrap and rethrow all specified exceptions as serialization errors.
    Can be used as a function decorator
    """

    def function_decorator(func: Callable[..., _RetT]) -> Callable[..., _RetT]:
        """Actual Function decorator responsible to wrap exceptions"""

        # noinspection PyMissingOrEmptyDocstring
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tuple(exceptions) as e:
                raise SerializationError() from e

        return new_func

    return function_decorator


class StringSerializer(abc.ABC):
    """Convert raw bytes into string representation and vice versa"""

    @staticmethod
    @abc.abstractmethod
    def get_format() -> BinaryStringFormat:
        pass

    @staticmethod
    @abc.abstractmethod
    def to_str(obj: bytes) -> str:
        """Convert raw bytes to string"""
        pass

    @staticmethod
    @abc.abstractmethod
    def from_str(rep: str) -> bytes:
        """Reconstruct raw bytes from string representation"""
        pass


class Base64StringConverter(StringSerializer):
    """
    Represents raw bytes as Base64 string representation. All created strings are valid ascii,
    this class can therefore also be used to send raw bytes in json payload via http
    """

    @staticmethod
    def get_format() -> BinaryStringFormat:
        return BinaryStringFormat.BASE64

    @staticmethod
    def to_str(obj: bytes) -> str:
        """
        Convert bytes object to base64 string
        :param obj: bytes
        :return: Base64 / ASCII string
        """
        encoding = base64.b64encode(obj)
        return encoding.decode(encoding="ascii")

    @staticmethod
    def from_str(rep: str) -> bytes:
        """
        Decodes Base64/ Ascii representation to original raw bytes
        :param rep: Base64/ ASCII string
        :return: bytes
        :raises ValueError if input is incorrectly padded or otherwise invalid
        """
        try:
            return base64.b64decode(rep)
        except binascii.Error:
            raise ValueError("Given string is not in base64 or incorrectly padded")


class ModelSerializer(abc.ABC):
    """
    Serialize tensorflow.keras.Model objects. Preserves architecture, parameters and optimizer state.
    Please be aware that not possibly all implemented methods support custom components (layers, models, ...)
    """

    @abc.abstractmethod
    def serialize(self, model: tf.keras.Model) -> bytes:
        """Convert model into bytes"""
        pass

    @abc.abstractmethod
    def deserialize(self, blob: bytes) -> tf.keras.Model:
        """Reconstruct model from raw bytes"""
        pass


class H5FullModelSerializer(ModelSerializer):
    """
    Serializes the full model as an HDF5 file-string
    """

    def __init__(self, save_traces: bool = True):
        super().__init__()
        self.save_traces = save_traces

    @classmethod
    def from_config(
        cls, config: H5FullModelSerializerConfig
    ) -> H5FullModelSerializerConfig:
        """
        Create serializer from config
        :param config: configuration object
        :return: instantiated serializer
        """
        options = config.dict(exclude={"type"})
        return cls(**options)

    @h5py_serialization_error_handler
    def serialize(self, model: tf.keras.Model) -> bytes:
        """
        Save model, including parameters and optimizer state, as raw bytes of h5py file
        :param model: Keras Model
        :return: raw bytes
        """
        with h5py.File(
            "does not matter", mode="w", driver="core", backing_store=False
        ) as h5file:
            model.save(
                filepath=h5file,
                include_optimizer=True,
                save_traces=self.save_traces,
            )
            return h5file.id.get_file_image()

    @h5py_serialization_error_handler
    def deserialize(self, blob: bytes) -> tf.keras.Model:
        """
        Reconstruct keras model from raw h5py file representation
        :param blob: bytes
        :return: Keras Model
        """
        fid = h5py.h5f.open_file_image(blob)
        with h5py.File(fid, mode="r+") as h5file:
            loaded_model = tf.keras.models.load_model(h5file)
        return loaded_model


class ModelSerializerBuilder:
    """Convenience class to directly create a serializer from its config"""

    @staticmethod
    def from_config(config: ModelSerializerConfig) -> ModelSerializer:
        """
        Create serializer from config
        :raises NotImplementedError if this serializer does not exist
        """
        if config.type == "h5":
            params: Optional[H5FullModelSerializerConfig] = config.params
            if config.params is not None:
                return H5FullModelSerializer.from_config(params)
            else:
                return H5FullModelSerializer()
        else:
            raise NotImplementedError(
                f"Serializer of type {config.type} does not exist"
            )


class WeightsSerializer(abc.ABC):
    """Serialize model parameters/ list of numpy arrays as bytes"""

    @abc.abstractmethod
    def serialize(self, weights: List[np.ndarray]) -> bytes:
        """Convert into raw bytes"""
        pass

    @abc.abstractmethod
    def deserialize(self, blob: bytes) -> List[np.ndarray]:
        """Convert bytes into parameters"""
        pass

    @abc.abstractmethod
    def get_config(self) -> WeightsSerializerConfig:
        """Get Configuration object from instance"""
        pass


class NpzWeightsSerializer(WeightsSerializer):
    """Serialize model parameters as numpy npz object"""

    def __init__(self, compressed: bool = False):
        self.compressed = compressed

    def get_config(self) -> WeightsSerializerConfig:
        """Get Configuration object from instance"""

        parameters = NpzWeightsSerializerConfig(compressed=self.compressed)
        return WeightsSerializerConfig(type=parameters.type, params=parameters)

    def serialize(self, weights: List[np.ndarray]) -> bytes:
        """Convert into raw bytes, also supports basic compression"""
        with io.BytesIO() as f:
            if self.compressed:
                np.savez_compressed(f, *weights)
            else:
                np.savez(f, *weights)
            return f.getvalue()

    @wrap_exceptions_as_serialization_error(ValueError, IOError)
    def deserialize(self, blob: bytes) -> List[np.ndarray]:
        """Convert bytes into parameters"""
        with io.BytesIO(blob) as f:
            npz_obj = np.load(f)
            return list(npz_obj.values())


class WeightsSerializerBuilder:
    """Convenience class to directly create a serializer from its config"""

    @staticmethod
    def from_config(config: WeightsSerializerConfig) -> WeightsSerializer:
        """
        Create serializer from config
        :raises NotImplementedError if this serializer does not exist
        """
        if config.type == "npz":
            params: NpzWeightsSerializerConfig = config.params
            return NpzWeightsSerializer(compressed=params.compressed)
        else:
            raise NotImplementedError(
                f"Serializer of type {config.type} does not exist"
            )


class ModelLoadError(Exception):
    """Model could not be loaded"""


class ModelLoader(abc.ABC):
    """Load keras model from arbitrary source"""

    @abc.abstractmethod
    def load(self) -> tf.keras.Model:
        """Load model"""
        pass


class PayloadModelLoader(ModelLoader):
    """
    Send serialized models directly as part of the configuration object.
    Not advisable for large models.
    """

    def __init__(self, payload: Union[str, bytes], serializer: ModelSerializer):
        self.payload = payload
        self.serializer = serializer

    @classmethod
    def from_config(cls, config: PayloadModelLoaderConfig) -> PayloadModelLoader:
        """Create loader from :class:`PayloadModelLoaderConfig`"""
        payload = config.payload
        serializer = ModelSerializerBuilder.from_config(config.serializer)
        return cls(payload=payload, serializer=serializer)

    def load(self) -> tf.keras.Model:
        """
        Deserialize payload and return model
        :raises ModelLoadError if payload is invalid or other error occurred during deserialization
        """
        try:
            if isinstance(self.payload, str):
                raw_bytes = Base64StringConverter.from_str(self.payload)
            else:
                raw_bytes = self.payload
            return self.serializer.deserialize(raw_bytes)
        except SerializationError as e:
            raise ModelLoadError("Model could not be deserialized") from e
        except ValueError as e:
            raise ModelLoadError("Malformed or otherwise invalid payload") from e


class SimpleModelLoader(ModelLoader):
    """Simply loads model from its architecture and serialized parameters"""

    def __init__(
        self,
        parameters: SerializedParameters,
        model: str,
        compiled: bool,
        optimizer: Optional[Union[str, Dict]] = None,
        loss: Optional[Union[str, Dict]] = None,
        metrics: Optional[List[str]] = None,
    ):
        self._parameters = parameters
        self.model = model
        self.compiled = compiled
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    @classmethod
    def from_config(cls, config: SimpleModelLoaderConfig) -> SimpleModelLoader:
        """Create loader from :class:`SimpleModelLoaderConfig`"""

        return SimpleModelLoader(
            parameters=config.params,
            model=config.model,
            compiled=config.compiled,
            optimizer=config.optimizer,
            loss=config.loss,
            metrics=config.metrics,
        )

    @cache
    def _load_except_weights(self) -> tf.keras.Model:
        try:
            model: tf.keras.Model = tf.keras.models.model_from_json(self.model)

            # Compile if specified
            if self.compiled:
                try:
                    if not self.loss or not self.optimizer:
                        raise ModelLoadError(
                            "If compiled=True, a loss has to be specified"
                        )
                    loss = tf.keras.losses.get(self.loss)
                    metrics = self.metrics or []
                    optimizer = tf.keras.optimizers.get(self.optimizer)

                    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                except ValueError as e:
                    raise ModelLoadError(e) from e

            return model

        except (JSONDecodeError, ValueError) as e:
            raise ModelLoadError(
                "Malformed or otherwise invalid model architecture. "
                "Check if model config is malformed or shapes of parameters do not match"
            )

    def load(self) -> tf.keras.Model:
        """Reconstruct model from config, deserialize parameters, and optionally compile model"""
        try:
            model = self._load_except_weights()
            weights = deserialize_parameters(self._parameters)
            model.set_weights(weights)
            return model
        except SerializationError as e:
            raise ModelLoadError("Weights could not be deserialized") from e


class ModelLoaderBuilder:
    """Convenience class to create loader from :class:`ModelLoaderConfig`"""

    @staticmethod
    def from_config(config: ModelLoaderConfig):
        """
        Construct a model loader from the config
        :raises NotImplementedError if the loader does not exist
        """
        if config.type == "payload":
            params: PayloadModelLoaderConfig = config.params
            return PayloadModelLoader.from_config(params)
        if config.type == "simple":
            params: SimpleModelLoaderConfig = config.params
            return SimpleModelLoader.from_config(params)
        else:
            raise NotImplementedError(f"Model loader {config.type} is not implemented")
