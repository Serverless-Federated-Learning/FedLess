from fedless.common.models import DatasetLoaderConfig
from fedless.datasets.char_prediction.dataset_loader import (
    CharacterPrediction,
    CharacterPredictionConfig,
)
from fedless.datasets.cifar.dataset_loader import CIFAR, CIFARConfig
from fedless.datasets.dataset_loaders import DatasetLoader
from fedless.datasets.emnist.dataset_loader import EMNIST, EMNISTConfig
from fedless.datasets.fedscale.google_speech.dataset_loader import (
    FedScale,
    FedScaleConfig,
)
from fedless.datasets.leaf.dataset_loader import LEAF, LEAFConfig
from fedless.datasets.mnist.dataset_loader import MNIST, MNISTConfig


class DatasetLoaderBuilder:
    """Convenience class to construct loaders from config"""

    @staticmethod
    def from_config(config: DatasetLoaderConfig) -> DatasetLoader:
        """
        Construct loader from config
        :raises NotImplementedError if the loader does not exist
        """
        if config.type == "leaf":
            params: LEAFConfig = config.params
            return LEAF(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
                user_indices=params.user_indices,
            )
        elif config.type == "mnist":
            params: MNISTConfig = config.params
            # location is added by default here
            return MNIST(
                split=params.split,
                indices=params.indices,
                proxies=params.proxies,
            )
        elif config.type == "speech":
            params: FedScaleConfig = config.params
            return FedScale(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
                user_indices=params.user_indices,
            )
        elif config.type == "cifar":
            params: CIFARConfig = config.params
            return CIFAR(
                dataset=params.dataset,
                label_mapping=params.label_mapping,
                split=params.split,
                indices=params.indices,
                proxies=params.proxies,
                dataset_url=params.dataset_url,
            )
        elif config.type == "emnist":
            params: EMNISTConfig = config.params
            return EMNIST(
                label_mapping=params.label_mapping,
                split=params.split,
                indices=params.indices,
                proxies=params.proxies,
                dataset_url=params.dataset_url,
            )
        elif config.type == "char_prediction":
            params: CharacterPredictionConfig = config.params
            return CharacterPrediction(
                location=params.location,
                sequence_length=params.sequence_length,
                http_params=params.http_params,
                steps=params.steps,
                dataset_name=params.dataset_name,
                user_indices=params.user_indices,
                max_sequences=params.max_sequences,
            )
        else:
            raise NotImplementedError(f"Dataset loader {config.type} is not implemented")
