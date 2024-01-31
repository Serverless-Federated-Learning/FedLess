from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

from fedless.datasets.dataset_loaders import DatasetLoader

from typing import Union, Dict, List, Optional

import tensorflow as tf
from pydantic import BaseModel, validate_arguments, AnyHttpUrl

from fedless.common.cache import cache

from enum import Enum
from pydantic import Field


class FedScaleDataset(str, Enum):
    """
    Officially supported datasets
    """

    SPEECH = "speech"


class FedScaleConfig(BaseModel):
    """Configuration parameters for LEAF dataset loader"""

    type: str = Field("speech", const=True)
    dataset: FedScaleDataset
    location: Union[AnyHttpUrl, Path]
    http_params: Dict = None
    user_indices: Optional[List[int]] = None


class FedScale(DatasetLoader):
    @validate_arguments
    def __init__(
        self,
        dataset: FedScaleDataset,
        location: Union[AnyHttpUrl, Path],
        http_params: Dict = None,
        user_indices: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.source = location
        self.http_params = http_params
        self.user_indices = user_indices

    @cache
    def load(self) -> tf.data.Dataset:
        """
        Load dataset
        :raise DatasetNotLoadedError when an error occurred in the process
        """
        tx_file_path = tf.keras.utils.get_file(cache_subdir="data", origin=self.source)
        with np.load(tx_file_path) as all_data:
            data = all_data["data"]
            labels = all_data["labels"]
            dataset = tf.data.Dataset.from_tensor_slices((data, labels))
            return dataset
