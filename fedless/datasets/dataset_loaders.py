import abc
import logging
from functools import reduce
from typing import Iterator

import tensorflow as tf


logger = logging.getLogger(__name__)


class DatasetNotLoadedError(Exception):
    """Dataset could not be loaded"""


class DatasetFormatError(DatasetNotLoadedError):
    """Source file containing data is malformed or otherwise invalid"""


def merge_datasets(datasets: Iterator[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Merge the given datasets into one by concatenating them
    :param datasets: Iterator with all datasets
    :return: Final combined dataset
    :raises TypeError in tf.data.Dataset.concatenate
    """
    return reduce(tf.data.Dataset.concatenate, datasets)


class DatasetLoader(abc.ABC):
    """Load arbitrary datasets"""

    @abc.abstractmethod
    def load(self) -> tf.data.Dataset:
        """Load dataset"""
        pass
