from typing import Iterable, Optional, Dict, List, Iterator


from fedless.datasets.dataset_loaders import (
    DatasetFormatError,
    DatasetLoader,
    DatasetNotLoadedError,
    merge_datasets,
)

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Union, Dict, Iterator, List, Optional, Tuple

import requests
import tensorflow as tf
from pydantic import BaseModel, validate_arguments, AnyHttpUrl
from requests import RequestException

from fedless.common.cache import cache

from enum import Enum
from pydantic import Field


class LeafDataset(str, Enum):
    """
    Officially supported datasets
    """

    FEMNIST = "femnist"
    REDDIT = "reddit"
    CELEBA = "celeba"
    SHAKESPEARE = "shakespeare"
    SENT140 = "sent140"


class LEAFConfig(BaseModel):
    """Configuration parameters for LEAF dataset loader"""

    type: str = Field("leaf", const=True)
    dataset: LeafDataset
    location: Union[AnyHttpUrl, Path]
    http_params: Dict = None
    user_indices: Optional[List[int]] = None


class LEAF(DatasetLoader):
    """
    Utility class to load and process the LEAF datasets as published in
    https://arxiv.org/pdf/1812.01097.pdf and https://github.com/TalwalkarLab/leaf
    """

    @validate_arguments
    def __init__(
        self,
        dataset: LeafDataset,
        location: Union[AnyHttpUrl, Path],
        http_params: Dict = None,
        user_indices: Optional[List[int]] = None,
    ):
        """
        Create dataset loader for the specified source
        :param dataset: Dataset name, one of :py:class:`fedless.common.models.LeafDataset`
        :param location: Location of dataset partition in form of a json file.
        :param http_params: Additional parameters to send with http request. Only used when location is an URL
         Use location:// to load from disk. For valid entries see :py:meth:`requests.get`
        """
        self.dataset = dataset
        self.source = location
        self.http_params = http_params
        self.user_indices = user_indices
        self._users = []

        if dataset not in [LeafDataset.FEMNIST, LeafDataset.SHAKESPEARE]:
            raise NotImplementedError()

    def _iter_dataset_files(self) -> Iterator[Union[AnyHttpUrl, Path]]:
        if isinstance(self.source, AnyHttpUrl):
            yield self.source
        elif isinstance(self.source, Path) and self.source.is_dir():
            for file in self.source.iterdir():
                if file.is_file() and file.suffix == ".json":
                    yield file
        else:
            yield self.source

    @property
    def users(self):
        return self._users

    def _convert_dict_to_dataset(
        self, file_content: Dict, user_indices: List[int] = None
    ) -> tf.data.Dataset:
        try:
            users = file_content["users"]
            user_data = file_content["user_data"]
            self._users = users
            for i, user in enumerate(users):
                if not user_indices or i in user_indices:
                    yield tf.data.Dataset.from_tensor_slices(
                        self._process_user_data(user_data[user])
                    )
        except (KeyError, TypeError, ValueError) as e:
            raise DatasetFormatError(e) from e

    def _process_user_data(self, user_data: Dict) -> Tuple:
        if self.dataset == LeafDataset.SHAKESPEARE:
            vocabulary = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
            vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
                standardize=None,
                split=tf.strings.bytes_split,
                vocabulary=[c for c in vocabulary],
            )
            return (
                vectorizer(tf.convert_to_tensor(user_data["x"])),
                vectorizer(tf.convert_to_tensor(user_data["y"])),
            )

        return user_data["x"], user_data["y"]

    def _process_all_sources(self) -> Iterator[tf.data.Dataset]:
        for source in self._iter_dataset_files():
            file_content: Dict = self._read_source(source)
            for dataset in self._convert_dict_to_dataset(
                file_content, user_indices=self.user_indices
            ):
                yield dataset

    def _read_source(self, source: Union[AnyHttpUrl, Path]) -> Dict:
        if isinstance(source, AnyHttpUrl):
            return self._fetch_url(source)
        else:
            return self._read_file_content(source)

    def _fetch_url(self, url: str):
        try:
            response = requests.get(url, params=self.http_params)
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            raise DatasetFormatError(f"Invalid JSON returned from {url}") from e
        except RequestException as e:
            raise DatasetNotLoadedError(e) from e

    @classmethod
    def _read_file_content(cls, path: Path) -> Dict:
        try:
            with path.open() as f:
                return json.load(f)
        except (JSONDecodeError, ValueError) as e:
            raise DatasetFormatError(e) from e
        except (IOError, OSError) as e:
            raise DatasetNotLoadedError(e) from e

    @cache
    def load(self) -> tf.data.Dataset:
        """
        Load dataset
        :raise DatasetNotLoadedError when an error occurred in the process
        """
        sources = self._process_all_sources()
        try:
            return merge_datasets(sources)
        except TypeError as e:
            raise DatasetFormatError(e) from e


def split_source_by_users(config: LEAFConfig) -> Iterable[LEAFConfig]:
    loader = LEAF(
        dataset=config.dataset,
        location=config.location,
        http_params=config.http_params,
        user_indices=config.user_indices,
    )
    loader.load()

    for i, _ in enumerate(loader.users):
        if not config.user_indices or i in config.user_indices:
            yield LEAFConfig(
                dataset=config.dataset,
                location=config.location,
                http_params=config.http_params,
                user_indices=[i],
            )


def split_sources_by_users(source_urls: List[LEAFConfig]) -> Iterator[LEAFConfig]:
    for source in source_urls:
        for config in split_source_by_users(source):
            yield config
