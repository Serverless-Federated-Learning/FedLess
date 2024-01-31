import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import requests
import requests_mock
from pydantic import ValidationError


from fedless.common.cache import _clear_cache
from fedless.common.models import DatasetLoaderConfig
from fedless.datasets.benchmark_configurator import DatasetLoaderBuilder
from fedless.datasets.dataset_loaders import DatasetFormatError, DatasetNotLoadedError
from fedless.datasets.leaf.dataset_loader import LEAF, LEAFConfig, LeafDataset
from .common import resource_folder_path, get_error_function

LEAF_RES_PATH = resource_folder_path() / "leaf"
FEMNIST_FILE_PATH = LEAF_RES_PATH / "femnist" / "femnist_test_data.json"
FEMNIST_FILE_CONTENT_DICT = json.loads(FEMNIST_FILE_PATH.read_text())
DUMMY_FEMNIST_FILE_URL = "https://fileserver/data/file.json"


def test_leaf_loader_iters_files_correctly():
    files = list(
        LEAF(dataset="femnist", location=FEMNIST_FILE_PATH)._iter_dataset_files()
    )
    assert files == [FEMNIST_FILE_PATH]

    files = list(
        LEAF(
            dataset="femnist", location=LEAF_RES_PATH / "femnist"
        )._iter_dataset_files()
    )
    assert files == [FEMNIST_FILE_PATH]

    # noinspection PyTypeChecker
    files = list(
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._iter_dataset_files()
    )
    assert files == [DUMMY_FEMNIST_FILE_URL]


def test_leaf_loader_handles_timeout(requests_mock: requests_mock.Mocker):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, exc=requests.exceptions.Timeout)
    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
            DUMMY_FEMNIST_FILE_URL
        )


def test_leaf_loader_handles_request_error(requests_mock: requests_mock.Mocker):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, exc=requests.exceptions.RequestException)
    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
            DUMMY_FEMNIST_FILE_URL
        )


def test_leaf_loader_handles_http_error(requests_mock: requests_mock.Mocker):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, exc=requests.exceptions.HTTPError)
    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
            DUMMY_FEMNIST_FILE_URL
        )


def test_leaf_loader_handles_url_error(requests_mock: requests_mock.Mocker):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, exc=requests.exceptions.InvalidURL)
    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
            DUMMY_FEMNIST_FILE_URL
        )


def test_leaf_loader_handles_too_many_redirects_error(
    requests_mock: requests_mock.Mocker,
):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, exc=requests.exceptions.TooManyRedirects)
    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
            DUMMY_FEMNIST_FILE_URL
        )


def test_leaf_loader_fetches_file_correctly(
    requests_mock: requests_mock.Mocker,
):
    dummy_data = {"att1": "value"}
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, json=dummy_data)

    returned_dict = LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
        DUMMY_FEMNIST_FILE_URL
    )
    assert dummy_data == returned_dict


def test_leaf_loader_parses_returned_json(
    requests_mock: requests_mock.Mocker,
):
    invalid_data = b'{att1: value"}'
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, content=invalid_data)

    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)._fetch_url(
            DUMMY_FEMNIST_FILE_URL
        )


def test_leaf_loader_loads_file():
    data = LEAF._read_file_content(FEMNIST_FILE_PATH)
    assert set(data.keys()) == {"users", "user_data", "num_samples"}


def test_leaf_loader_raises_error_on_invalid_path():
    with pytest.raises(DatasetNotLoadedError):
        LEAF._read_file_content(LEAF_RES_PATH / "femnist")

    with pytest.raises(DatasetNotLoadedError):
        LEAF._read_file_content(LEAF_RES_PATH / "femnist" / "does-not-exist.txt")


def test_leaf_loader_raises_error_on_invalid_file():
    with pytest.raises(DatasetNotLoadedError):
        LEAF._read_file_content(LEAF_RES_PATH / "femnist")

    with pytest.raises(DatasetNotLoadedError):
        LEAF._read_file_content(LEAF_RES_PATH / "femnist" / "invalid_file.txt")


def test_leaf_loader_uses_http_params():
    with requests_mock.Mocker(real_http=False) as m:
        m.register_uri(
            "GET",
            "http://url-does-not-exist-fedless.gripe?a=1&b=2",
            content=FEMNIST_FILE_PATH.read_bytes(),
            complete_qs=True,
        )
        loader = LEAF(
            dataset="femnist",
            location="http://url-does-not-exist-fedless.gripe",
            http_params={"a": 1, "b": 2},
        )
        loader.load()


def test_leaf_loader_works_for_url(requests_mock: requests_mock.Mocker):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, content=FEMNIST_FILE_PATH.read_bytes())
    loader = LEAF(dataset="femnist", location=DUMMY_FEMNIST_FILE_URL)
    data = loader._read_source(loader.source)
    assert set(data.keys()) == {"users", "user_data", "num_samples"}


def test_leaf_loader_works_for_file():
    loader = LEAF(dataset="femnist", location=FEMNIST_FILE_PATH)
    data = loader._read_source(loader.source)
    assert set(data.keys()) == {"users", "user_data", "num_samples"}


def test_leaf_femnist_converter_handles_invalid_file():
    invalid_content_dict = {"users": ["a"], "user_data": {"b": {}}}
    with pytest.raises(DatasetFormatError):
        list(
            LEAF(
                dataset="femnist", location=FEMNIST_FILE_PATH
            )._convert_dict_to_dataset((invalid_content_dict))
        )

        invalid_content_dict = {
            "users": ["a"],
            "user_data": {"a": {"x": "value"}},
        }
        with pytest.raises(DatasetFormatError):
            list(LEAF._convert_dict_to_dataset(invalid_content_dict))


@pytest.mark.parametrize(
    "location", [FEMNIST_FILE_PATH, FEMNIST_FILE_PATH.parent, DUMMY_FEMNIST_FILE_URL]
)
def test_leaf_femnist_loads_correctly(location, requests_mock: requests_mock.Mocker):
    requests_mock.get(DUMMY_FEMNIST_FILE_URL, content=FEMNIST_FILE_PATH.read_bytes())
    true_labels = np.array([33, 33, 11, 32, 35, 38, 54, 56], dtype=np.int)

    loader = LEAF(dataset="femnist", location=location)
    dataset = loader.load()
    labels = [labels for features, labels in dataset.as_numpy_iterator()]

    assert len(dataset) == 8
    assert np.equal(true_labels, labels).all()


def test_leaf_femnist_loads_only_specified_user():
    true_labels = np.array([38, 54, 56], dtype=np.int)

    loader = LEAF(dataset="femnist", location=FEMNIST_FILE_PATH, user_indices=[1])
    dataset = loader.load()
    labels = [labels for features, labels in dataset.as_numpy_iterator()]

    assert len(dataset) == 3
    assert np.equal(true_labels, labels).all()


@patch("tensorflow.data.Dataset.concatenate")
def test_leaf_femnist_throws_error_when_concatenation_fails(concat_mock):
    _clear_cache()
    concat_mock.side_effect = get_error_function(TypeError)

    with pytest.raises(DatasetNotLoadedError):
        LEAF(dataset="femnist", location=FEMNIST_FILE_PATH).load()


@pytest.mark.parametrize(
    "location", [FEMNIST_FILE_PATH, FEMNIST_FILE_PATH.parent, DUMMY_FEMNIST_FILE_URL]
)
def test_dataset_loader_builder_returns_leaf_loader(location):
    config = DatasetLoaderConfig(
        type="leaf",
        params=LEAFConfig(dataset="femnist", location=location, http_params=None),
    )
    # noinspection PyTypeChecker
    loader: LEAF = DatasetLoaderBuilder.from_config(config)
    assert loader.dataset == LeafDataset.FEMNIST
    assert loader.source == location
    assert loader.http_params is None


def test_dataset_loader_builder_raises_not_implemented_error():
    config = MagicMock(DatasetLoaderConfig)
    config.type = "does-not-exist"

    with pytest.raises(NotImplementedError):
        DatasetLoaderBuilder.from_config(config)


def test_leaf_config_validates_fields():
    location = "www.google.com/test"
    http_params = None
    dataset = "femnist"
    with pytest.raises(ValidationError):
        LEAFConfig(
            dataset="invalid-dataset", location=location, http_params=http_params
        )

    with pytest.raises(ValidationError):
        LEAFConfig(dataset=dataset, location=location, http_params="invalid")
