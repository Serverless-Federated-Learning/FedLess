from itertools import zip_longest

import pytest
import numpy as np
import tensorflow as tf
from fedless.aggregator.exceptions import UnknownCardinalityError
from fedless.aggregator.fed_avg_aggregator import (
    FedAvgAggregator,
    StreamFedAvgAggregator,
)


from fedless.common.models import (
    ClientResult,
    NpzWeightsSerializerConfig,
    WeightsSerializerConfig,
    SerializedParameters,
    BinaryStringFormat,
)
from fedless.common.serialization import NpzWeightsSerializer, Base64StringConverter


@pytest.fixture
def dummy_parameters():
    return [
        [
            np.array([[[3.0, 0.0, 5.0], [1.0, -5.0, 2.0]]]),
            np.array([[[4.0, 0.0, 5.0], [9.0, -5.0, 2.0]]]),
        ],
        [
            np.array([[[4.0, 2.0, 5.0], [1.0, -8.0, -5.0]]]),
            np.array([[[-2.0, -5.0, 5.0], [1.0, -8.0, -20.0]]]),
        ],
        [
            np.array([[[7.0, 3.0, 9.0], [3.0, -123.0, -4.0]]]),
            np.array([[[7.0, 3.0, 9.0], [3.0, -123.0, -4.0]]]),
        ],
    ]


@pytest.fixture
def dummy_cardinalities():
    return [1.0, 2.0, 0.0]


@pytest.fixture
def dummy_client_results(dummy_parameters, dummy_cardinalities):
    results = []
    for i, params in enumerate(dummy_parameters):
        weights_bytes = NpzWeightsSerializer().serialize(params)
        if i % 2 == 0:
            blob = Base64StringConverter.to_str(weights_bytes)
            result = ClientResult(
                parameters=SerializedParameters(
                    blob=blob,
                    serializer=WeightsSerializerConfig(
                        type="npz", params=NpzWeightsSerializerConfig()
                    ),
                    string_format=BinaryStringFormat.BASE64,
                ),
                cardinality=dummy_cardinalities[i],
            )
        else:
            result = ClientResult(
                parameters=SerializedParameters(
                    blob=weights_bytes,
                    serializer=WeightsSerializerConfig(
                        type="npz", params=NpzWeightsSerializerConfig()
                    ),
                    string_format=BinaryStringFormat.NONE,
                ),
                cardinality=dummy_cardinalities[i],
            )

        results.append(result)
    return results


@pytest.fixture
def dummy_expected_result():
    return [
        np.array([[[3.6666666667, 1.33333333, 5.0], [1.0, -7.0, -2.66666666667]]]),
        np.array(
            [[[0.0, -3.33333333333, 5.0], [3.66666666667, -7.0, -12.666666666667]]]
        ),
    ]


def test_fedavg_aggregate_calculation(
    dummy_parameters, dummy_cardinalities, dummy_expected_result
):
    aggregator = FedAvgAggregator()
    final_params = aggregator._aggregate(
        parameters=dummy_parameters, weights=dummy_cardinalities
    )

    assert all(
        [np.allclose(a, b) for a, b in zip_longest(final_params, dummy_expected_result)]
    )


def test_fedavg_aggregate_function(dummy_client_results, dummy_expected_result):
    aggregator = FedAvgAggregator()
    final_params, _ = aggregator.aggregate(client_results=dummy_client_results)
    assert all(
        [np.allclose(a, b) for a, b in zip_longest(final_params, dummy_expected_result)]
    )


def test_fedavg_throws_error_on_invalid_cardinality(dummy_client_results):
    dummy_client_results[0].cardinality = tf.data.INFINITE_CARDINALITY

    with pytest.raises(UnknownCardinalityError):
        FedAvgAggregator().aggregate(client_results=dummy_client_results)


def test_fedavg_recovers_on_invalid_cardinality(
    dummy_client_results, dummy_expected_result
):
    dummy_client_results[0].cardinality = tf.data.INFINITE_CARDINALITY

    final_params, _ = FedAvgAggregator().aggregate(
        client_results=dummy_client_results, default_cardinality=1.0
    )
    assert all(
        [np.allclose(a, b) for a, b in zip_longest(final_params, dummy_expected_result)]
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 10, 50])
def test_streamfedavg_aggregate_function(
    dummy_client_results, dummy_expected_result, chunk_size
):
    aggregator = StreamFedAvgAggregator(chunk_size=chunk_size)
    final_params, _ = aggregator.aggregate(client_results=dummy_client_results)
    assert all(
        [np.allclose(a, b) for a, b in zip_longest(final_params, dummy_expected_result)]
    )
