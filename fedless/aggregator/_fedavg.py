import os
import time

import numpy as np
from keras.losses import sparse_categorical_crossentropy
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import (
    AttackType,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable tensorflow logs

from multiprocessing import Pool, set_start_method
import random

import click

# import pandas as pd

from fedless.aggregator.aggregation import FedAvgAggregator
from fedless.datasets.benchmark_configurator import DatasetLoaderBuilder
from fedless.datasets.mnist.dataset_loader import MNISTConfig
from fedless.datasets.mnist.helpers import create_mnist_train_data_loader_configs
from fedless.datasets.mnist.model import create_mnist_cnn
from fedless.client import default_handler
from fedless.common.models import (
    Hyperparams,
    ClientInvocationParams,
    DatasetLoaderConfig,
    SimpleModelLoaderConfig,
    ModelLoaderConfig,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    LocalDifferentialPrivacyParams,
)
from fedless.common.serialization import (
    serialize_model,
    NpzWeightsSerializer,
    Base64StringConverter,
)


def simulate_mia(train, test, model):
    from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import (
        membership_inference_attack,
    )
    from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import (
        AttackInputData,
    )

    x_train = train.map(lambda x, y: x)
    x_test = test.map(lambda x, y: x)
    y_train = list(train.map(lambda x, y: y).as_numpy_iterator())
    y_test = list(test.map(lambda x, y: y).as_numpy_iterator())
    train_predict = model.predict(x_train.batch(128))
    test_predict = model.predict(x_test.batch(128))
    loss_train = sparse_categorical_crossentropy(y_train, train_predict).numpy()
    loss_test = sparse_categorical_crossentropy(y_test, test_predict).numpy()
    attacks_result = membership_inference_attack.run_attacks(
        AttackInputData(
            loss_train=loss_train,
            loss_test=loss_test,
            labels_train=np.asanyarray(y_train),
            labels_test=np.asanyarray(y_test),
        ),
        attack_types=[
            AttackType.THRESHOLD_ATTACK,
            # AttackType.RANDOM_FOREST,
            AttackType.LOGISTIC_REGRESSION,
            # AttackType.MULTI_LAYERED_PERCEPTRON,
        ],
    )
    print(attacks_result.summary())
    results = attacks_result.get_result_with_max_attacker_advantage()
    return {
        "mia-auc": results.get_auc(),
        "mia-attacker-advantage": results.get_attacker_advantage(),
    }


@click.command()
@click.option("--devices", type=int, default=100)
@click.option("--epochs", type=int, default=200)
@click.option("--local-epochs", type=int, default=10)
@click.option("--local-batch-size", type=int, default=10)
@click.option("--clients-per-round", type=int, default=5)
@click.option("--l2-norm-clip", type=float, default=1.0)
@click.option("--noise-multiplier", type=float, default=1.0)
@click.option("--num-microbatches", type=int, default=0)
@click.option("--local-dp/--no-local-dp", type=bool, default=False)
@click.option("--mia/--no-mia", type=bool, default=False)
def run(
    devices,
    epochs,
    local_epochs,
    local_batch_size,
    clients_per_round,
    l2_norm_clip,
    noise_multiplier,
    num_microbatches,
    local_dp,
    mia,
):
    # Setup
    privacy_params = (
        LocalDifferentialPrivacyParams(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches or None,
        )
        if l2_norm_clip != 0.0 and noise_multiplier != 0.0 and local_dp
        else None
    )
    hyperparams = Hyperparams(
        batch_size=local_batch_size,
        epochs=local_epochs,
        metrics=["accuracy"],
        optimizer="Adam",
        local_privacy=privacy_params,
    )
    data_configs = list(
        create_mnist_train_data_loader_configs(n_devices=devices, n_shards=200)
    )
    test_config = DatasetLoaderConfig(type="mnist", params=MNISTConfig(split="test"))
    mia_train_set = DatasetLoaderBuilder.from_config(data_configs[0]).load()
    test_set = DatasetLoaderBuilder.from_config(test_config).load()
    model = create_mnist_cnn()
    serialized_model = serialize_model(model)
    weight_bytes = NpzWeightsSerializer().serialize(model.get_weights())
    # weight_string = Base64StringConverter.to_str(weight_bytes)
    global_params = SerializedParameters(
        blob=weight_bytes,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig()
        ),
    )

    round_results = []
    test_accuracy = -1.0
    epoch = 0
    start_time = time.time()
    while test_accuracy < 0.95 and epoch < epochs:
        model_loader = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=global_params,
                model=serialized_model.model_json,
                compiled=True,
                optimizer=serialized_model.optimizer,
                loss=serialized_model.loss,
                metrics=serialized_model.metrics,
            ),
        )

        invocation_params = []
        data_configs_for_round = random.sample(data_configs, clients_per_round)
        for data_config in data_configs_for_round:
            client_invocation_params = ClientInvocationParams(
                data=data_config, hyperparams=hyperparams, model=model_loader
            )
            invocation_params.append(
                (
                    client_invocation_params.data,
                    client_invocation_params.model,
                    client_invocation_params.hyperparams,
                    None,
                    NpzWeightsSerializer(),
                    Base64StringConverter(),
                    False,
                )
            )
        clients_invoked_time = time.time()
        with Pool() as p:
            results = p.starmap(default_handler, invocation_params)
        clients_finished_time = time.time()

        new_parameters, _ = FedAvgAggregator().aggregate(results)
        new_parameters_bytes = NpzWeightsSerializer(compressed=False).serialize(
            new_parameters
        )
        # new_parameters_string = Base64StringConverter.to_str(new_parameters_bytes)
        global_params = SerializedParameters(
            blob=new_parameters_bytes,
            serializer=WeightsSerializerConfig(
                type="npz", params=NpzWeightsSerializerConfig(compressed=False)
            ),
        )

        model.set_weights(new_parameters)
        test_eval = model.evaluate(test_set.batch(32), return_dict=True, verbose=False)

        mia_results = simulate_mia(mia_train_set, test_set, model) if mia else {}
        print(mia_results)

        test_accuracy = test_eval["accuracy"]
        epoch += 1
        print(f"Epoch {epoch}/{epochs}: {test_eval}")
        round_results.append(
            {
                "test_loss": test_eval["loss"],
                "test_accuracy": test_eval["accuracy"],
                "epoch": epoch,
                "devices": devices,
                "epochs": epochs,
                "local_epochs": local_epochs,
                "clients_call_duration": clients_finished_time - clients_invoked_time,
                "clients_per_round": clients_per_round,
                "client_histories": [result.history for result in results],
                "privacy_params": privacy_params.json() if privacy_params else None,
                "privacy_guarantees": [
                    result.privacy_guarantees.json()
                    for result in results
                    if result.privacy_guarantees
                ],
                **mia_results,
            }
        )

        # pd.DataFrame.from_records(round_results).to_csv(
        #     f"results_{devices}_{epochs}_{local_epochs}_{local_batch_size}"
        #     f"_{clients_per_round}_{l2_norm_clip}_{noise_multiplier}_{local_dp}_{num_microbatches}_{start_time}.csv"
        # )


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    run()
