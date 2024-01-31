import random

from fedless.common.models import DatasetLoaderConfig
from fedless.datasets.char_prediction.dataset_loader import CharacterPredictionConfig
from fedless.datasets.dataset_loader_builder import DatasetLoaderBuilder
from fedless.datasets.leaf.dataset_loader import LEAFConfig


def get_feddf_char_prediction_data_configs(clients, n_rounds, feddf_params, file_server, proxies=None):

    configs = []

    for client_idx in range(clients):
        train_private = DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset="shakespeare",
                location=f"{file_server}/data/leaf-non-iid/leaf/data/shakespeare/data/"
                f"train/user_{client_idx}_train_9.json",
            ),
        )

        test_private = DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset="shakespeare",
                location=f"{file_server}/data/leaf-non-iid/leaf/data/shakespeare/data/"
                f"test/user_{client_idx}_test_9.json",
            ),
        )

        # train_public = DatasetLoaderConfig(
        #     type="char_prediction",
        #     params=CharacterPredictionConfig(
        #         dataset_name="nietzsche",
        #         location="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
        #         sequence_length=80,
        #         steps=3,
        #         split="train",
        #         max_sequences=10000,
        #     ),
        # )

        configs.append({"train_data": train_private, "test_data": test_private, "val_data": test_private})

    alignment_configs_train = list(
        create_public_char_prediction_data_loader_configs_random(
            character_prediction_conf=CharacterPredictionConfig(
                dataset_name="nietzsche",
                location="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
                sequence_length=80,
                steps=3,
                split="train",
                max_sequences=10000,
            ),
            cardinality=feddf_params.n_distillation_data,
            n_rounds=n_rounds,
        )
    )

    for config in configs:
        config["public_alignment_data"] = alignment_configs_train

    return configs


def get_fedmd_char_prediction_data_configs(clients, n_rounds, fedmd_params, file_server, proxies=None):

    configs = []
    for client_idx in range(clients):
        train_private = DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset="shakespeare",
                # location=f"{file_server}/data/leaf/data/shakespeare/data/" f"train/user_{client_idx}_train_9.json",
                location=f"{file_server}/data/leaf-non-iid/leaf/data/shakespeare/data/"
                f"train/user_{client_idx}_train_9.json",
            ),
        )
        test_private = DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset="shakespeare",
                # location=f"{file_server}/data/leaf/data/shakespeare/data/" f"test/user_{client_idx}_test_9.json",
                location=f"{file_server}/data/leaf-non-iid/leaf/data/shakespeare/data/"
                f"test/user_{client_idx}_test_9.json",
            ),
        )

        train_public = DatasetLoaderConfig(
            type="char_prediction",
            params=CharacterPredictionConfig(
                dataset_name="nietzsche",
                location="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
                sequence_length=80,
                steps=3,
                split="train",
                max_sequences=10000,
            ),
        )

        test_public = DatasetLoaderConfig(
            type="char_prediction",
            params=CharacterPredictionConfig(
                dataset_name="nietzsche",
                location="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
                sequence_length=80,
                steps=3,
                split="test",
                max_sequences=10000,
            ),
        )

        configs.append(
            {
                "train_data": train_private,
                "test_data": test_private,
                "public_train_data": train_public,
                "public_test_data": test_public,
            }
        )

    alignment_configs_train = list(
        create_public_char_prediction_data_loader_configs_random(
            character_prediction_conf=CharacterPredictionConfig(
                dataset_name="nietzsche",
                location="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
                sequence_length=80,
                steps=3,
                split="train",
                max_sequences=10000,
            ),
            cardinality=fedmd_params.logit_alignment_hyperparams.n_alignment,
            n_rounds=n_rounds,
        )
    )

    for config in configs:
        config["public_alignment_data"] = alignment_configs_train

    return configs


def create_public_char_prediction_data_loader_configs_random(
    character_prediction_conf: CharacterPredictionConfig,
    cardinality: int,
    n_rounds: int,
):

    dataset_loader_config = DatasetLoaderConfig(type="char_prediction", params=character_prediction_conf)
    dataset_loader = DatasetLoaderBuilder.from_config(dataset_loader_config)
    total_indices = dataset_loader.get_total_data_indices()

    for _ in range(n_rounds):
        character_prediction_conf.user_indices = random.sample(list(total_indices), cardinality)

        yield DatasetLoaderConfig(
            type="char_prediction",
            params=character_prediction_conf,
        )
