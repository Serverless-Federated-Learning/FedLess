import logging
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import pymongo
import ray
import ray.tune as tune
from keras.callbacks import EarlyStopping
from ray.tune import Tuner
from ray.tune.integration.keras import TuneReportCallback
from tensorflow import keras as keras

from fedless.common.models import (
    BinaryStringFormat,
    ModelLoaderConfig,
    SerializedParameters,
    SimpleModelLoaderConfig,
)
from fedless.common.persistence.client_daos import (
    ClientConfigDao,
    ClientModelDao,
    ClientParameterDao,
)
from fedless.common.serialization import (
    ModelLoaderBuilder,
    NpzWeightsSerializer,
    WeightsSerializer,
)
from fedless.datasets.dataset_loader_builder import DatasetLoaderBuilder

logger = logging.getLogger(__name__)


def train_func(config):
    client_id = config["client_id"]
    session_id = config["session_id"]
    round_id = config["round_id"]
    db = config["db"]
    validation_split = config["validation_split"]
    use_existing_model = config["use_existing_model"]
    existing_session_id = config["existing_session_id"]
    existing_client_mapping = config["existing_client_mapping"]
    patience = config["patience"]
    min_delta = config["min_delta"]

    client_parameter_dao = ClientParameterDao(db=db)
    model_dao = ClientModelDao(db=db)
    config_dao = ClientConfigDao(db=db)

    client_config = config_dao.load(client_id=client_id)

    model = model_dao.load(session_id=session_id, client_id=client_id)
    hyperparams = client_config.hyperparams.fedmd.pre_train_hyperparams

    if use_existing_model and (client_id in existing_client_mapping):
        existing_client_id = existing_client_mapping[client_id]
        existing_client_parameters = client_parameter_dao.load(
            session_id=existing_session_id, client_id=existing_client_id, round_id=-3
        )

        client_parameter_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            params=existing_client_parameters,
        )

        model_config = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=existing_client_parameters,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )

        model_loader = ModelLoaderBuilder.from_config(model_config)
        model = model_loader.load()

    else:

        latest_client_parameters = client_parameter_dao.load_latest(session_id=session_id, client_id=client_id)
        model_config = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=latest_client_parameters,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )

        model_loader = ModelLoaderBuilder.from_config(model_config)
        model = model_loader.load()

        data_loader = DatasetLoaderBuilder.from_config(client_config.data.public_train_data)

        dataset = data_loader.load()

        weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=client_config.compress_model)

        if validation_split:
            dataset = dataset.shuffle(len(dataset))
            cardinality = float(dataset.cardinality())
            train_validation_split_idx = int(cardinality - cardinality * validation_split)
            train_dataset = dataset.take(train_validation_split_idx)
            val_dataset = dataset.skip(train_validation_split_idx)
            train_dataset = train_dataset.batch(hyperparams.batch_size)
            val_dataset = val_dataset.batch(hyperparams.batch_size)
            early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=patience, min_delta=min_delta)
        else:
            train_dataset = dataset.batch(hyperparams.batch_size)
            val_dataset = None
            early_stopping = None

        model.fit(
            train_dataset,
            epochs=hyperparams.epochs,
            shuffle=hyperparams.is_shuffle,
            validation_data=val_dataset,
            verbose=hyperparams.verbose,
            callbacks=[TuneReportCallback({"accuracy": "accuracy"}), early_stopping],
        )

        # serialization error
        logger.debug(f"Serializing model parameters")
        weights_serialized = weights_serializer.serialize(model.get_weights())
        parameters = SerializedParameters(
            blob=weights_serialized,
            serializer=weights_serializer.get_config(),
            string_format=(BinaryStringFormat.NONE),
        )
        logger.info(f"Finished serializing model parameters of size {sys.getsizeof(weights_serialized)} bytes")

        client_parameter_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            params=parameters,
        )

    # Calculating initial public test accuracy
    # public_test_data_loader = DatasetLoaderBuilder.from_config(client_config.data.public_test_data)
    # test_data = public_test_data_loader.load()
    # test_data = test_data.batch(client_config.hyperparams.batch_size)
    # evaluation_result = model.evaluate(test_data, return_dict=True, verbose=hyperparams.verbose)

    # return {
    #     "test_accuracy": evaluation_result["accuracy"],
    #     "test_loss": evaluation_result["loss"],
    #     "client_id": client_id,
    # }

    return {
        "test_accuracy": -999,
        "test_loss": -999,
        "client_id": "abc",
    }


class FedMDRayTrainer:
    def __init__(
        self,
        validation_split,
        min_delta,
        patience,
        session_id,
        round_id,
        clients,
        db: pymongo.MongoClient,
        cluster_svc_addr=None,
        train_local=False,
        use_existing_model=False,
        existing_session_id=None,
    ):
        self.train_func = train_func
        self.validation_split = validation_split
        self.min_delta = min_delta
        self.patience = patience
        self.db = db
        self.session_id = session_id
        self.clients = clients
        self.cluster_svc_addr = cluster_svc_addr
        self.round_id = round_id
        self.train_local = train_local
        self.use_existing_model = use_existing_model
        self.existing_session_id = existing_session_id
        self.model_dao = ClientModelDao(db=self.db)
        self.config_dao = ClientConfigDao(db=self.db)

    def get_client_mapping(self, current_clients, existing_model_session):

        client_mapping = {}

        existing_model_client_mapping = self.model_dao.get_model_client_mapping(existing_model_session)

        for client in current_clients:
            current_client_model_config = self.model_dao.load(session_id=self.session_id, client_id=client)
            if current_client_model_config.model_type in existing_model_client_mapping.keys():
                client_mapping[client] = existing_model_client_mapping[current_client_model_config.model_type][0]
            else:
                client_mapping[client] = None

        return client_mapping

    def fit_parallel(self):

        client_ids = [client.client_id for client in self.clients]

        if self.use_existing_model:
            client_mapping = self.get_client_mapping(client_ids, self.existing_session_id)

            results = pd.DataFrame()
            for client_id in client_ids:
                parameters = {
                    "client_id": client_id,
                    "db": self.db,
                    "session_id": self.session_id,
                    "round_id": self.round_id,
                    "validation_split": self.validation_split,
                    "use_existing_model": self.use_existing_model,
                    "existing_session_id": self.existing_session_id,
                    "min_delta": self.min_delta,
                    "patience": self.patience,
                    "existing_client_mapping": client_mapping,
                }

                client_result = self.train_func(parameters)
                results = results.append(
                    {
                        "client_id": client_id,
                        "test_accuracy": client_result["test_accuracy"],
                        "test_loss": client_result["test_loss"],
                        "total_time_ray": f"Used pretrained from - {self.existing_session_id}",
                    },
                    ignore_index=True,
                )

            return results

        model_client_mapping = self.model_dao.get_model_client_mapping(self.session_id)

        # TODO Improve this logic
        unique_train_client_ids = []
        remaining_client_ids_mapping = {}
        for _, value in model_client_mapping.items():
            unique_train_client_ids.append(value[0])
            for remaining_client in value[1:]:
                remaining_client_ids_mapping[remaining_client] = value[0]

        # parameters = {
        #             "client_id": unique_train_client_ids[0],
        #             "db": self.db,
        #             "session_id": self.session_id,
        #             "round_id": self.round_id,
        #             "validation_split": self.validation_split,
        #             "use_existing_model": self.use_existing_model,
        #             "existing_session_id": self.existing_session_id,
        #             "existing_client_mapping": None,
        #         }

        # self.train_func(parameters)
        parameters = {
            "client_id": tune.grid_search(unique_train_client_ids),
            "db": self.db,
            "session_id": self.session_id,
            "round_id": self.round_id,
            "validation_split": self.validation_split,
            "use_existing_model": self.use_existing_model,
            "existing_session_id": self.existing_session_id,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "existing_client_mapping": None,
        }

        if self.train_local:
            ray.init()
        else:
            runtime_env = {"pip": ["http://10.103.195.168:80/data/fedless-0.0.0-py3-none-any.whl"]}
            ray.init(f"ray://{self.cluster_svc_addr}", runtime_env=runtime_env)

        tuner = Tuner(
            tune.with_resources(self.train_func, resources={"cpu": 4, "gpu": 0}),
            param_space=parameters,
        )

        ray_results = tuner.fit()
        ray.shutdown()

        assert (
            ray_results.num_errors == 0
        ), f"{ray_results.num_errors} errors occured during initital public training on the central server"

        ray_results_df = pd.DataFrame()
        for ray_result in ray_results:
            ray_results_df = ray_results_df.append(
                {
                    "client_id": ray_result.metrics["client_id"],
                    "test_accuracy": ray_result.metrics["test_accuracy"],
                    "test_loss": ray_result.metrics["test_loss"],
                    "total_time_ray": ray_result.metrics["time_total_s"],
                },
                ignore_index=True,
            )

        # # # #Distributing parameters to other remaining clients in same model prototype groups
        remaining_client_results = pd.DataFrame()
        for client_id in remaining_client_ids_mapping.keys():
            parameters = {
                "client_id": client_id,
                "db": self.db,
                "session_id": self.session_id,
                "round_id": self.round_id,
                "validation_split": self.validation_split,
                "use_existing_model": True,
                "existing_session_id": self.session_id,
                "min_delta": self.min_delta,
                "patience": self.patience,
                "existing_client_mapping": remaining_client_ids_mapping,
            }

            client_result = train_func(parameters)
            remaining_client_results = remaining_client_results.append(
                {
                    "client_id": client_id,
                    "test_accuracy": client_result["test_accuracy"],
                    "test_loss": client_result["test_loss"],
                },
                ignore_index=True,
            )

        return pd.concat([ray_results_df, remaining_client_results])
