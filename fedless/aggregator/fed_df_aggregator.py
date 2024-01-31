import logging

import tensorflow as tf
from tensorflow import keras

from fedless.aggregator.exceptions import InsufficientClientResults
from fedless.aggregator.fed_avg_aggregator import FedAvgAggregator
from fedless.common.models import (
    ModelLoaderConfig,
    SerializedParameters,
    SimpleModelLoaderConfig,
)
from fedless.common.persistence.client_daos import (
    ClientConfigDao,
    ClientModelDao,
    ClientParameterDao,
    ClientResultDao,
)
from fedless.common.serialization import (
    ModelLoaderBuilder,
    NpzWeightsSerializer,
    WeightsSerializer,
    deserialize_parameters,
)
from fedless.datasets.dataset_loader_builder import DatasetLoaderBuilder
from fedless.datasets.models import remove_last_layer

logger = logging.getLogger(__name__)


class FedDFAggregator:
    def __init__(self, mongo_client, session_id, round_id, model_type, aggregation_hyper_params):
        self.mongo_client = mongo_client
        self.session_id = session_id
        self.round_id = round_id
        self.model_type = model_type
        self.aggregation_hyper_params = aggregation_hyper_params

        self.client_model_dao = ClientModelDao(db=mongo_client)
        self.client_parameter_dao = ClientParameterDao(db=mongo_client)
        self.client_result_dao = ClientResultDao(db=mongo_client)
        self.config_dao = ClientConfigDao(db=mongo_client)

        # Getting clients for this given model type
        self.clients_model = self.client_model_dao.get_model_client_mapping(session_id=session_id)[model_type]
        self.clients_in_round = self.client_result_dao.load_client_id_in_round(session_id=session_id, round_id=round_id)
        self.distillation_clients = list(set(self.clients_model).intersection(self.clients_in_round))

        self.clients_configs = list(
            self.config_dao.load_for_clients(session_id=session_id, client_ids=self.clients_in_round)
        )

    def perform_round_db_cleanup(self, mongo_client, session_id, round_id):
        pass

    def get_initial_master_model(self):

        # Pick any model from the prototype client models and initialize that with fedavg results
        serialized_model = self.client_model_dao.load(
            session_id=self.session_id, client_id=self.distillation_clients[0]
        )
        model: tf.keras.Model = tf.keras.models.model_from_json(serialized_model.model_json)

        logger.info("Performing prototype level federated average initialization")
        fedavg_parameters = self.model_type_fedavg()

        model.set_weights(fedavg_parameters)
        logit_model = remove_last_layer(model)
        logit_model.compile(
            optimizer=tf.keras.optimizers.get(serialized_model.optimizer),
            loss=self.kl_divergence_loss,
            metrics=[],
        )

        return model, logit_model

    def model_type_fedavg(self):

        # Intitialize for model fusion by performing FedAvg on all the client architectures and initialing model with this
        _, client_results = self.client_result_dao.load_results_for_client_round(
            session_id=self.session_id, round_id=self.round_id, client_ids=self.distillation_clients
        )
        if not client_results:
            raise InsufficientClientResults(
                f"Found no client results for session {self.session_id} and round {self.round_id}"
            )

        fedavg_aggregator = FedAvgAggregator()
        fedavg_parameters, _ = fedavg_aggregator.aggregate(client_results=client_results)

        # Distribute serialized server model parameters/weights back to each of the clients
        # weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=self.clients_configs[0].compress_model)
        # weights_serialized = weights_serializer.serialize(fedavg_parameters)
        # weights_serialized = weights_serializer.serialize(server_model.get_weights())
        # serialized_params = SerializedParameters(blob=weights_serialized, serializer=weights_serializer.get_config())

        # for client in self.distillation_clients:
        #     self.client_parameter_dao.save(
        #         session_id=self.session_id,
        #         round_id=self.round_id,
        #         client_id=client,
        #         params=serialized_params,
        #     )

        # return len(self.distillation_clients)
        return fedavg_parameters

    @staticmethod
    def kl_divergence_loss(student_logits, teacher):

        kld = keras.losses.KLDivergence()
        teacher_activation = tf.nn.softmax(teacher, axis=1)
        student_activation = tf.nn.softmax(student_logits, axis=1)

        return kld(
            teacher_activation,
            student_activation,
        )

    def get_model_performance(self, model, dataset):

        eval_model = tf.keras.models.clone_model(model)
        eval_model.set_weights(model.get_weights())
        eval_model.compile(metrics=["accuracy"])
        return eval_model.evaluate(
            dataset.batch(self.aggregation_hyper_params.feddf_hyperparams.pseudo_batch_size),
            return_dict=True,
            verbose=False,
        )

    def ensemble_distillation(self):

        server_model, server_logit_model = self.get_initial_master_model()

        client_models = []

        # Getting current client logit models
        for client_config in self.clients_configs:

            # Creating model for this client
            model = self.client_model_dao.load(session_id=self.session_id, client_id=client_config.client_id)

            latest_weights = self.client_result_dao.load(
                session_id=self.session_id, client_id=client_config.client_id, round_id=self.round_id
            ).parameters

            model = ModelLoaderConfig(
                type="simple",
                params=SimpleModelLoaderConfig(
                    params=latest_weights,
                    model=model.model_json,
                    compiled=True,
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics,
                ),
            )

            logit_model = remove_last_layer(ModelLoaderBuilder.from_config(model).load())
            client_models.append(logit_model)

        # Creating public alignment dataloader
        distillation_data_loader = DatasetLoaderBuilder.from_config(
            self.clients_configs[0].data.public_alignment_data[self.round_id]
        )
        dataset = distillation_data_loader.load()
        data = dataset.batch(self.aggregation_hyper_params.feddf_hyperparams.pseudo_batch_size)
        distill_iterator = iter(data)

        # Creating validation data loader
        validation_data_loader = DatasetLoaderBuilder.from_config(self.clients_configs[0].data.val_data)
        val_dataset = validation_data_loader.load()

        n_pseudo_batches = 0

        # Early stopping parameters
        patience_count = 0
        best_acc = 0

        while n_pseudo_batches < self.aggregation_hyper_params.feddf_hyperparams.n_pseudo_batches:
            # We keep looping over the distillation dataset until loop terminates
            try:
                pseudo_data = distill_iterator.get_next()[0]
            except tf.errors.OutOfRangeError:
                distill_iterator = iter(data)
                pseudo_data = distill_iterator.get_next()[0]

            # Get all student/client predictions on distillation unlabelled data batch
            client_teacher_logits = [teacher.predict(pseudo_data, verbose=False) for teacher in client_models]

            weights = [1.0 / len(client_teacher_logits)] * len(client_teacher_logits)

            client_teacher_avg_logits = sum(
                [teacher_logit * weight for teacher_logit, weight in zip(client_teacher_logits, weights)]
            )

            logit_alignment_batch = tf.data.Dataset.from_tensor_slices((pseudo_data, client_teacher_avg_logits)).batch(
                len(pseudo_data)
            )

            server_logit_model.fit(logit_alignment_batch, verbose=True)

            if (n_pseudo_batches + 1) % self.aggregation_hyper_params.feddf_hyperparams.eval_batches_frequency == 0:
                server_model.set_weights(server_logit_model.get_weights())

                validated_perf = self.get_model_performance(model=server_model, dataset=val_dataset)
                logger.info(
                    f"Batch {n_pseudo_batches + 1}/{self.aggregation_hyper_params.feddf_hyperparams.n_pseudo_batches}:  Student Validation Metrics={validated_perf}."
                )

                # Early Stopping
                current_accuracy = validated_perf["accuracy"]

                if current_accuracy > best_acc:
                    best_server_weights = server_model.get_weights()
                    best_acc = current_accuracy
                    patience_count = 0

                else:
                    patience_count += 1
                    logger.info(f"Current patience count: {patience_count}")

                    if patience_count >= self.aggregation_hyper_params.feddf_hyperparams.patience:
                        logger.info("Early stopping the distillation process!")
                        break

            n_pseudo_batches += 1

        # Getting final model performance on test data after distillation process
        test_data_loader = DatasetLoaderBuilder.from_config(self.clients_configs[0].data.test_data)
        test_dataset = test_data_loader.load()
        test_cardinality = test_dataset.cardinality()

        server_model.set_weights(best_server_weights)
        final_test_perf = self.get_model_performance(model=server_model, dataset=test_dataset)
        # final_test_perf = self.get_model_performance(model=server_model, dataset=test_dataset)

        # Distribute serialized server model parameters/weights back to each of the clients
        weights_serializer: WeightsSerializer = NpzWeightsSerializer(compressed=self.clients_configs[0].compress_model)
        weights_serialized = weights_serializer.serialize(server_model.get_weights())
        # weights_serialized = weights_serializer.serialize(server_model.get_weights())
        serialized_params = SerializedParameters(blob=weights_serialized, serializer=weights_serializer.get_config())

        for client in self.clients_model:
            self.client_parameter_dao.save(
                session_id=self.session_id,
                round_id=self.round_id,
                client_id=client,
                params=serialized_params,
            )

        return len(self.clients_model), final_test_perf, test_cardinality
