import click
import flwr as fl
import tensorflow as tf
from fedless.datasets.benchmark_configurator import DatasetLoaderBuilder
from fedless.datasets.leaf import LeafDataset
from fedless.datasets.leaf.dataset_loader import LEAF
from fedless.datasets.leaf.model import create_femnist_cnn, create_shakespeare_lstm
from fedless.datasets.mnist.helpers import create_mnist_train_data_loader_configs
from fedless.datasets.mnist.model import create_mnist_cnn


FILE_SERVER = "http://138.246.235.175:81"


class FedlessClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        epochs: int,
        batch_size: int,
    ):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.dataset.batch(self.batch_size), epochs=self.epochs)
        return self.model.get_weights(), int(self.dataset.cardinality()), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset.batch(self.batch_size))
        print(
            f"Evaluating: {config}: {(loss, int(self.test_dataset.cardinality()), {'accuracy': accuracy})}"
        )
        return (
            loss,
            int(self.test_dataset.cardinality()),
            {"accuracy": accuracy},
        )


# noinspection PydanticTypeChecker,PyTypeChecker
@click.command()
@click.option("--server", type=str, required=True)
@click.option("--dataset", type=str, required=True)
@click.option("--partition", type=int, required=True)
@click.option("--batch-size", type=int, default=16)
@click.option("--epochs", type=int, default=1)
@click.option("--optimizer", type=str, default="Adam")
@click.option("--lr", type=float, default=0.001)
@click.option("--clients-total", type=int, default=200)
def run(
    server: str,
    dataset: str,
    partition: int,
    batch_size: int,
    epochs: int,
    optimizer: str,
    lr: float,
    clients_total: int,
) -> None:
    if dataset.lower() == "femnist":
        model = create_femnist_cnn()
        train_set = LEAF(
            dataset=LeafDataset.FEMNIST,
            location=f"{FILE_SERVER}/data/leaf/data/femnist/data/"
            f"train/user_{partition}_train_9.json",
        ).load()
        test_set = LEAF(
            dataset=LeafDataset.FEMNIST,
            location=f"{FILE_SERVER}/data/leaf/data/femnist/data/"
            f"test/user_{partition}_test_9.json",
        ).load()
    elif dataset.lower() == "shakespeare":
        model = create_shakespeare_lstm()
        train_set = LEAF(
            dataset=LeafDataset.SHAKESPEARE,
            location=f"{FILE_SERVER}/data/leaf/data/shakespeare/data/"
            f"train/user_{partition}_train_9.json",
        ).load()
        test_set = LEAF(
            dataset=LeafDataset.SHAKESPEARE,
            location=f"{FILE_SERVER}/data/leaf/data/shakespeare/data/"
            f"test/user_{partition}_test_9.json",
        ).load()
    elif dataset.lower() == "mnist":
        model = create_mnist_cnn()
        assert (
            partition < clients_total
        ), f"Partition index {partition} too large. Must be < {clients_total}"
        partition_config = list(
            create_mnist_train_data_loader_configs(
                n_devices=clients_total,
                n_shards=600,
            )
        )[partition]
        train_set = DatasetLoaderBuilder.from_config(partition_config).load()
        test_set = None
    else:
        raise NotImplementedError()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.optimizer.learning_rate.assign(lr)

    client = FedlessClient(
        model=model,
        dataset=train_set,
        test_dataset=test_set,
        epochs=epochs,
        batch_size=batch_size,
    )

    fl.client.start_numpy_client(server, client=client)


if __name__ == "__main__":
    run()
