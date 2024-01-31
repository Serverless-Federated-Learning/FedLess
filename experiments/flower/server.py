import concurrent.futures
from typing import Optional, Tuple, List, Dict

import click
import flwr as fl
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from fedless.datasets.mnist.dataset_loader import MNIST
from fedless.datasets.mnist.model import create_mnist_cnn


# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


def fit_clients(client_instructions):
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        futures = [
            executor.submit(fit_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def fit_client(client: ClientProxy, ins):
    """Refine parameters on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res


def evaluate_clients(client_instructions):
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def evaluate_client(client: ClientProxy, ins) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    test_set = MNIST(split="test").load().batch(10)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, Dict]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_set)
        return loss, {"accuracy": accuracy}

    return evaluate


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        print(rnd, results, failures)
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        accuracy_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.metrics.get("accuracy", 0.0),
                    evaluate_res.loss,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {"accuracy": accuracy_aggregated}


@click.command()
@click.option("--dataset", type=str)
@click.option("--min-num-clients", type=int, default=5)
@click.option("--rounds", type=int, default=200)
@click.option("--port", type=int, default=31532)
def run(dataset, min_num_clients, rounds, port):
    client_manager = fl.server.SimpleClientManager()
    strategy = AggregateCustomMetricStrategy(
        min_fit_clients=min_num_clients,
        min_eval_clients=min_num_clients,
        min_available_clients=min_num_clients,
        eval_fn=get_eval_fn(create_mnist_cnn()) if dataset.lower() == "mnist" else None,
    )
    # if dataset.lower() != "mnist":
    fl.server.server.evaluate_clients = evaluate_clients
    fl.server.server.fit_clients = fit_clients
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        f"[::]:{port}",
        server,
        config={"num_rounds": rounds},
    )


if __name__ == "__main__":
    run()
