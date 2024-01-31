import asyncio
import logging
import os
import random
import uuid
from itertools import cycle
from pathlib import Path
from typing import List, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import click
import numpy as np

from fedless.common.models import (
    ClientConfig,
    ClientPersistentHistory,
    DataSet,
    FedProxParams,
    MongodbConnectionConfig,
)
from fedless.common.persistence.client_daos import ClientConfigDao, ClientHistoryDao
from fedless.controller.misc import parse_yaml_file
from fedless.controller.models import ClientFunctionConfigList, ExperimentConfig
from fedless.controller.strategies.strategy_selector import select_strategy
from fedless.datasets.benchmark_configurator import (
    create_data_configs,
    create_mnist_test_config,
)
from fedless.datasets.init_models import (
    create_model,
    init_store_model,
    init_store_model_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(
        [
            "mnist",
            "femnist",
            "shakespeare",
            "speech",
            "fedmd_mnist",
            "fedmd_cifar",
            "fedmd_shakespeare",
            "feddf_cifar",
            "feddf_shakespeare",
            "feddf_mnist",
        ],
        case_sensitive=False,
    ),
    required=True,
)
@click.option(
    "-c",
    "--config",
    help="Config file with faas platform and client function information",
    type=click.Path(),
    required=True,
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["fedavg", "fedlesscan", "fedprox", "fedmd", "feddf"], case_sensitive=False),
    required=True,
)
# @click.option(
#     "--clients",
#     type=int,
#     help="number of clients",
#     required=True,
# )
@click.option(
    "--clients-in-round",
    type=int,
    help="number of clients sampled per round",
    required=True,
)
@click.option(
    "--stragglers",
    type=int,
    help="number of allowed stragglers per round",
    default=0,
)
@click.option(
    "--timeout",
    type=float,
    help="maximum wait time for functions to finish",
    default=300,
)
@click.option(
    "--rounds",
    type=int,
    help="Total number of training rounds",
    default=100,
)
@click.option(
    "--max-accuracy",
    help="stop training if this test accuracy is reached",
    type=float,
    default=0.99,
)
@click.option(
    "-o",
    "--out",
    help="directory where logs will be stored",
    type=click.Path(),
    required=True,
)
@click.option(
    "--tum-proxy/--no-tum-proxy",
    help="use in.tum.de proxy",
    default=False,
)
@click.option(
    "--proxy-in-evaluator/--no-proxy-in-evaluator",
    help="use proxy also in evaluation function",
    default=False,
)
@click.option(
    "--mock/--no-mock",
    help="use mocks",
    default=False,
)

# straggler simulation per client if specified in function or with percentages
@click.option(
    "--simulate-stragglers",
    help="define a percentage of the clients to straggle, this option overrides the invocation delay if specified in the function",
    type=float,
    default=0.0,
)
@click.option(
    "--mu",
    help="param for fedprox training",
    type=float,
    default=0.001,
)
@click.option(
    "--private-data-dirichlet-alpha",
    help="param for controlling data heterogeneity",
    type=float,
    default=100,
)
def run(
    dataset: str,
    config: str,
    strategy: str,
    # clients: int,
    clients_in_round: int,
    stragglers: int,
    timeout: float,
    rounds: int,
    max_accuracy: float,
    out: str,
    tum_proxy: bool,
    proxy_in_evaluator: bool,
    mock: bool,
    simulate_stragglers: float,
    mu: float,
    private_data_dirichlet_alpha: float,
):
    session = str(uuid.uuid4())
    log_dir = Path(out) if out else Path(config).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    config: ExperimentConfig = parse_yaml_file(config, model=ExperimentConfig)
    with (log_dir / f"config_{session}.json").open("w+") as f:
        f.write(config.json())

    # Configure proxy if specified
    proxies = (
        {
            "https": "http://proxy.in.tum.de:8080/",
            "http": "http://proxy.in.tum.de:8080/",
            "https://138.246.233.81": "",
            "http://138.246.233.81": "",
            "http://138.246.233.217": "",
            "https://138.246.233.217": "",
            "https://127.0.0.1": "",
            "http://127.0.0.1": "",
            "https://localhost": "",
            "http://localhost": "",
        }
        if tum_proxy
        else None
    )

    # Creating heterogenous models for each client in case of fedmd
    n_clients = len(config.clients.functions)
    model = create_model(dataset, config)
    data_configs, max_test_clients_count = create_data_configs(
        dataset, n_clients, rounds, config, private_data_dirichlet_alpha, proxies=proxies
    )

    clients = store_client_configs(
        session=session,
        strategy=strategy,
        clients=config.clients,
        num_clients=n_clients,
        data_configs=data_configs,
        database=config.database,
        stragglers_precentage=simulate_stragglers,
        fedprox_global_mu=mu,
    )

    if "fedmd" in dataset.lower() or "feddf" in dataset.lower():
        assert len(model["local_models"]) == len(
            clients
        ), "Number of specified models specified is not equal to number of clients"
        init_store_model_client(
            session=session,
            models=model["local_models"],
            database=config.database,
            clients=clients,
            store_json_serializable=False,
        )

    else:
        # Only initializing global server model
        init_store_model(
            session=session,
            model=model["global_model"],
            database=config.database,
            store_json_serializable=False,
        )

    inv_params = {
        "session": session,
        "cognito": config.cognito,
        "clients": clients,
        "evaluator_config": config.evaluator,
        "aggregator_config": config.aggregator,
        "mongodb_config": config.database,
        "allowed_stragglers": stragglers,
        "client_timeout": timeout,
        "save_dir": log_dir,
        "global_test_data": (
            create_mnist_test_config(proxies=(proxies if proxy_in_evaluator else None))
            if dataset.lower() == "mnist"
            else None
        ),
        "proxies": proxies,
        "mock": mock,
        "max_test_client_count": max_test_clients_count,
    }

    strategy = select_strategy(strategy, inv_params)

    asyncio.run(
        strategy.fit(
            n_clients_in_round=clients_in_round,
            max_rounds=rounds,
            max_accuracy=max_accuracy,
        )
    )


def store_client_configs(
    session: str,
    strategy: str,
    clients: ClientFunctionConfigList,
    num_clients: int,
    data_configs: List[DataSet],
    database: MongodbConnectionConfig,
    stragglers_precentage: float,
    fedprox_global_mu: float,
) -> List[ClientConfig]:
    client_config_dao = ClientConfigDao(database)
    client_history_dao = ClientHistoryDao(database)
    n_clients = sum(function.replicas for function in clients.functions)
    clients_unrolled = list(f for f in clients.functions for _ in range(f.replicas))
    logger.info(
        f"{len(data_configs)} data configurations given with the "
        f"instruction to setup {num_clients} clients from {n_clients} potential endpoints."
    )
    # todo add delay param for all clients
    # stragglers_delay_list = [-1, -2]
    stragglers_delay_list = [-1]
    # stragglers_delay_list = [-1]

    num_stragglers = int(stragglers_precentage * num_clients)
    logger.info(f"simulate stragglers {num_stragglers} clients for {num_clients}.")
    data_shards = iter(data_configs)
    function_iter = cycle(clients_unrolled)
    if strategy == "fedprox" and fedprox_global_mu > 0:
        st = FedProxParams()
        st.mu = fedprox_global_mu
        clients.hyperparams.fedprox = st

    default_hyperparms = clients.hyperparams
    final_configs = []
    stragglers_idx_list = random.sample(list(np.arange(num_clients)), num_stragglers)
    for idx, shard in enumerate(data_shards):
        client = next(function_iter)
        #  addition fedprox on specific functions or custom hp for specific functions
        hp = client.hyperparams or default_hyperparms
        client_id = str(uuid.uuid4())

        client_config = ClientConfig(
            session_id=session,
            client_id=client_id,
            function=client.function,
            data=DataSet(**shard),
            hyperparams=hp,
        )

        # add straggler
        if idx in stragglers_idx_list:
            client_config.function.invocation_delay = random.sample(stragglers_delay_list, 1)[0]
            # client_config.function.invocation_delay = -2

        client_history = ClientPersistentHistory(
            client_id=client_id,
            session_id=session,
        )

        logger.info(f"Initializing client {client_id} of type " f"{client.function.type}")
        client_config_dao.save(client_config)
        logger.info(f"Initializing client_history for {client_id} of type " f"{client.function.type}")
        client_history_dao.save(client_history)
        final_configs.append(client_config)
    logger.info(f"Configured and stored all {len(data_configs)} clients configurations...")
    return final_configs


if __name__ == "__main__":
    run()
