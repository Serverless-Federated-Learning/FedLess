import logging

# ignore DeprecationWarning
import warnings

# suppress warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)
# suppress WARNING:asyncio:Executing
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

import asyncio
import json
import random
import time
import uuid
from itertools import cycle
from pathlib import Path
from typing import List, Union, Tuple

import click
import numpy as np
from fedless.common.models.aggregation_models import AggregationStrategy
from fedless.common.models.models import ScaffoldParams

from fedless.controller.misc import parse_yaml_file
from fedless.controller.models import (
    ExperimentConfig,
    ClientFunctionConfigList,
)
from fedless.datasets.benchmark_configurator import (
    create_model,
    init_store_model,
    create_mnist_test_config,
    create_data_configs,
)

from fedless.controller.strategies.strategy_selector import select_strategy
from fedless.common.models import (
    ClientConfig,
    ClientPersistentHistory,
    MongodbConnectionConfig,
    DatasetLoaderConfig,
    FedProxParams,
    FedNovaParams,
)
from fedless.common.persistence.client_daos import (
    ClientConfigDao,
    ClientControlDao,
    ClientHistoryDao,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(
        ["mnist", "femnist", "shakespeare", "speech"], case_sensitive=False
    ),
    required=True,
    # help='Evaluation dataset. One of ("mnist", "femnist", "shakespeare")',
)
@click.option(
    "-c",
    "--config",
    help="Config file with faas platform and client function information",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(
        [
            "fedavg",
            "fedlesscan",
            "fedprox",
            "fednova",
            "scaffold",
            "fedasync",
            "fedscore",
        ],
        case_sensitive=False,
    ),
    required=True,
)
@click.option(
    "--clients",
    type=int,
    help="number of clients",
    required=True,
)
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
    help="maximum wait time for functions to finish",
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
    help="use mocks for both client/aggregator",
    default=False,
)
@click.option(
    "--mock-aggregator/--no-mock-aggregator",
    help="use mocks for only for aggregator",
    default=False,
)
@click.option(
    "--mock-cold-start/--no-mock-cold-start",
    help="perform the aggregation synchronously/asynchronously",
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
    "--buffer-ratio",
    help="Buffer ratio to aggregation",
    type=float,
)
def run(
    dataset: str,
    config: str,
    strategy: str,
    clients: int,
    clients_in_round: int,
    stragglers: int,
    timeout: float,
    rounds: int,
    max_accuracy: float,
    out: str,
    tum_proxy: bool,
    proxy_in_evaluator: bool,
    mock: bool,
    mock_aggregator: bool,
    mock_cold_start: bool,
    simulate_stragglers: float,
    buffer_ratio: float,
):

    session = str(uuid.uuid4())
    log_dir = Path(out) if out else Path(config).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    config: ExperimentConfig = parse_yaml_file(config, model=ExperimentConfig)

    # if more clients provided in config, select top N clients
    if len(config.clients.functions) > clients:
        config.clients.functions = config.clients.functions[:clients]

    # raise error if client count doesn't match
    if not mock and len(config.clients.functions) != clients:
        raise ValueError(
            f"Number of clients ({clients}) doesn't match the number of client functions ({len(config.clients.functions)}) provided in the config file."
        )

    # For Fednova & SCAFFOLD
    config.aggregator.hyperparams.lr = config.clients.hyperparams.SGD_learning_rate

    # SCAFFOLD
    config.aggregator.hyperparams.total_client_count = clients

    # save config (include strategy)
    config.clients.hyperparams.strategy = strategy
    with (log_dir / f"config_{session}.json").open("w+") as f:
        session_config = config.dict()
        session_config.update({"mock_cold_start": mock_cold_start})
        f.write(json.dumps(session_config))
        # f.write(config.json())

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

    model = create_model(dataset)
    file_server = config.file_server.url if config.file_server is not None else None
    data_configs, max_test_clients_count = create_data_configs(
        dataset,
        clients,
        file_server,
    )  # , proxies=proxies)

    # cold start all clients in first round by default
    config.clients.hyperparams.cold_start = mock_cold_start
    clients = store_client_configs(
        session=session,
        strategy=strategy,
        clients=config.clients,
        num_clients=clients,
        data_configs=data_configs,
        database=config.database,
        stragglers_precentage=simulate_stragglers,
        mu=config.clients.hyperparams.fedprox.mu,
    )

    init_store_model(
        session=session,
        model=model,
        strategy=strategy,
        database=config.database,
        store_json_serializable=False,
    )

    # asyn execution only for FedScore
    sync = not strategy in ["fedasync", "fedscore"]
    # overwrite config if buffer ratio provided
    if buffer_ratio is not None:
        config.aggregator.hyperparams.buffer_ratio = buffer_ratio
    config.aggregator.hyperparams.is_synchronous = sync

    # if strategy only performs normal FedAvg overwrite tolerance (reset to 0)
    if strategy in ["fedavg", "fedprox", "fednova", "scaffold"]:
        config.aggregator.hyperparams.tolerance = 0

    inv_params = {
        "session": session,
        "cognito": config.cognito,
        "clients": clients,
        "evaluator_config": config.evaluator,
        "aggregator_config": config.aggregator,
        "mongodb_config": config.database,
        "allowed_stragglers": stragglers,
        "client_timeout": timeout,
        "evaluation_timeout": 60.0,
        "save_dir": log_dir,
        "global_test_data": (
            create_mnist_test_config(proxies=(proxies if proxy_in_evaluator else None))
            if dataset.lower() == "mnist"
            else None
        ),
        "proxies": proxies,
        "mock": mock,
        "mock_aggregator": mock_aggregator,
        "max_test_client_count": max_test_clients_count,
        "mock_cold_start": mock_cold_start,
        "is_synchronous": sync,
        "buffer_ratio": config.aggregator.hyperparams.buffer_ratio,
    }

    fedless_strategy = select_strategy(strategy, inv_params)

    t_start = time.time()
    asyncio.run(
        fedless_strategy.fit(
            n_clients_in_round=clients_in_round,
            max_rounds=rounds,
            max_accuracy=max_accuracy,
        )
    )
    t_end = time.time()

    # clear all client local controls from database
    if strategy == AggregationStrategy.SCAFFOLD:
        logger.info("Clearing local controls..")
        ClientControlDao(config.database).delete_local_controls(session_id=session)

    with (log_dir / f"x_Done_{session}.txt").open("w+") as file:
        file.write(f"DONE in: {round((t_end - t_start) / 60, 2)} / minutes")


def store_client_configs(
    session: str,
    strategy: str,
    clients: ClientFunctionConfigList,
    num_clients: int,
    data_configs: List[
        Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]
    ],
    database: MongodbConnectionConfig,
    stragglers_precentage: float,
    mu: float,  # for fedprox/fednova
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
    # stragglers_delay_list = [-1]

    # stragglers_delay_list = [10, 25, 50, 100, 200, 400]
    stragglers_delay_list = [-1]

    # clients.cool_start_time

    num_stragglers = int(stragglers_precentage * num_clients)
    logger.info(f"simulate stragglers {num_stragglers} clients for {num_clients}.")
    data_shards = iter(data_configs)
    function_iter = cycle(clients_unrolled)

    clients.hyperparams.strategy = strategy

    if strategy == AggregationStrategy.FedProx and clients.hyperparams.fedprox is None:
        st = FedProxParams()
        st.mu = mu
        clients.hyperparams.fedprox = st

    if strategy == AggregationStrategy.FedNova and clients.hyperparams.fednova is None:
        st = FedNovaParams()
        st.mu = mu
        clients.hyperparams.fednova = st

    if (
        strategy == AggregationStrategy.SCAFFOLD
        and clients.hyperparams.scaffold is None
    ):
        clients.hyperparams.scaffold = ScaffoldParams()

    default_hyperparms = clients.hyperparams
    final_configs = []
    stragglers_idx_list = random.sample(list(np.arange(num_clients)), num_stragglers)
    for idx, shard in enumerate(data_shards):
        client = next(function_iter)
        #  addition fedprox on specific functions or custom hp for specific functions
        hp = client.hyperparams or default_hyperparms
        client_id = str(uuid.uuid4())
        train_config, test_config = shard if isinstance(shard, tuple) else (shard, None)
        client_config = ClientConfig(
            session_id=session,
            client_id=client_id,
            function=client.function,
            data=train_config,
            test_data=test_config,
            hyperparams=hp,
        )
        # add straggler
        if idx in stragglers_idx_list:
            client_config.function.invocation_delay = random.sample(
                stragglers_delay_list, 1
            )[0]
            # client_config.function.invocation_delay = -2

        logger.debug(
            f"Initializing client {client_id} of type " f"{client.function.type}"
        )
        client_config_dao.save(client_config)
        final_configs.append(client_config)

        # only log history for FedlesScan
        if strategy == "fedlesscan":
            logger.debug(
                f"Initializing client_history for {client_id} of type "
                f"{client.function.type}"
            )
            client_history = ClientPersistentHistory(
                client_id=client_id,
                session_id=session,
            )
            client_history_dao.save(client_history)

    logger.info(
        f"Configured and stored all {len(data_configs)} clients configurations..."
    )
    return final_configs


if __name__ == "__main__":
    run()
