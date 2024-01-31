import asyncio
import logging
import os
import random
import time
import uuid
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import urllib3
from pydantic import ValidationError
from requests import Session

from fedless.common.models import (
    AggregationStrategy,
    AggregatorFunctionParams,
    AggregatorFunctionResult,
    ClientConfig,
    ClientPersistentHistory,
    DatasetLoaderConfig,
    EvaluatorParams,
    EvaluatorResult,
    FunctionInvocationConfig,
    InvocationResult,
    InvokerParams,
    MongodbConnectionConfig,
)
from fedless.common.persistence.client_daos import ClientHistoryDao
from fedless.controller.invocation import InvocationError, invoke_sync, retry_session
from fedless.controller.misc import fetch_cognito_auth_token, run_in_executor
from fedless.controller.mocks.mock_aggregation import MockAggregator
from fedless.controller.mocks.mock_client import MockClient
from fedless.controller.models import AggregationFunctionConfig, CognitoConfig
from fedless.controller.strategies.fedmd_ray_trainer import FedMDRayTrainer
from fedless.controller.strategies.fl_strategy import FLStrategy
from fedless.controller.strategies.Intelligent_selection import ClientSelectionScheme
from fedless.datasets.clean_db import clean_db

logger = logging.getLogger(__name__)


class FedMDStrategy(FLStrategy, ABC):
    def __init__(
        self,
        clients: List,
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionInvocationConfig,
        aggregator_config: AggregationFunctionConfig,
        selection_strategy: ClientSelectionScheme,
        aggregation_strategy: AggregationStrategy,
        client_timeout: float = 300,
        cognito: Optional[CognitoConfig] = None,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        allowed_stragglers: int = 0,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
        invocation_delay: float = None,
        evaluation_timeout: float = 30.0,
        mock: bool = False,
        max_test_client_count: int = 0,
    ):

        super().__init__(clients, selection_strategy, aggregation_strategy)

        self.session: str = session or str(uuid.uuid4())
        self.log_metrics = []
        self.client_timings = []

        self.mongodb_config = mongodb_config
        self.global_test_data = global_test_data

        self.proxies = proxies or {}

        self.cognito = cognito
        self.evaluation_timeout = evaluation_timeout

        self._aggregator: Optional[AggregationFunctionConfig] = aggregator_config
        self._evaluator: Optional[FunctionInvocationConfig] = evaluator_config

        self.client_timeout: float = client_timeout
        self.clients: List[ClientConfig] = clients
        self.save_dir = save_dir
        self.invocation_delay: Optional[float] = invocation_delay
        self.mock = mock

        self.max_test_client_count = max_test_client_count

    def save_round_results(self, session: str, round: str, dir: Optional[Path] = None, **kwargs) -> None:
        self.log_metrics.append({"session_id": session, "round_id": round, **kwargs})

        if not dir:
            dir = Path.cwd()
        pd.DataFrame.from_records(self.log_metrics).to_csv(dir / f"timing_{session}.csv", index=False)
        pd.DataFrame.from_records(self.client_timings).to_csv(dir / f"clients_{session}.csv", index=False)

    def save_invocation_details(self, session: str, round: int, dir: Optional[Path] = None, **kwargs) -> None:

        if not dir:
            dir = Path.cwd()
        preps_dict = {"session_id": session, "round_id": round, **kwargs}
        add_header = not os.path.isfile(dir / f"invocation_{session}.csv")
        pd.DataFrame.from_records([preps_dict]).to_csv(
            dir / f"invocation_{session}.csv", mode="a", index=False, header=add_header
        )

    @property
    def aggregator(self) -> AggregationFunctionConfig:
        if not self._aggregator:
            raise ValueError()
        return self._aggregator

    @property
    def evaluator(self) -> FunctionInvocationConfig:
        if not self._evaluator:
            raise ValueError()
        return self._evaluator

    # Comment: To convert sync requests to async requests, we pass this decorator - run_in_executor
    @run_in_executor
    def _async_call_request(
        self,
        function: FunctionInvocationConfig,
        data: Dict,
        session: Optional[Session] = None,
        timeout: float = 300,
    ) -> Dict:
        return invoke_sync(function_config=function, data=data, session=session, timeout=timeout)

    async def invoke_async(
        self,
        function: FunctionInvocationConfig,
        data: Dict,
        session: Optional[Session] = None,
        timeout: float = 300,
    ) -> Dict:
        return await self._async_call_request(function, data, session, timeout=timeout)

    # Mock function call
    def call_mock_aggregator(self, round: int) -> AggregatorFunctionResult:
        params = AggregatorFunctionParams(
            session_id=self.session,
            round_id=round,
            database=self.mongodb_config,
            test_data=self.global_test_data,
            aggregation_strategy=self.aggregation_strategy,
            aggregation_hyper_params=self.aggregator.hyperparams,
        )
        aggregator = MockAggregator(params=params)
        result = aggregator.run_aggregator()
        try:
            return AggregatorFunctionResult.parse_obj(result)
        except ValidationError as e:
            raise ValueError(f"Aggregator returned invalid result.") from e

    def call_aggregator(self, round: int) -> AggregatorFunctionResult:
        params = AggregatorFunctionParams(
            session_id=self.session,
            round_id=round,
            database=self.mongodb_config,
            test_data=self.global_test_data,
            aggregation_strategy=self.aggregation_strategy,
            aggregation_hyper_params=self.aggregator.hyperparams
            # **self.aggregator.hyperparams
        )
        session = Session()
        # session.proxies.update(self.proxies)
        result = invoke_sync(
            self.aggregator.function,
            data=params.dict(),
            session=retry_session(backoff_factor=1.0, retries=5, session=session),
        )
        try:
            return AggregatorFunctionResult.parse_obj(result)
        except ValidationError as e:
            raise ValueError(f"Aggregator returned invalid result.") from e

    async def inv_mock(self, function, data: InvokerParams, session: Session, round, client_id):
        try:
            if self.invocation_delay:
                await asyncio.sleep(random.uniform(0.0, self.invocation_delay))
            t_start = time.time()
            logger.info(f"***->>> invoking client ${client_id} with time out ${self.client_timeout}")
            cl = MockClient(data)

            res = await cl.run_client()

            dt_call = time.time() - t_start
            self.client_timings.append(
                {
                    "client_id": client_id,
                    "session_id": self.session,
                    "invocation_time": t_start,
                    "function": {"function": "mock"},
                    "seconds": dt_call,
                    "eval": data.evaluate_only,
                    "round": round,
                }
            )
            return res
        except InvocationError as e:
            return str(e)

    def call_evaluator(self, round: int) -> EvaluatorResult:
        params = EvaluatorParams(
            session_id=self.session,
            round_id=round + 1,
            database=self.mongodb_config,
            test_data=self.global_test_data,
        )
        session = Session()
        # session.proxies.update(self.proxies)
        result = invoke_sync(
            self.evaluator,
            data=params.dict(),
            session=retry_session(backoff_factor=1.0, retries=5, session=session),
        )
        try:
            return EvaluatorResult.parse_obj(result)
        except ValidationError as e:
            raise ValueError(f"Evaluator returned invalid result.") from e

    def handle_client_inv_results(self, clients, success, errors):

        all_clients: Set = set(map(lambda client: client.client_id, clients))

        client_history_dao = ClientHistoryDao(db=self.mongodb_config)
        preps_dict = {
            "succs": len(success),
            "failed": len(errors),
            "pending": len(all_clients) - len(success) - len(errors),
        }

        self.save_invocation_details(self.session, round, self.save_dir, **preps_dict)

        # reset client backoff
        for suc in success:
            successfull_client: ClientPersistentHistory = client_history_dao.load(suc.client_id)
            successfull_client.client_backoff = 0
            client_history_dao.save(successfull_client)
            all_clients.remove(suc.client_id)

        if len(success) != (len(clients)):
            logger.error(errors)
            raise Exception(
                f"Only {len(success)}/{len(clients)} clients finished this round, " f"required are {len(clients)}."
            )

        logger.info(f"Received results from {len(success)}/{len(clients)} client functions")

    async def fit(self, n_clients_in_round: int, max_rounds: int, max_accuracy: Optional[float] = None):

        # Number of clients in round must be equal to total number of clients.
        # Selection strategy will just randomly shuffle the client order once.
        clients = self.selectionStrategy.select_clients(
            n_clients_in_round=n_clients_in_round, clients_pool=self.clients, round=None, max_rounds=max_rounds
        )
        logger.info(f"Using {len(self.clients)} clients/models for the training process")

        # Perform Initial Training on public data and private data, also uses Early Stopping

        # succs, errors = await self.call_clients(round=-2, clients=clients, action="transfer_learning_public")
        # self.handle_client_inv_results(self.clients, succs, errors)

        # Perform initial public data training on the controller server with option to use a Ray cluster for parallel training
        logger.info("Performing initial public data training until convergence for each client")

        # clean_db(self.mongodb_config, ["de2c7cc3-c67f-4b19-8af9-efc126451feb", "9cd9c958-d446-4ee7-8e3a-17de5ed18b00"])

        init_training_results = FedMDRayTrainer(
            validation_split=0.2,
            min_delta=0.001,
            patience=8,
            session_id=self.session,
            clients=clients,
            round_id=-3,
            db=self.mongodb_config,
            train_local=False,
            use_existing_model=True,
            existing_session_id="a8990eec-032d-48b9-bc26-b62151b3fddc",
            cluster_svc_addr="10.244.4.179:10001",
        ).fit_parallel()

        logger.info(
            f"Completed the initial public training on the central server for all clients of session: {self.session}"
        )
        # clean_db(self.mongodb_config, ["de2c7cc3-c67f-4b19-8af9-efc126451feb", "9cd9c958-d446-4ee7-8e3a-17de5ed18b00","4d504e61-ba79-405e-b439-c48bfb3e8f29", self.session ])

        init_metrics = {}
        for _, row in init_training_results.iterrows():
            client_id = row["client_id"]
            init_metrics[client_id + "_public_test_accuracy"] = row["test_accuracy"]
            init_metrics[client_id + "_public_test_loss"] = row["test_loss"]
            init_metrics[client_id + "_total_time_ray"] = row["total_time_ray"]

        logger.info("Performing initial private data training until convergence")
        succs, errors = await self.call_clients(round=-2, clients=clients, action="transfer_learning_private")
        self.handle_client_inv_results(self.clients, succs, errors)

        # Initial training complete
        # Getting test accuracy after initial public training and fine tuning on private dataset
        logger.info(
            "Performing initial evaluation of public+private dataset training before starting collaboration phase"
        )
        succs, errors = await self.call_clients(round=-1, clients=clients, action=None, evaluate_only=True)

        for succ_res in succs:
            test_loss = succ_res.test_metrics.metrics.get("loss")
            test_acc = succ_res.test_metrics.metrics.get("accuracy")
            cardinality = succ_res.test_metrics.cardinality
            client_id = succ_res.client_id

            init_metrics[client_id + "_cardinality"] = cardinality
            init_metrics[client_id + "_test_loss"] = test_loss
            init_metrics[client_id + "_test_accuracy"] = test_acc

        self.save_round_results(
            session=self.session,
            round="init",
            dir=self.save_dir,
            **init_metrics,
        )

        # Now we start collaboration fitting rounds
        for round in range(max_rounds):

            logger.info(f"Sampled {len(self.clients)} for round {round}")

            # This has to be for all individual clients
            metrics = await self.fit_round(round, clients)

            # Get individual client public test accuracy after each round
            succs, errors = await self.call_clients(round=round, clients=clients, action=None, evaluate_only=True)

            # Updating test metrics for each model type
            for succ_res in succs:
                test_loss = succ_res.test_metrics.metrics.get("loss")
                test_acc = succ_res.test_metrics.metrics.get("accuracy")
                cardinality = succ_res.test_metrics.cardinality
                client_id = succ_res.client_id

                metrics[client_id + "_cardinality"] = cardinality
                metrics[client_id + "_test_loss"] = test_loss
                metrics[client_id + "_test_accuracy"] = test_acc

            logger.info(f"Round {round} finished. Round Metrics: ={metrics}")
            self.save_round_results(
                session=self.session,
                round=round,
                dir=self.save_dir,
                **metrics,
            )

    async def fit_round(self, round: int, clients: List[ClientConfig]) -> Tuple[float, float, Dict]:

        round_start_time = time.time()
        metrics_misc = {}

        # Invoke clients for running forward pass and then storing resulting logits back to MongoDB
        logger.info("Performing communication step.")
        t_client_communicate_start = time.time()
        succs, errors = await self.call_clients(round, clients, action="communicate")
        self.handle_client_inv_results(clients, succs, errors)

        t_client_communicate_end = time.time()

        # Now run the aggregator and it stores results to MongoDB
        logger.info(f"Invoking Aggregator")
        t_agg_start = time.time()
        agg_res: AggregatorFunctionResult = (
            self.call_mock_aggregator(round) if self.mock else self.call_aggregator(round)
        )
        t_agg_end = time.time()
        logger.info(f"Aggregator combined result of {agg_res.num_clients} clients.")

        logger.info("Performing digest step.")
        t_client_digest_start = time.time()
        succs, errors = await self.call_clients(round, clients, action="digest")
        self.handle_client_inv_results(clients, succs, errors)
        logger.info("Performing revisit step.")
        t_client_digest_end = time.time()
        succs, errors = await self.call_clients(round, clients, action="revisit")
        self.handle_client_inv_results(clients, succs, errors)
        t_client_revisit_end = time.time()

        metrics_misc.update(
            {
                "round_seconds": time.time() - round_start_time,
                "clients_communicate_seconds": t_client_communicate_end - t_client_communicate_start,
                "clients_digest_seconds": t_client_digest_end - t_client_digest_start,
                "clients_revisit_seconds": t_client_revisit_end - t_client_digest_end,
                "aggregator_seconds": t_agg_end - t_agg_start,
                "num_clients_round": len(clients),
            }
        )

        return metrics_misc

    # Function for strategically calling the clients
    # Options for actions - digest, revisit, communicate, transfer_learning
    async def call_clients(
        self,
        round: int,
        clients: List[ClientConfig],
        evaluate_only: bool = False,
        action: str = "transfer_learning_public",
        algorithm: str = "fedmd",
    ) -> Tuple[List[InvocationResult], List[str]]:
        urllib3.disable_warnings()
        tasks = []

        http_headers = {}
        if self.cognito:
            token = fetch_cognito_auth_token(
                user_pool_id=self.cognito.user_pool_id,
                region_name=self.cognito.region_name,
                auth_endpoint=self.cognito.auth_endpoint,
                invoker_client_id=self.cognito.invoker_client_id,
                invoker_client_secret=self.cognito.invoker_client_secret,
                required_scopes=self.cognito.required_scopes,
            )
            http_headers = {"Authorization": f"Bearer {token}"}

        for client in clients:
            session = Session()
            session.headers.update(http_headers)
            session.proxies.update(self.proxies)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
                evaluate_only=evaluate_only,
                invocation_delay=client.function.invocation_delay,
                action=action,
                algorithm=algorithm,
            )

            # function with closure for easier logging
            async def _inv(function, data, session, round, client_id):
                try:
                    if self.invocation_delay:
                        await asyncio.sleep(random.uniform(0.0, self.invocation_delay))
                    t_start = time.time()
                    logger.info(f"***->>> invoking client ${client_id} with time out ${self.client_timeout}")
                    res = await self.invoke_async(
                        function,
                        data.dict(),
                        session=session,
                        timeout=self.client_timeout if not evaluate_only else self.evaluation_timeout,
                    )
                    dt_call = time.time() - t_start

                    # logs for only training
                    if not data.evaluate_only:
                        self.client_timings.append(
                            {
                                "client_id": client_id,
                                "session_id": self.session,
                                "invocation_time": t_start,
                                "function": function.json(),
                                "seconds": dt_call,
                                "eval": evaluate_only,
                                "round": round,
                                "action": action,
                                "algorithm": algorithm,
                            }
                        )
                    return res
                except InvocationError as e:
                    return str(e)

            client_invoker_func = self.inv_mock if self.mock else _inv
            tasks.append(
                asyncio.create_task(
                    client_invoker_func(
                        function=client.function,
                        data=params,
                        session=session,
                        round=round,
                        client_id=client.client_id,
                    ),
                    name=client.client_id,
                )
            )

        done, pending = await asyncio.wait(tasks)
        results = list(map(lambda f: f.result(), done))
        suc, errs = [], []
        for res in results:
            try:
                suc.append(InvocationResult.parse_obj(res))
            except ValidationError:
                errs.append(res)
        return suc, errs
