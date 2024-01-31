import asyncio
import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
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
from fedless.controller.misc import run_in_executor
from fedless.controller.mocks.mock_aggregation import MockAggregator
from fedless.controller.mocks.mock_client import MockClient
from fedless.controller.models import AggregationFunctionConfig
from fedless.controller.strategies.fl_strategy import FLStrategy
from fedless.controller.strategies.Intelligent_selection import ClientSelectionScheme

logger = logging.getLogger(__name__)


class ServerlessFlStrategy(FLStrategy, ABC):
    def __init__(
        self,
        clients: List,
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionInvocationConfig,
        aggregator_config: AggregationFunctionConfig,
        selection_strategy: ClientSelectionScheme,
        aggregation_strategy: AggregationStrategy,
        client_timeout: float = 300,
        allowed_stragglers: int = 0,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
        invocation_delay: float = None,
        mock: bool = False,
        max_test_client_count: int = 0,
    ):
        super().__init__(
            clients=clients,
            selectionStrategy=selection_strategy,
            aggregation_strategy=aggregation_strategy,
        )
        # if clients for test are zero mean all the clients can be used for testing
        # other than that we have the number of the clients used for testing
        if max_test_client_count == 0:
            self.max_test_client_count = len(clients)
        else:
            self.max_test_client_count = max_test_client_count
        urllib3.disable_warnings()
        self.session: str = session or str(uuid.uuid4())
        self.log_metrics = []
        self.client_timings = []
        self.allowed_stragglers = allowed_stragglers

        self.mongodb_config = mongodb_config
        self.global_test_data = global_test_data

        self._aggregator: Optional[AggregationFunctionConfig] = aggregator_config
        self._evaluator: Optional[FunctionInvocationConfig] = evaluator_config

        self.client_timeout: float = client_timeout
        self.clients: List[ClientConfig] = clients
        self.save_dir = save_dir
        self.proxies = proxies or {}
        self.invocation_delay: Optional[float] = invocation_delay
        self.mock = mock

    def save_round_results(self, session: str, round: int, dir: Optional[Path] = None, **kwargs) -> None:
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

    @run_in_executor
    def _async_call_request(
        self,
        function: FunctionInvocationConfig,
        data: Dict,
        session: Optional[Session] = None,
        timeout: float = 300,
    ) -> Dict:
        # session = retry_session(backoff_factor=0.5, retries=5, session=session)
        return invoke_sync(function_config=function, data=data, session=session, timeout=timeout)

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

    # Mock function call TODO:remove
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

    async def invoke_async(
        self,
        function: FunctionInvocationConfig,
        data: Dict,
        session: Optional[Session] = None,
        timeout: float = 300,
    ) -> Dict:
        return await self._async_call_request(function, data, session, timeout=timeout)

    async def fit_round(self, round: int, clients: List[ClientConfig]) -> Tuple[float, float, Dict]:
        round_start_time = time.time()
        metrics_misc = {}
        loss, acc = None, None

        all_clients: Set = set(map(lambda client: client.client_id, clients))
        # Invoke clients
        t_clients_start = time.time()
        succs, errors = await self.call_clients(round, clients)
        # add last failed round idx
        client_history_dao = ClientHistoryDao(db=self.mongodb_config)

        preps_dict = {
            "succs": len(succs),
            "failed": len(errors),
            "pending": len(all_clients) - len(succs) - len(errors),
        }
        self.save_invocation_details(self.session, round, self.save_dir, **preps_dict)
        # reset client backoff
        for suc in succs:
            successfull_client: ClientPersistentHistory = client_history_dao.load(suc.client_id)
            successfull_client.client_backoff = 0
            client_history_dao.save(successfull_client)
            all_clients.remove(suc.client_id)
        # todo failed get the backoff
        # pending does not

        # add client backoff and add the missing rounds
        # the backoff is computed from the last missed round
        for failed_client_id in all_clients:

            failed_client: ClientPersistentHistory = client_history_dao.load(failed_client_id)
            failed_client.missed_rounds.append(round)
            if failed_client.client_backoff <= 0:
                failed_client.client_backoff = 1
            else:
                # todo change to linear backoff
                # todo exponential backoff
                logger.info(f"using exponential backoff for client {failed_client_id}")
                failed_client.client_backoff *= 2
            client_history_dao.save(failed_client)

        if len(succs) < (len(clients) - self.allowed_stragglers):
            logger.error(errors)
            raise Exception(
                f"Only {len(succs)}/{len(clients)} clients finished this round, "
                f"required are {len(clients) - self.allowed_stragglers}."
            )
        logger.info(f"Received results from {len(succs)}/{len(clients)} client functions")

        t_clients_end = time.time()

        logger.info(f"Invoking Aggregator")
        t_agg_start = time.time()
        agg_res: AggregatorFunctionResult = (
            self.call_mock_aggregator(round) if self.mock else self.call_aggregator(round)
        )
        # agg_res = self.call_aggregator(round)
        # agg_res: AggregatorFunctionResult = self.call_mock_aggregator(round)
        t_agg_end = time.time()
        logger.info(f"Aggregator combined result of {agg_res.num_clients} clients.")
        metrics_misc["aggregator_seconds"] = t_agg_end - t_agg_start

        if agg_res.global_test_results:
            # logger.info(f"Running global evaluator function")
            # t_eval_start = time.time()
            # eval_res = self.call_evaluator(round)
            # t_eval_end = time.time()
            # metrics_misc["evaluator_seconds"] = t_eval_end - t_eval_start
            loss = agg_res.global_test_results.metrics.get("loss")
            acc = agg_res.global_test_results.metrics.get("accuracy")
        else:
            logger.info(f"Computing test statistics from clients")
            if hasattr(self, "evaluate_clients"):
                n_clients_in_round = len(clients)
                # randomly select clients form evaluation
                # choose subset for testing if applicable
                eval_clients_list = self.clients[: min(len(clients), self.max_test_client_count)]
                eval_clients = random.sample(
                    eval_clients_list,
                    min(n_clients_in_round, self.max_test_client_count),
                )
                logger.info(f"Selected {len(eval_clients)} for evaluation...")
                metrics = await self.evaluate_clients(agg_res.new_round_id, eval_clients)
            else:
                if not agg_res.test_results:
                    raise ValueError(f"Clients or aggregator did not return local test results...")
                metrics = self.aggregate_metrics(metrics=agg_res.test_results, metric_names=["loss", "accuracy"])
            loss = metrics.get("mean_loss")
            acc = metrics.get("mean_accuracy")

        metrics_misc.update(
            {
                "round_seconds": time.time() - round_start_time,
                "clients_finished_seconds": t_clients_end - t_clients_start,
                "num_clients_round": len(clients),
                "global_test_accuracy": acc,
                "global_test_loss": loss,
            }
        )

        logger.info(f"Round {round}: loss={loss}, acc={acc}")
        self.save_round_results(
            session=self.session,
            round=round,
            dir=self.save_dir,
            **metrics_misc,
        )
        return loss, acc, metrics_misc

    @abstractmethod
    async def call_clients(self, round: int, clients: List[ClientConfig]) -> Tuple[List[InvocationResult], List[str]]:
        pass
