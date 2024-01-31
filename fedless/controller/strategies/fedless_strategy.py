import asyncio
import datetime
import itertools
import logging
import random
import time
import uuid
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import urllib3
from pydantic import ValidationError
from requests import Session
from fedless.common.models.models import InvocationHistory, InvocationStatus
from fedless.common.persistence.client_daos import ClientResultDao, InvocationHistoryDao

from fedless.controller.misc import fetch_cognito_auth_token
from fedless.controller.strategies.client_selection import ClientSelectionScheme
from fedless.controller.strategies.serverless_strategy import ServerlessFlStrategy
from fedless.controller.invocation import InvocationError, InvocationTimeOut

from fedless.controller.models import AggregationFunctionConfig, CognitoConfig
from fedless.common.models import (
    ClientConfig,
    MongodbConnectionConfig,
    DatasetLoaderConfig,
    InvocationResult,
    InvokerParams,
    FunctionInvocationConfig,
    AggregationStrategy,
)

logger = logging.getLogger(__name__)


class FedlessStrategy(ServerlessFlStrategy):
    def __init__(
        self,
        clients: List[ClientConfig],
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionInvocationConfig,
        aggregator_config: AggregationFunctionConfig,
        selection_strategy: ClientSelectionScheme,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FedAvg,
        client_timeout: float = 300,  # 5 mins default
        cognito: Optional[CognitoConfig] = None,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        allowed_stragglers: int = 0,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
        invocation_delay: float = None,
        evaluation_timeout: float = 30.0,  # 30 sec default
        mock: bool = False,
        mock_aggregator: bool = False,
        mock_cold_start: bool = False,
        max_test_client_count: int = 0,
        is_synchronous: bool = True,
        buffer_ratio: float = 0.5,
        **kwargs,
    ):

        super().__init__(
            session=session,
            clients=clients,
            mongodb_config=mongodb_config,
            evaluator_config=evaluator_config,
            aggregator_config=aggregator_config,
            global_test_data=global_test_data,
            selection_strategy=selection_strategy,
            aggregation_strategy=aggregation_strategy,
            is_synchronous=is_synchronous,
            client_timeout=client_timeout,
            allowed_stragglers=allowed_stragglers,
            save_dir=save_dir,
            proxies=proxies,
            invocation_delay=invocation_delay,
            mock=mock,
            mock_aggregator=mock_aggregator,
            max_test_client_count=max_test_client_count,
        )
        self.cognito = cognito
        self.evaluation_timeout = evaluation_timeout
        self.mock_cold_start = mock_cold_start
        self.buffer_ratio = buffer_ratio

    # *** MAIN ENTRYPOINY
    async def call_clients(
        self, round: int, clients: List[ClientConfig], evaluate_only: bool = False
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

        cold_client_ids = list(
            self.invocation_history_dao.get_cold_start_clients(self.session, clients)
        )

        # add invocation id, and init inv records
        # enable cold start if client is cold
        for client in clients:
            client.function.cool_start = (
                self.mock_cold_start and client.client_id in cold_client_ids
            )
            client.invocation_id = str(uuid.uuid4())

        if not evaluate_only:
            invocation_time = time.time()
            invocation_histories: List[InvocationHistory] = [
                InvocationHistory(
                    invocation_id=client.invocation_id,
                    client_id=client.client_id,
                    round_id=round,
                    session_id=self.session,
                    status=InvocationStatus.running,
                    cold_start=client.function.cool_start,
                    invocation_delay=client.function.invocation_delay,
                    invocation_time=invocation_time,
                )
                for client in clients
            ]
            self.invocation_history_dao.save_batch_invocations(invocation_histories)

        for client in clients:
            session = Session()
            session.headers.update(http_headers)
            session.proxies.update(self.proxies)
            # session = retry_session(backoff_factor=1.0, session=session)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                invocation_id=client.invocation_id,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
                evaluate_only=evaluate_only,
                invocation_delay=client.function.invocation_delay,
            )

            cold_start_duration = client.hyperparams.cold_start_duration

            # function with closure for easier logging
            async def _inv(
                function: FunctionInvocationConfig,
                data: InvokerParams,
                session: Session,
                round: int,
                client_id: str,
                invocation_id: str,
            ):
                try:
                    # TODO: invocation delay
                    if self.invocation_delay:
                        await asyncio.sleep(random.uniform(0.0, self.invocation_delay))

                    # normal distribution for cold start duration
                    if function.cool_start:
                        await asyncio.sleep(
                            abs(
                                random.gauss(
                                    cold_start_duration, cold_start_duration * 0.1
                                )
                            )
                        )

                    t_start = time.time()
                    logger.debug(
                        f"***->>> invoking client {client_id} with timeout {self.client_timeout}\n ---->>> invocation_id: {data.invocation_id}"
                    )
                    res = await self.invoke_async(
                        function,
                        data.dict(),
                        session=session,
                        timeout=self.client_timeout
                        if not evaluate_only
                        else self.evaluation_timeout,
                    )
                    t_end = time.time()
                    dt_call = t_end - t_start

                    # logs for only training
                    if not data.evaluate_only:
                        # log training time
                        self.client_timings.append(
                            {
                                "client_id": client_id,
                                "session_id": self.session,
                                "invocation_id": invocation_id,
                                "invocation_time": t_start,
                                "complete_time": t_end,
                                "seconds": dt_call,
                                "cold_start": function.cool_start,
                                "function": function.json(),
                                "url": function.params.url if hasattr(function.params, 'url') else None,
                                "eval": evaluate_only,
                                "round": round,
                            }
                        )

                        self.invocation_history_dao.inv_done(
                            invocation_id,
                            t_start,
                            t_end,
                        )
                    return res
                except InvocationTimeOut as e:
                    if not data.evaluate_only:
                        self.invocation_history_dao.inv_timeout(invocation_id)
                    return str(e)
                except InvocationError as e:
                    if not data.evaluate_only:
                        self.invocation_history_dao.inv_fail(invocation_id)
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
                        invocation_id=client.invocation_id,
                    ),
                    name=client.client_id,
                )
            )

        if self.is_synchronous or evaluate_only:
            # aggregate synchronous: wait till time out
            done, pending = await asyncio.wait(tasks)
            results = list(map(lambda f: f.result(), done))
        else:
            result_dao = ClientResultDao(db=self.mongodb_config)

            tolerance = self._aggregator.hyperparams.tolerance
            tolerance_round = round - tolerance  # current_round - tolerance

            # anonymous function for result counting
            result_dao = ClientResultDao(db=self.mongodb_config)
            count_results = lambda: result_dao.count_available_results(
                session_id=self.session, tolerance_round=tolerance_round
            )

            inv_history_dao = self.invocation_history_dao
            count_running = lambda: inv_history_dao.get_running_count(
                session_id=self.session, tolerance_round=tolerance_round
            )

            # count results every 0.1 seconds, till buffer is filled
            buffer_size = np.ceil(self.buffer_ratio * len(clients)).astype("int")
            while (
                total_results := count_results()
            ) < buffer_size and count_running() != 0:
                await asyncio.sleep(0.1)

            done = list(filter(lambda task: task.done(), tasks))
            untracked_done = total_results - len(done)

            logger.info(
                f"[Asyn] {total_results} (buffer={buffer_size}) results found in database "
            )

            results = list(map(lambda f: f.result(), done))

            # add untracked dummy result (to prevent error)
            results += [
                InvocationResult(
                    session_id="Async: untracked result",
                    round_id=-1,
                    client_id="Async: untracked result",
                ).dict()
            ] * untracked_done

        suc, errs = [], []
        for res in results:
            try:
                suc.append(InvocationResult.parse_obj(res))
            except ValidationError:
                errs.append(res)
        return suc, errs

    async def evaluate_clients(self, round: int, clients: List[ClientConfig]) -> Dict:
        succ, fails = await self.call_clients(round, clients, evaluate_only=True)
        logger.info(
            f"{len(succ)} client evaluations returned, {len(fails)} failures... {fails}"
        )
        client_metrics = [res.test_metrics for res in succ]
        return self.aggregate_metrics(
            metrics=client_metrics, metric_names=["loss", "accuracy"]
        )
