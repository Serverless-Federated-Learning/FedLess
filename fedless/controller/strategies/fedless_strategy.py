import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import urllib3
from pydantic import ValidationError
from requests import Session

from fedless.common.models import (
    AggregationStrategy,
    ClientConfig,
    DatasetLoaderConfig,
    FunctionInvocationConfig,
    InvocationResult,
    InvokerParams,
    MongodbConnectionConfig,
)
from fedless.controller.invocation import InvocationError
from fedless.controller.misc import fetch_cognito_auth_token
from fedless.controller.models import AggregationFunctionConfig, CognitoConfig
from fedless.controller.strategies.Intelligent_selection import ClientSelectionScheme
from fedless.controller.strategies.serverless_strategy import ServerlessFlStrategy

logger = logging.getLogger(__name__)


class FedlessStrategy(ServerlessFlStrategy):
    def __init__(
        self,
        clients: List[ClientConfig],
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionInvocationConfig,
        aggregator_config: AggregationFunctionConfig,
        selection_strategy: ClientSelectionScheme,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.PER_ROUND,
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
        max_test_client_count: int = 0,
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
            client_timeout=client_timeout,
            allowed_stragglers=allowed_stragglers,
            save_dir=save_dir,
            proxies=proxies,
            invocation_delay=invocation_delay,
            mock=mock,
            max_test_client_count=max_test_client_count,
        )
        self.cognito = cognito
        self.evaluation_timeout = evaluation_timeout

    async def call_clients(
        self, round: int, clients: List[ClientConfig], evaluate_only: bool = False, algorithm: str = "fedless"
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
            # session = retry_session(backoff_factor=1.0, session=session)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
                evaluate_only=evaluate_only,
                invocation_delay=client.function.invocation_delay,
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
                                "algorith": algorithm,
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

    async def evaluate_clients(self, round: int, clients: List[ClientConfig]) -> Dict:
        succ, fails = await self.call_clients(round, clients, evaluate_only=True)
        logger.info(f"{len(succ)} client evaluations returned, {len(fails)} failures... {fails}")
        client_metrics = [res.test_metrics for res in succ]
        return self.aggregate_metrics(metrics=client_metrics, metric_names=["loss", "accuracy"])
