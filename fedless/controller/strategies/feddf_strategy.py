import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import urllib3
from pydantic import ValidationError
from requests import Session

from fedless.common.models import (
    AggregationStrategy,
    AggregatorFunctionParams,
    AggregatorFunctionResult,
    ClientConfig,
    DatasetLoaderConfig,
    FunctionInvocationConfig,
    InvocationResult,
    InvokerParams,
    MongodbConnectionConfig,
)
from fedless.common.persistence.client_daos import (
    ClientHistoryDao,
    ClientModelDao,
    ClientPersistentHistory,
)
from fedless.controller.invocation import InvocationError, retry_session
from fedless.controller.misc import fetch_cognito_auth_token
from fedless.controller.mocks.mock_aggregation import MockAggregator
from fedless.controller.models import AggregationFunctionConfig, CognitoConfig
from fedless.controller.strategies.Intelligent_selection import ClientSelectionScheme
from fedless.controller.strategies.serverless_strategy import ServerlessFlStrategy

logger = logging.getLogger(__name__)


class FedDFStrategy(ServerlessFlStrategy):
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
        self.session = session
        self.evaluation_timeout = evaluation_timeout
        self.aggregator_config = aggregator_config
        self.mongodb_config = mongodb_config

    async def fit(self, n_clients_in_round: int, max_rounds: int, max_accuracy: float):

        aggregators = self.aggregator_config.agg_functions
        for round in range(max_rounds):
            # clients = self.sample_clients(n_clients_in_round, self.clients)
            clients = self.selectionStrategy.select_clients(n_clients_in_round, self.clients, round, max_rounds)
            logger.info(f"Sampled {len(clients)} for round {round}")
            metrics = await self.fit_round(round, clients, aggregators)
            logger.info(f"Round {round} finished.")

    async def call_aggregators(
        self, round: int, aggregators: List[str], model_types: List[str], action: str
    ) -> List[AggregatorFunctionResult]:

        urllib3.disable_warnings()
        tasks = []

        assert len(aggregators) >= len(
            model_types
        ), "Aggregator functions are less than unique model prototypes. Please define more aggregators."

        for idx, model_type in enumerate(model_types):
            session = Session()
            params = AggregatorFunctionParams(
                session_id=self.session,
                round_id=round,
                database=self.mongodb_config,
                test_data=self.global_test_data,
                aggregation_strategy=self.aggregation_strategy,
                model_type=model_type,
                action=action,
                aggregation_hyper_params=aggregators[idx].hyperparams
                # **self.aggregator.hyperparams
            )

            # function with closure for easier logging
            async def _inv(function, params):
                logger.info(f"***->>> invoking aggregator {function.params.url} for model type: {params.model_type}")

                res = await self.invoke_async(
                    function,
                    params.dict(),
                    session=session,
                    timeout=self.client_timeout,
                )

                try:
                    return AggregatorFunctionResult.parse_obj(res)
                except ValidationError as e:
                    raise ValueError(f"Aggregator returned invalid result.") from e

            aggregator_invoker_func = self.inv_mock_aggregator if self.mock else _inv
            # aggregator_invoker_func = self.inv_mock_aggregator
            tasks.append(
                asyncio.create_task(
                    aggregator_invoker_func(function=aggregators[idx].function, params=params),
                    name=model_type,
                )
            )

        done, pending = await asyncio.wait(tasks)
        results = list(map(lambda f: f.result(), done))
        suc, errs = [], []
        for res in results:
            try:
                suc.append(AggregatorFunctionResult.parse_obj(res))
            except ValidationError:
                errs.append(res)
        return suc, errs

    async def inv_mock_aggregator(self, function, params):

        logger.info(f"***->>> invoking mock aggregator for model type: {params.model_type}")
        cl = MockAggregator(params=params)

        res = cl.run_aggregator()

        try:
            return AggregatorFunctionResult.parse_obj(res)
        except ValidationError as e:
            raise ValueError(f"Aggregator returned invalid result.") from e

    async def fit_round(
        self, round: int, clients: List[ClientConfig], aggregators: List[AggregationFunctionConfig]
    ) -> Tuple[float, float, Dict]:
        round_start_time = time.time()
        metrics_misc = {}

        all_clients: Set = set(map(lambda client: client.client_id, clients))

        client_model_dao = ClientModelDao(db=self.mongodb_config)
        model_client_mappings = client_model_dao.get_model_client_mapping(session_id=self.session)
        unique_models_in_round = []
        for model, mapped_clients in model_client_mappings.items():
            if len(all_clients.intersection(mapped_clients)) > 0:
                unique_models_in_round.append(model)

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

        # Call aggregators for model level FedAvg
        # t_agg_fedavg_start = time.time()
        # agg_succs, agg_errors = await self.call_aggregators(round, aggregators, unique_models_in_round, "fedavg")
        # t_agg_fedavg_end = time.time()

        # assert len(agg_errors) == 0, "All aggregators did not complete succesfully!"

        # metrics_misc["aggregator_fedvg_seconds"] = t_agg_fedavg_end - t_agg_fedavg_start

        logger.info(f"Invoking Aggregators for ensemble distillation")

        # Call aggregators for model level FedAvg
        t_agg_distillation_start = time.time()
        agg_succs, agg_errors = await self.call_aggregators(round, aggregators, unique_models_in_round, "distillation")
        t_agg_distillation_end = time.time()

        assert len(agg_errors) == 0, "All aggregators did not complete succesfully!"

        metrics_misc["aggregator_distillation_seconds"] = t_agg_distillation_end - t_agg_distillation_start

        test_results = []

        # Updating test metrics for each model type
        # Individual client evaluation and averaging for Shakespeare else perofrming global aggregator evaluation
        if self.clients[0].data.train_data.params.dataset == "shakespeare":

            # Calling all clients for evaluation
            succs, errors = await self.call_clients(round, self.clients, evaluate_only=True)
            for succ_res in succs:
                test_loss = succ_res.test_metrics.metrics.get("loss")
                test_acc = succ_res.test_metrics.metrics.get("accuracy")
                cardinality = succ_res.test_metrics.cardinality
                client_id = succ_res.client_id

                metrics_misc[client_id + "_cardinality"] = cardinality
                metrics_misc[client_id + "_test_loss"] = test_loss
                metrics_misc[client_id + "_test_accuracy"] = test_acc
        else:
            for agg_succ in agg_succs:
                test_loss = agg_succ.test_results.metrics.get("loss")
                test_acc = agg_succ.test_results.metrics.get("accuracy")
                model_type = agg_succ.test_results.metrics.get("model_type")
                test_results.append({"model_type": model_type, "test_loss": test_loss, "test_accuracy": test_acc})

            for test_result in test_results:
                model_type = test_result["model_type"]
                metrics_misc[model_type + "_test_loss"] = test_result["test_loss"]
                metrics_misc[model_type + "_test_accuracy"] = test_result["test_accuracy"]

        metrics_misc.update(
            {
                "round_seconds": time.time() - round_start_time,
                "clients_finished_seconds": t_clients_end - t_clients_start,
                "num_clients_round": len(clients),
            }
        )

        logger.info(f"Round {round}: Metrics={metrics_misc}")
        self.save_round_results(
            session=self.session,
            round=round,
            dir=self.save_dir,
            **metrics_misc,
        )
        return metrics_misc

    async def call_clients(
        self, round: int, clients: List[ClientConfig], evaluate_only: bool = False, algorithm: str = "feddf"
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

    async def evaluate_clients(self, round: int, clients: List[ClientConfig]) -> Dict:
        succ, fails = await self.call_clients(round, clients, evaluate_only=True)
        logger.info(f"{len(succ)} client evaluations returned, {len(fails)} failures... {fails}")
        client_metrics = [res.test_metrics for res in succ]
        return self.aggregate_metrics(metrics=client_metrics, metric_names=["loss", "accuracy"])
