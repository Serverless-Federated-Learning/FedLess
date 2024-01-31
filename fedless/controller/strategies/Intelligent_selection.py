# from functools import reduce
import logging
import operator as op
import os
import random
from abc import ABC
from collections import defaultdict
from itertools import cycle, groupby, islice
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from fedless.common.models import ClientPersistentHistory, MongodbConnectionConfig
from fedless.common.persistence.client_daos import ClientHistoryDao, ClientModelDao

logger = logging.getLogger(__name__)


class ClientSelectionScheme(ABC):
    def __init__(self, execution_func, function_params={}):
        """
        execution_func: function to execute the selection procedure
        function_params: parameters for the function to be called
        """
        self.excution_func = execution_func
        self.function_params = function_params

    def select_clients(self, n_clients_in_round: int, clients_pool: List, round: int, max_rounds: int) -> List:
        """
        :clients: number of clients
        :pool: List[ClientConfig] contains each client necessary info for selection
        :func: function to order the clients
        :return: list of clients selected from the pool based on selection criteria
        """
        return self.excution_func(n_clients_in_round, clients_pool, round, max_rounds, **self.function_params)


class RandomClientSelection(ClientSelectionScheme):
    def __init__(self):
        super().__init__(self.sample_clients)

    def sample_clients(self, clients: int, pool: List, round: int, max_rounds: int) -> List:
        return random.sample(pool, clients)


class DBScanClientSelection(ClientSelectionScheme):
    def __init__(self, db_config: MongodbConnectionConfig, session, log_dir=Path.cwd(), execution_func=None):
        if execution_func is None:
            super().__init__(self.db_fit)
        else:
            super().__init__(execution_func)
        self.db_config = db_config
        self.session = session
        self.log_dir = log_dir
        self.start_clustering_round = -1

    def compute_ema(
        self,
        training_times: list,
        latest_ema: float,
        latest_updated: int,
        smoothingFactor: float = 0.5,
    ):
        """
        Parameters
        ----------
        training_times : list
            The name of the animal
        latest_ema : float
            The last ema computation
        latest_update_idx : int, optional
            the last idx for ema computed before
        """
        updated_ema = latest_ema
        for i in range(latest_updated + 1, len(training_times)):
            updated_ema = updated_ema * smoothingFactor + (1 - smoothingFactor) * training_times[i]
        return updated_ema

    def get_client_ema(self, times_list: List, alpha: float = 0.5) -> float:

        if len(times_list) == 0:
            return 0
        ema = 0
        i = 1
        ema = times_list[0]
        while i < len(times_list):
            ema = ema * (1 - alpha) + alpha * times_list[i]
            i += 1
        # return (random.sample([50,80,120,140,160],1))[0]
        return ema

    def get_missed_rounds_ema(self, times_list: List, round: int, max_training_time: float) -> float:
        # to match penalties we increase values in the list +1 to avoid zero at the beginnning and have 100 penalty for missing round
        times_one_based = np.divide(np.array(times_list) + 1, round)
        # higher alpha for marger penalty
        # 1.5 is th
        return self.get_client_ema(times_one_based, alpha=0.8) * 1.5 * max_training_time

    def sort_clusters(self, clients: List[ClientPersistentHistory], labels: list, round: int):
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # dict of {cluster_idx: (ema,[client_list])}
        cluster_number_map = {}

        # get the largest possible training time this is computed everytime in the clustering
        # so its effect is in this round only
        max_training_time = max([max(client.training_times, default=0) for client in clients], default=0)

        for idx in range(len(labels)):
            client_cluster_idx = labels[idx]
            # client_ema = clients[idx].ema
            client_ema = self.get_client_ema(clients[idx].training_times)
            total_ema = client_ema

            client_penalty = self.get_missed_rounds_ema(clients[idx].missed_rounds, round, max_training_time)
            # each missed round have a penalty where the penalty percentage is max_training_time* how fresh miss is
            # avoid div by zero in round
            # client penalty is ema on the defiend penalty

            # add penalty to training time
            total_ema += client_penalty

            if client_cluster_idx in cluster_number_map:
                old_cluster_data = cluster_number_map[client_cluster_idx]
                # append new client
                old_cluster_data[1].append(clients[idx])
                cluster_number_map[client_cluster_idx] = (
                    old_cluster_data[0] + total_ema,
                    old_cluster_data[1],
                )
            else:
                cluster_number_map[client_cluster_idx] = (total_ema, [clients[idx]])
        # sort clusters based on avg ema per cluster
        # we didnt sort with the missed rounds because fast clients will probably not miss alot of rounds

        # normalize by no of clients/cluster
        for (
            cluster_idx,
            (
                cluster_total_ema,
                cluster_clients,
            ),
        ) in cluster_number_map.items():
            cluster_number_map[cluster_idx] = (
                cluster_total_ema / len(cluster_clients),
                cluster_clients,
            )
        return dict(sorted(cluster_number_map.items(), key=lambda x: x[1][0]))
        # pass

    def save_clustering_count(self, round: int, **kwargs) -> None:

        preps_dict = {"session_id": self.session, "round_id": round, **kwargs}
        add_headers = not os.path.isfile(self.log_dir / f"clusters_{self.session}.csv")

        # cluster num for each round
        pd.DataFrame.from_records([preps_dict]).to_csv(
            self.log_dir / f"clusters_{self.session}.csv",
            mode="a",
            index=False,
            header=add_headers,
        )
        # # cluster:avg_ema
        # pd.DataFrame.from_records([preps_dict]).to_csv(
        #     self.log_dir / f"clusters_details_{session}.csv",mode='a',index=False,header=False
        # )

    def save_clustering_details(self, round: int, sorted_clusters, selected_clients, max_training_time) -> None:

        cluster_dict_list = []
        for (
            cluster_idx,
            (
                cluster_ema,
                cluster_client_configs,
            ),
        ) in sorted_clusters.items():

            for client_config in cluster_client_configs:
                # round is never zero because clustering works after first round
                client_penalty = self.get_missed_rounds_ema(client_config.missed_rounds, round, max_training_time)
                cluster_dict_list += [
                    {
                        "round_id": round,
                        "cluster_ema": cluster_ema,
                        "cluster_id": cluster_idx,
                        "client_id": client_config.client_id,
                        "client_ema": self.get_client_ema(client_config.training_times),
                        "penalty": client_penalty,
                        "backoff": client_config.client_backoff,
                        "missed_rounds": len(client_config.missed_rounds),
                    }
                ]

        selected_clients_ids = []
        for client_config in selected_clients:
            selected_clients_ids += [{"round_id": round, "client_id": client_config.client_id}]

        # add headers for first write only
        add_headers = not os.path.isfile(self.log_dir / f"clusters_details_{self.session}.csv")
        # cluster num for each round
        pd.DataFrame.from_records(cluster_dict_list).to_csv(
            self.log_dir / f"clusters_details_{self.session}.csv",
            mode="a",
            index=False,
            header=add_headers,
        )

        pd.DataFrame.from_records(selected_clients_ids).to_csv(
            self.log_dir / f"selected_details_{self.session}.csv",
            mode="a",
            index=False,
            header=add_headers,
        )

    def filter_rookies(
        self, clients: List[ClientPersistentHistory], round: int
    ) -> Tuple[List[ClientPersistentHistory], List[ClientPersistentHistory]]:
        rookies = []
        rest_clients = []
        stragglers = []
        for client in clients:
            if len(client.training_times) == 0 and len(client.missed_rounds) == 0:
                # if client.latest_updated == -1:
                rookies.append(client)
            elif len(client.missed_rounds) > 0 and client.client_backoff + client.missed_rounds[-1] >= round:
                stragglers.append(client)
            else:
                rest_clients.append(client)
        return rookies, rest_clients, stragglers

    def db_fit(self, n_clients_in_round: int, pool: List, round, max_rounds) -> list:
        history_dao = ClientHistoryDao(db=self.db_config)
        # all data of the session
        all_data = list(history_dao.load_all(session_id=self.session))

        # try and run rookies first
        rookie_clients, rest_clients, stragglers = self.filter_rookies(all_data, round)
        # use the list t o get separate the clients
        # rest_clients = all_data
        # rookie_clients = []
        # todo update latest updated for rookies and stragglers selected

        if len(rookie_clients) >= n_clients_in_round:
            logger.info(f"selected rookies {n_clients_in_round} of {len(rookie_clients)}")
            return self.select_candidates_from_pool(random.sample(rookie_clients, n_clients_in_round), pool)

        # clients selected from the clustering
        n_clients_from_clustering = min(n_clients_in_round - len(rookie_clients), len(rest_clients))
        if n_clients_from_clustering > 0 and self.start_clustering_round == -1:
            self.start_clustering_round = round
        # clients selected from the stragglers which miss rounds alot in the backoff sequence
        n_from_stragglers = n_clients_in_round - (n_clients_from_clustering + len(rookie_clients))

        # sample stragglers

        round_stragglers = random.sample(stragglers, n_from_stragglers)

        logger.info(f"selected rookies {len(rookie_clients)}, remaining {n_clients_from_clustering}")
        # update the latest updated for rookies
        # todo: filter the clients with non fulfilled backoffs and use them iff the rest does not complete the required number
        max_training_time = max(
            [max(client.training_times, default=0) for client in rest_clients],
            default=0,
        )
        training_data = []
        for client_data in rest_clients:
            client_training_times = client_data.training_times
            client_missed_rounds = client_data.missed_rounds

            ema = self.get_client_ema(client_training_times)
            # todo div by cardinality if exist else assume 1 so the number will be bigger and it will get demoted
            # ema of missed rounds so later rounds have higher penalty factor
            missed_rounds_ema = self.get_missed_rounds_ema(client_missed_rounds, round, max_training_time)
            client_data.latest_updated = round
            history_dao.save(client_data)
            training_data.append([ema, missed_rounds_ema])
        # use last update with backoff

        # todo convert to mins
        labels, best_score, best_eps = self.perform_clustering(training_data=training_data, eps_step=0.1)
        sorted_clusters = self.sort_clusters(rest_clients, labels, round)
        # logs
        self.save_clustering_count(
            round,
            **{
                "num_clusters": len(sorted_clusters),
                "score": best_score,
                "eps": best_eps,
            },
        )
        ######

        cluster_idx_list = np.arange(start=0, stop=len(sorted_clusters))
        # actuall running perc without the first rookie rounds
        perc = ((round - self.start_clustering_round) / max(max_rounds - self.start_clustering_round, 1)) * 100
        start_cluster_idx = np.percentile(cluster_idx_list, perc, interpolation="lower")

        round_candidates_history = (
            rookie_clients
            + self.sample_starting_from(sorted_clusters, start_cluster_idx, n_clients_from_clustering)
            + round_stragglers
        )
        # todo in rest clients or all of them?

        # save logs
        # save 2 files
        # 1- round -> client to cluster
        # 2- round -> selected client_id list
        self.save_clustering_details(round, sorted_clusters, round_candidates_history, max_training_time)

        return self.select_candidates_from_pool(round_candidates_history, pool)

    def select_candidates_from_pool(self, round_candidates_history: list, pool):
        round_candidates_ids = list(map(lambda x: x.client_id, round_candidates_history))
        round_candidates = filter(lambda x: x.client_id in round_candidates_ids, pool)
        return list(round_candidates)

    def perform_clustering(self, training_data, eps_step):
        best_labels = None
        best_score = 0
        best_eps = 0
        X = StandardScaler().fit_transform(training_data)
        # we train on training time in mins
        # distance should be time in mins
        # we try distances of upt to 2 mins apart
        for eps in np.arange(0.01, 2, eps_step):
            logger.info(f"trying eps in range 0.01-{eps} with step {eps_step}")
            db = DBSCAN(eps=eps, min_samples=2).fit(X)
            labels = db.labels_

            if best_labels is None:
                best_labels = labels
                best_eps = eps
            # Number of clusters in labels, ignoring noise if present.
            n_lables = len(set(labels))
            # n_clusters_ = n_lables - (1 if -1 in labels else 0)
            # if n_clusters_ == 1:
            #     logger.info("stopping, samples are all in one cluster")
            #     break
            n_noise_ = list(labels).count(-1)
            if n_lables <= len(X) - 1 and n_lables > 1:
                clustering_score = metrics.calinski_harabasz_score(X, labels)
                logger.info(f"clustering score : {clustering_score}")
                if clustering_score > best_score:
                    best_score = clustering_score
                    best_labels = labels
                    best_eps = eps
                    logger.info(
                        f"updated clustering score:{clustering_score}, n_labels = {n_lables}, n_noise = {n_noise_}"
                    )
            else:
                logger.info(f"number of clusters not enough , labels = {n_lables}, noise = {n_noise_}")
        return best_labels, best_score, best_eps

    def sample_starting_from(self, sorted_clusters, start_cluster_idx: int, n_clients_from_clustering: int) -> list:
        # return clients which run the least
        cluster_list = list(sorted_clusters.items())
        returned_samples = []
        while n_clients_from_clustering > 0:
            cluster = cluster_list[start_cluster_idx]
            cluster_clients = cluster[1][1]
            cluster_size = len(cluster_clients)
            if cluster_size >= n_clients_from_clustering:
                cluster_clients_sorted = sorted(
                    cluster_clients,
                    key=lambda client: len(client.training_times) + len(client.missed_rounds),
                )
                returned_samples += cluster_clients_sorted[:n_clients_from_clustering]
                n_clients_from_clustering = 0
            else:
                n_clients_from_clustering -= cluster_size
                returned_samples += cluster_clients
            # if clusters are done go back and fetch from the faster clients
            start_cluster_idx = (start_cluster_idx + 1) % len(cluster_list)

        return returned_samples
        # return random.sample(pool, n_clients_in_round)

        pass


class DBScanModelClientSelection(DBScanClientSelection):
    def __init__(self, db_config: MongodbConnectionConfig, session, log_dir=Path.cwd()):
        super().__init__(db_config, session, log_dir, self.db_fit)
        self.start_clustering_round = -1

    def roundrobin(self, *iterables):
        # took from here https://docs.python.org/3/library/itertools.html#itertools-recipes

        # "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
        # Recipe credited to George Sakkis

        pending = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while pending:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                pending -= 1
                nexts = cycle(islice(nexts, pending))

    def db_fit(self, n_clients_in_round: int, pool: List, round, max_rounds) -> list:
        history_dao = ClientHistoryDao(db=self.db_config)
        client_models_dao = ClientModelDao(db=self.db_config)

        # all data of the session
        all_data = list(history_dao.load_all(session_id=self.session))
        model_client_mapping = client_models_dao.get_model_client_mapping(session_id=self.session)

        assert n_clients_in_round >= len(
            model_client_mapping.keys()
        ), "Clients in each round must atleast be equal to number of distinct model types"
        rookie_model_clients = {}
        rest_model_clients = {}
        stragglers_model_clients = {}
        all_data_model_clients = {}
        total_rookie_clients = 0
        total_rest_clients = 0
        total_stragglers = 0

        for model_type, clients in model_client_mapping.items():
            # Filtering the client history all_data for each model type
            all_data_model_clients[model_type] = [
                client_history for client_history in all_data if client_history.client_id in clients
            ]

            # try and run rookies first
            (
                rookie_model_clients[model_type],
                rest_model_clients[model_type],
                stragglers_model_clients[model_type],
            ) = self.filter_rookies(all_data_model_clients[model_type], round)
            total_rookie_clients += len(rookie_model_clients[model_type])
            total_rest_clients += len(rest_model_clients[model_type])
            total_stragglers += len(stragglers_model_clients[model_type])
            # use the list t o get separate the clients
            # rest_clients = all_data
            # rookie_clients = []
            # todo update latest updated for rookies and stragglers selected

        round_robin_rookie_selections = list(self.roundrobin(*rookie_model_clients.values()))

        if total_rookie_clients >= n_clients_in_round:
            # Start picking clients up for each model type using round robin
            logger.info(f"selected rookies {n_clients_in_round} of {total_rookie_clients}")
            return self.select_candidates_from_pool(round_robin_rookie_selections[:n_clients_in_round], pool)

        # clients selected from the clustering
        n_clients_from_clustering = min(n_clients_in_round - total_rookie_clients, total_rest_clients)
        if n_clients_from_clustering > 0 and self.start_clustering_round == -1:
            self.start_clustering_round = round
        # clients selected from the stragglers which miss rounds alot in the backoff sequence
        n_from_stragglers = n_clients_in_round - (n_clients_from_clustering + total_rookie_clients)

        # sample stragglers, round robin for stragglers
        # round_stragglers = random.sample(stragglers, n_from_stragglers)
        round_robin_stragglers = list(self.roundrobin(*stragglers_model_clients.values()))[:n_from_stragglers]

        logger.info(f"selected rookies {total_rookie_clients}, remaining {n_clients_from_clustering}")
        # update the latest updated for rookies
        # todo: filter the clients with non fulfilled backoffs and use them iff the rest does not complete the required number

        model_clusters = {}
        for model_type in model_client_mapping.keys():
            max_training_time = max(
                [max(client.training_times, default=0) for client in rest_model_clients[model_type]],
                default=0,
            )
            training_data = []
            for client_data in rest_model_clients[model_type]:
                client_training_times = client_data.training_times
                client_missed_rounds = client_data.missed_rounds

                ema = self.get_client_ema(client_training_times)
                # todo div by cardinality if exist else assume 1 so the number will be bigger and it will get demoted
                # ema of missed rounds so later rounds have higher penalty factor
                missed_rounds_ema = self.get_missed_rounds_ema(client_missed_rounds, round, max_training_time)
                client_data.latest_updated = round
                history_dao.save(client_data)
                training_data.append([ema, missed_rounds_ema])
            # use last update with backoff

            # todo convert to mins
            labels, best_score, best_eps = self.perform_clustering(training_data=training_data, eps_step=0.1)
            sorted_clusters = self.sort_clusters(rest_model_clients[model_type], labels, round)
            # logs
            self.save_clustering_count(
                round,
                **{
                    "num_clusters": len(sorted_clusters),
                    "score": best_score,
                    "eps": best_eps,
                    "model_type": model_type,
                },
            )
            ######

            cluster_idx_list = np.arange(start=0, stop=len(sorted_clusters))
            # actuall running perc without the first rookie rounds
            perc = ((round - self.start_clustering_round) / max(max_rounds - self.start_clustering_round, 1)) * 100
            start_cluster_idx = np.percentile(cluster_idx_list, perc, interpolation="lower")

            model_clusters[model_type] = self.sample_starting_from(
                sorted_clusters, start_cluster_idx, len(rest_model_clients[model_type])
            )
            self.save_clustering_details(round, sorted_clusters, max_training_time)

        round_robin_cluster_selections = list(self.roundrobin(*model_clusters.values()))[:n_clients_from_clustering]

        round_candidates_history = (
            round_robin_rookie_selections + round_robin_cluster_selections + round_robin_stragglers
        )
        # todo in rest clients or all of them?

        # save logs
        # save 2 files
        # 1- round -> client to cluster
        # 2- round -> selected client_id list

        return self.select_candidates_from_pool(round_candidates_history, pool)

    def save_clustering_details(self, round: int, sorted_clusters, max_training_time) -> None:

        cluster_dict_list = []
        for (
            cluster_idx,
            (
                cluster_ema,
                cluster_client_configs,
            ),
        ) in sorted_clusters.items():

            for client_config in cluster_client_configs:
                # round is never zero because clustering works after first round
                client_penalty = self.get_missed_rounds_ema(client_config.missed_rounds, round, max_training_time)
                cluster_dict_list += [
                    {
                        "round_id": round,
                        "cluster_ema": cluster_ema,
                        "cluster_id": cluster_idx,
                        "client_id": client_config.client_id,
                        "client_ema": self.get_client_ema(client_config.training_times),
                        "penalty": client_penalty,
                        "backoff": client_config.client_backoff,
                        "missed_rounds": len(client_config.missed_rounds),
                    }
                ]

        # selected_clients_ids = []
        # for client_config in selected_clients:
        #     selected_clients_ids += [{"round_id": round, "client_id": client_config.client_id}]

        # add headers for first write only
        add_headers = not os.path.isfile(self.log_dir / f"clusters_details_{self.session}.csv")
        # cluster num for each round
        pd.DataFrame.from_records(cluster_dict_list).to_csv(
            self.log_dir / f"clusters_details_{self.session}.csv",
            mode="a",
            index=False,
            header=add_headers,
        )

        # pd.DataFrame.from_records(selected_clients_ids).to_csv(
        #     self.log_dir / f"selected_details_{self.session}.csv",
        #     mode="a",
        #     index=False,
        #     header=add_headers,
        # )
