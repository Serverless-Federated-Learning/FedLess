import csv
import logging
import math
import os
from typing import List
import numpy as np
import pandas as pd
from pathlib import Path

import random
from fedless.common.models.models import ClientScore
from fedless.common.persistence import client_daos
from fedless.common.persistence.client_daos import ClientConfigDao, InvocationHistoryDao

from fedless.controller.strategies.client_selection.client_selection_scheme import (
    ClientSelectionScheme,
)
from fedless.common.models import MongodbConnectionConfig
from fedless.common.persistence import ClientScoreDao

logger = logging.getLogger(__name__)


class ScoreBasedClientSelection(ClientSelectionScheme):
    def __init__(
        self,
        buffer_ratio: float,
        session_id: str,
        db_config: MongodbConnectionConfig,
        adjustment_rate: float = 0.2,
        penalty_rounds: int = 5,
        boost_clients: bool = True,
        partial_random_selection: bool = False,
        log_dir=Path.cwd(),
    ):
        super().__init__(self.sample_clients)

        # exponential decay older scores
        # exponential promote not select clients until selected
        self.adjustment_rate = adjustment_rate if adjustment_rate < 1.0 else 0.2
        self.decay_rate = 1.0 - self.adjustment_rate
        self.promotion_rate = 1.0 + self.adjustment_rate

        self.penalty_rounds = penalty_rounds
        self.boost_clients = boost_clients
        self.partial_random_selection = partial_random_selection

        # if sync only = 1
        self.buffer_ratio = buffer_ratio  # size of clients to wait til aggregate
        self.session_id = session_id
        self.db_config = db_config
        self.log_dir = log_dir
        self.client_score_dao: ClientScoreDao = ClientScoreDao(db=db_config)
        self.invocation_history_dao: InvocationHistoryDao = InvocationHistoryDao(
            db=db_config
        )

    def update_booster_value(
        self,
        selected_ids: List[str],
        available_client_ids: set,
        promotion_rate: float = None,
    ):
        if promotion_rate is None:
            promotion_rate = self.promotion_rate

        # promote the inactive clients to have higher prob being selected
        # by gradually increasing its' scores
        # and reset selected clients to 1
        self.client_score_dao.reset_booster_value(selected_ids)

        not_selected_ids = list(available_client_ids - set(selected_ids))
        self.client_score_dao.update_booster_value(not_selected_ids, promotion_rate)

        # -- maybe boost clients while cooling down -> try to prevent from full cool down
        # Almost cool down, invoke with higher prob

    def calculate_weighted_average_score(
        self, client: ClientScore, k: int = 5
    ) -> float:
        def calculate_client_score(
            epoch, number_of_examples, batch_size, training_time
        ) -> float:
            # MAIN FACTOR
            # assumption: raw data size same (resolution), as the input size for the models are static

            # EXTRA FACTOR
            # diversity of local dataset
            # labeled data -> number of classes and distribution
            # hard for unlabeled data (language, audio, etc.) -> variance?

            # Data size weight (data ratio) * processing time weight
            # (number_of_examples / total_examples) * (epoch * number_of_examples / batch_size) / training_time
            # => total_examples is a static value that applies globally, and doesn't have effect on the score distribution
            return (
                number_of_examples
                * (epoch * number_of_examples / batch_size)
                / training_time
            )

        # --- other options -> moving average:
        # Weighted moving average (5,4,3, ...)
        # Exponential Moving Average (0.9^0, 0.9^1, 0.9^2, ...)
        # -
        # without storing the complete history,
        # and just updating the average with newest value

        decay_rate = self.decay_rate

        training_times = client.training_times
        nr_times = len(training_times)
        k = nr_times if nr_times < k else k

        if nr_times == 0:
            return 0

        last_k_scores = training_times[-k:]

        weighted_sum = sum(
            [
                (decay_rate**idx)
                * calculate_client_score(
                    client.epochs, client.cardinality, client.batch_size, training_time
                )
                for idx, training_time in enumerate(reversed(last_k_scores))
            ]
        )
        total_weight = sum([decay_rate**i for i in range(k)])

        # booster value magnifies the score gradually until selected (reset to 1)
        return client.booster_value * (weighted_sum / total_weight)

    def save_results_to_csv(self) -> None:
        client_score_list = self.client_score_dao._collection.find(
            {"session_id": self.session_id}, {"_id": False}
        )

        keys = ClientScore.__fields__.keys()

        with open(self.log_dir / f"client_scores_{self.session_id}.csv", "w") as file:
            csv_writer = csv.DictWriter(file, fieldnames=keys, restval="N/A")
            csv_writer.writeheader()
            csv_writer.writerows(client_score_list)

    def sample_clients(
        self, clients_in_round: int, clients: List, round, max_rounds
    ) -> List:
        score_dao = self.client_score_dao
        inv_dao = self.invocation_history_dao

        # exclude clients from selection for given rounds
        failed_client_id = set(
            self.invocation_history_dao.get_last_failed_clients(
                self.session_id, round - self.penalty_rounds
            )
        )

        all_client_id = (
            set(map(lambda client: client.client_id, clients)) - failed_client_id
        )
        all_invoked_client_ids = set(inv_dao.get_distinct_inv_ids(self.session_id))

        # get available/idle clients, and busy clients
        busy_client_ids = set(inv_dao.get_busy_inv_client_ids(self.session_id))
        available_client_ids = all_client_id - busy_client_ids

        # if available clients less than required, select all
        if clients_in_round > len(available_client_ids):
            logger.warning(
                f"Insufficient clients available: {len(available_client_ids)}/{clients_in_round} (select all available clients)"
            )
            return list(filter(lambda c: c.client_id in available_client_ids, clients))

        invoked_client_ids = available_client_ids.intersection(all_invoked_client_ids)
        uninvoked_client_ids = available_client_ids - all_invoked_client_ids

        client_selection = []
        total_uninvoked_clients = len(uninvoked_client_ids)

        # fresh clients always have priority
        # if not sufficent -> sample by score/random
        if total_uninvoked_clients >= clients_in_round:
            client_selection = random.sample(uninvoked_client_ids, clients_in_round)
        else:
            client_selection = list(uninvoked_client_ids)
            missing_amount = clients_in_round - total_uninvoked_clients

            # select random partially
            if self.partial_random_selection:
                sample_from_score = int(math.ceil(missing_amount * self.buffer_ratio))
                sample_random = missing_amount - sample_from_score
            else:
                sample_from_score = missing_amount
                sample_random = 0

            all_client_stats = list(score_dao.load_all())

            client_stats: List[ClientScore] = list(
                filter(
                    lambda stats: stats.client_id in invoked_client_ids,
                    all_client_stats,
                )
            )

            ids = []
            scores = []
            for stats in client_stats:
                score = self.calculate_weighted_average_score(stats)
                ids.append(stats.client_id)
                scores.append(score)

            prob = np.array(scores) / sum(scores)

            # score base selection for buffer size
            sample_from_score = (
                sample_from_score if len(ids) >= sample_from_score else len(ids)
            )

            client_selection += np.random.choice(
                ids, size=sample_from_score, p=prob, replace=False
            ).tolist()

            # random selection for rest
            client_selection += random.sample(
                available_client_ids - set(client_selection), k=sample_random
            )

            # promote not selected clients, and reset selected clients
            if self.boost_clients:
                self.update_booster_value(client_selection, available_client_ids)

        # save scores to csv
        self.save_results_to_csv()

        return list(filter(lambda c: c.client_id in client_selection, clients))
