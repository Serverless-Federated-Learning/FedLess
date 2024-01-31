from fedless.common.models.models import ClientConfig, MongodbConnectionConfig
from fedless.common.persistence.client_daos import InvocationHistoryDao
from fedless.controller.strategies.client_selection import ClientSelectionScheme
from typing import List
import random


class RandomClientSelection(ClientSelectionScheme):
    def __init__(self):
        super().__init__(self.sample_clients)

    def sample_clients(self, clients: int, pool: List, round, max_rounds) -> List:
        return random.sample(pool, k=clients)


# Async Random selection needs to prevent to reselect busy clients
class AsyncRandomClientSelection(ClientSelectionScheme):
    def __init__(self, session_id: str, db_config: MongodbConnectionConfig):
        super().__init__(self.sample_clients)
        self.session = session_id
        self.invocation_history_dao: InvocationHistoryDao = InvocationHistoryDao(
            db=db_config
        )

    def sample_clients(
        self, clients: int, pool: List[ClientConfig], round, max_rounds
    ) -> List:
        exec_timeout = 20 * 60
        busy_client_ids = set(
            self.invocation_history_dao.get_busy_inv_client_ids(self.session, exec_timeout)
        )
        available_pool = list(
            filter(lambda c: not c.client_id in busy_client_ids, pool)
        )
        return random.sample(available_pool, k=clients)
