from abc import ABC
from typing import List


class ClientSelectionScheme(ABC):
    def __init__(self, execution_func, function_params={}):
        """
        execution_func: function to execute the selection procedure
        function_params: parameters for the function to be called
        """
        self.excution_func = execution_func
        self.function_params = function_params

    def select_clients(
        self, n_clients_in_round: int, clients_pool: List, round, max_rounds
    ) -> List:
        """
        :clients: number of clients
        :pool: List[ClientConfig] contains each client necessary info for selection
        :func: function to order the clients
        :return: list of clients selected from the pool based on selection criteria
        """
        return self.excution_func(
            n_clients_in_round, clients_pool, round, max_rounds, **self.function_params
        )
