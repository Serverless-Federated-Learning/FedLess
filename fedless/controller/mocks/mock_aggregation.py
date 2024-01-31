from fedless.common.models import AggregatorFunctionParams

from fedless.aggregator.aggregation import default_aggregation_handler


class MockAggregator:
    def __init__(
        self,
        params: AggregatorFunctionParams,
        delete_results_after_finish: bool = True,
    ):
        self.session_id = params.session_id
        self.round_id = params.round_id
        self.database = params.database
        self.serializer = params.serializer
        self.test_data = params.test_data
        self.delete_results_after_finish = delete_results_after_finish
        self.aggregation_strategy = params.aggregation_strategy
        self.aggregation_hyper_params = params.aggregation_hyper_params

    def run_aggregator(self):
        return default_aggregation_handler(
            self.session_id,
            self.round_id,
            self.database,
            self.serializer,
            self.test_data,
            self.delete_results_after_finish,
            self.aggregation_strategy,
            self.aggregation_hyper_params,
        )
