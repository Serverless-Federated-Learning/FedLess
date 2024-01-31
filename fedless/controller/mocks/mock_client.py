from fedless.client import ClientError, fedless_mongodb_handler
from fedless.controller.invocation import InvocationError
from fedless.common.models import InvokerParams


class MockClient:
    def __init__(self, config: InvokerParams):
        self.config = config

    async def run_client(self):
        try:

            return fedless_mongodb_handler(
                session_id=self.config.session_id,
                round_id=self.config.round_id,
                client_id=self.config.client_id,
                invocation_id=self.config.invocation_id,
                database=self.config.database,
                evaluate_only=self.config.evaluate_only,
                invocation_delay=self.config.invocation_delay,
            )
        except ClientError as e:
            raise InvocationError(str(e))
