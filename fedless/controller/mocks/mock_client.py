from fedless.client import ClientError, master_handler
from fedless.common.models import InvokerParams
from fedless.controller.invocation import InvocationError


class MockClient:
    def __init__(self, config: InvokerParams):
        self.config = config

    async def run_client(self):
        try:

            return master_handler(
                session_id=self.config.session_id,
                round_id=self.config.round_id,
                client_id=self.config.client_id,
                database=self.config.database,
                evaluate_only=self.config.evaluate_only,
                invocation_delay=self.config.invocation_delay,
                action=self.config.action,
                algorithm=self.config.algorithm,
            )

        except ClientError as e:
            raise InvocationError(str(e))
