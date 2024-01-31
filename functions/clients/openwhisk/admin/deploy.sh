#!/usr/bin/env bash

set -e

wsk -i action update \
  create_resource_server \
  handlers.py \
  --main create_resource_server \
  --docker mohamedazab/fedless-openwhisk:clients \
  --memory 1024 \
  --timeout 60000 \
  --web raw \
  --web-secure false \
  --param region "$COGNITO_USER_POOL_REGION" \
  --param userpool_id "$COGNITO_USER_POOL_ID" \
  --param expected_client_id "$COGNITO_SERVER_CLIENT_ID" \
  --param aws_access_key_id "$FEDLESS_ADMIN_AWS_ACCESS_KEY_ID" \
  --param aws_access_key_secret "$FEDLESS_ADMIN_AWS_ACCESS_KEY_SECRET" \
  --param invoker_client_id "$COGNITO_INVOKER_CLIENT_ID"


# Print info if deployed successfully
wsk -i action get create_resource_server

# Print url to invoke function
API_HOST=$(wsk -i property get --apihost -o raw)
NAMESPACE=$(wsk -i property get --namespace -o raw)
ENDPOINT="https://$API_HOST/api/v1/web/$NAMESPACE/default/create_resource_server.json"
echo "To invoke function, run:"
echo "curl -X POST -k $ENDPOINT -H \"Authorization: Bearer xxx\""