#!/usr/bin/env bash

set -e

# Generate random token
token=$(openssl rand -base64 30)
region="eu-west-1"
userpool_id="$COGNITO_USER_POOL_ID"
expected_client_id="$COGNITO_INVOKER_CLIENT_ID"
required_scope="client-functions/invoke"

wsk -i action update \
  authorizer \
  authorizer.py \
  --docker mohamedazab/fedless-openwhisk:clients \
  --memory 256 \
  --timeout 2000 \
  --web raw \
  --param region "$region" \
  --param userpool_id "$userpool_id" \
  --param expected_client_id "$expected_client_id" \
  --param required_scope "$required_scope"

wsk -i action update \
  client \
  main.py \
  --docker mohamedazab/fedless-openwhisk \
  --memory 2048 \
  --timeout 60000

wsk -i action update \
  client-secure \
  --sequence authorizer,client \
  --web raw \
  --web-secure "$token"

# Print info if deployed successfully
wsk -i action get client-secure

# Print url to invoke function
API_HOST=$(wsk -i property get --apihost -o raw)
NAMESPACE=$(wsk -i property get --namespace -o raw)
ENDPOINT="https://$API_HOST/api/v1/web/$NAMESPACE/default/client-secure.json"
echo "To invoke function, run:"
echo "curl -X POST -H \"X-Require-Whisk-Auth: $token\" -k $ENDPOINT -H \"Authorization: Bearer xxx\""
