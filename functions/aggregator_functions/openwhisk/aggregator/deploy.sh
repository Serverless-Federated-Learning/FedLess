#!/usr/bin/env bash

set -e

wsk -i action update \
  aggregator \
  main.py \
  --docker mohamedazab/fedless-openwhisk:clients \
  --memory 1024 \
  --timeout 60000 \
  --web raw \
  --web-secure false
  #--concurrency 1

# Print info if deployed successfully
wsk -i action get aggregator

# Print url to invoke function
API_HOST=$(wsk -i  property get --apihost -o raw)
NAMESPACE=$(wsk -i  property get --namespace -o raw)
ENDPOINT="https://$API_HOST/api/v1/web/$NAMESPACE/default/aggregator.json"
echo "To invoke function, run:"
echo "curl -X POST -k $ENDPOINT"
