#!/usr/bin/env bash
set -e

TIMEOUT=20m
TIMEOUT_UPSTREAM=19m55s

if ! command -v faas-cli &>/dev/null; then
  curl -sL https://cli.openfaas.com | sudo sh
fi

if ! command -v arkade &>/dev/null; then
  curl -SLsf https://get.arkade.dev | sudo sh
fi

arkade install openfaas \
  --set gateway.upstreamTimeout=$TIMEOUT_UPSTREAM \
  --set gateway.writeTimeout=$TIMEOUT \
  --set gateway.readTimeout=$TIMEOUT \
  --set faasnetes.writeTimeout=$TIMEOUT \
  --set faasnetes.readTimeout=$TIMEOUT \
  --set queueWorker.ackWait=$TIMEOUT

# pull template for function deployment
faas-cli template pull
faas-cli template pull https://github.com/openfaas-incubator/python-flask-template

# --- add private repo (optional)
# kubectl create secret docker-registry tum-repo \
#     --docker-server=registry.caps.in.tum.de \
#     --docker-username=caps \
#     --docker-password='<PASSWORD>' \
#     --namespace openfaas-fn
# kubectl patch serviceaccount default -n openfaas-fn -p '{"imagePullSecrets": [{"name": "tum-repo"}]}'