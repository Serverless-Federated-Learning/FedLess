#!/usr/bin/env bash

# the node needs to have the label: databases=mongo
# Deploys the parameter server (mongodb database) on the local cluster or upgrades it if it already exists

DB_USERNAME="${FEDLESS_MONGODB_USERNAME?"Need to set FEDLESS_MONGODB_USERNAME"}"
DB_PASSWORD="${FEDLESS_MONGODB_PASSWORD?"Need to set FEDLESS_MONGODB_PASSWORD"}"
PORT="${FEDLESS_MONGODB_PORT?"Need to set FEDLESS_MONGODB_PORT"}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
HELM_CHART_DIR="$ROOT_DIR/scripts/deployment_scripts/kubernetes/parameter-server"

# Install helm if necessary
if ! command -v helm &>/dev/null; then
  echo "Helm installation not found, installing it now"
  curl https://baltocdn.com/helm/signing.asc | sudo apt-key add -
  sudo apt-get install apt-transport-https --yes
  echo "deb https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
  sudo apt-get update
  sudo apt-get install helm --yes
fi


if [ -d "$HELM_CHART_DIR" ]; then
  ### Take action if $DIR exists ###
  echo "Installing helm chart ${HELM_CHART_DIR}..."
  helm install parameter-server \
    "$HELM_CHART_DIR" \
    --set secrets.mongodb_username="$DB_USERNAME" \
    --set secrets.mongodb_password="$DB_PASSWORD" \
    --set service.port="$PORT" \
    # --set nodeSelector.databases=mongo \
  # shellcheck disable=SC2181
  if [ $? -eq 0 ]; then
    echo "Parameter Server was successfully deployed"
  else
    echo "Parameter Server was not installed correctly. Trying to cleanup..."
    helm uninstall parameter-server
  fi

else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "Error: ${DIR} not found. Can not continue."
  exit 1
fi
