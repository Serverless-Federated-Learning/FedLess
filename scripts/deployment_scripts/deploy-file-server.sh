#!/usr/bin/env bash

# Deploys the nginx file server on the local cluster

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
K8_FILES_DIR="$ROOT_DIR/scripts/deployment_scripts/kubernetes/data-file-server"

kubectl apply -f "$K8_FILES_DIR"
