kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
K8_FILES_DIR="$ROOT_DIR/scripts/deployment_scripts/kubernetes/kubernetes-dashboard"

kubectl apply -f "$K8_FILES_DIR"

kubectl proxy