#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Switching to project root directory $ROOT_DIR"
cd "$ROOT_DIR"

echo "Build package"
python setup.py bdist_wheel

aws codeartifact login --tool twine \
  --repository "$AWS_CODEARTIFACT_REPOSITORY" \
  --domain "$AWS_CODEARTIFACT_DOMAIN" \
  --domain-owner "$AWS_CODEARTIFACT_DOMAIN_OWNER"

python3 -m twine upload dist/* --repository codeartifact --skip-existing
