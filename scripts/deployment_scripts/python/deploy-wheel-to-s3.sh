#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

echo "Switching to project root directory $ROOT_DIR"
cd "$ROOT_DIR"

# Build python project
echo "Build package"
.venv/bin/python setup.py bdist_wheel

echo "Copying wheel to local server"
cp "$ROOT_DIR/dist/fedless-0.0.0-py3-none-any.whl" "/datasets"

# Building image
# echo "Build and Push Image"
# aws s3 cp "$ROOT_DIR/dist/fedless-0.0.0-py3-none-any.whl" "s3://fedless/fedless-0.0.0-py3-none-any.whl" \
#   --acl public-read

