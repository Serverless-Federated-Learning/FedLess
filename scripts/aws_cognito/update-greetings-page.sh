#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

aws s3 cp "$SCRIPT_DIR/index.html" "s3://fedless-user-page" --acl public-read
