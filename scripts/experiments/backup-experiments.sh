#!/usr/bin/env bash

set -e

bucket_name="fedless-experiment-artifacts"
bucket_region="eu-central-1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
out_directory="$root_directory/out"
save_dir="fedless-experiments-$(git rev-parse --short HEAD)-$(date +%s)"

echo "Switching to project root directory $root_directory"
cd "$root_directory"

echo "Checking if buck $bucket_name exists"
if ! aws s3api head-bucket --bucket "$bucket_name" --region "$bucket_region" &>/dev/null; then
  echo "Bucket $bucket_name does not exist, creating it now..."
  aws s3api create-bucket --bucket "$bucket_name" --region "$bucket_region" \
    --create-bucket-configuration LocationConstraint="$bucket_region"
fi

echo "Zipping directory $out_directory"
zip -r "$save_dir.zip" "$out_directory"

echo "Saving files from $out_directory"
aws s3 cp "$save_dir.zip" "s3://$bucket_name/$save_dir.zip" --region "$bucket_region"

echo "Removing temporary zip file"
rm "$save_dir.zip"
