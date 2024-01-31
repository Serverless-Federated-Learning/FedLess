#!/usr/bin/env bash


for i in {1..200}; do
  function_name="http-indep-${i}"
  echo "Deploying function $function_name"
  # shellcheck disable=SC2140
  gcloud beta functions deploy "$function_name" \
    --gen2 \
    --runtime python38 \
    --trigger-http \
    --entry-point="http" \
    --allow-unauthenticated \
    --memory=2048MB \
    --timeout=540s \
    --region europe-west4 \
    --max-instances 50 &
    # --set-env-vars TF_ENABLE_ONEDNN_OPTS=1 \
    # --set-build-env-vars GIT_COMMIT_IDENTIFIER="@$COMMIT_HASH",GITHUB_AUTH_TOKEN="$GITHUB_AUTH_TOKEN" &
  if [ $((i % 25)) -eq 0 ]; then
    echo "Waiting for previous functions to finish deployment"
    wait
  fi
done
wait
