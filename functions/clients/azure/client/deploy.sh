#!/usr/bin/env bash
set -e

for i in {1..5}; do
  location="germanywestcentral"
  resource_group_name="fedlessfl$i"
  app_name="fedless-fl-client-$i"
  storage_name="storage$resource_group_name"

  az group create \
    --name "$resource_group_name" \
    --location "$location"
  az storage account create --name "$storage_name" \
    --location "$location" \
    --resource-group "$resource_group_name" \
    --sku Standard_LRS
  az functionapp create \
    --resource-group "$resource_group_name" \
    --consumption-plan-location "$location" \
    --runtime python \
    --runtime-version 3.8 \
    --functions-version 3 \
    --name "$app_name" \
    --storage-account "$storage_name" \
    --os-type linux
  sleep 20
  func azure functionapp publish "$app_name" --python
  az functionapp config appsettings set --name "$app_name" \
    --resource-group "$resource_group_name" \
    --settings TF_ENABLE_ONEDNN_OPTS=1 COGNITO_USER_POOL_REGION="$COGNITO_USER_POOL_REGION" COGNITO_USER_POOL_ID="$COGNITO_USER_POOL_ID" COGNITO_INVOKER_CLIENT_ID="$COGNITO_INVOKER_CLIENT_ID" COGNITO_REQUIRED_SCOPE="client-functions/invoke"
done
