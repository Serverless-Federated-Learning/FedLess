#!/usr/bin/env bash
set -e

REGION=eu-west-1
USER_POOL_NAME="fedless-pool-1-test"

# Assumes a fully configured aws cli with sufficient privileges given to IAM user/ role

# Check if user pool by this name already exists
pools_with_name=$(aws cognito-idp list-user-pools \
  --region "$REGION" \
  --max-results 40 |
  jq ".UserPools[] | select(.Name == \"$USER_POOL_NAME\")")

if [ -n "$pools_with_name" ]; then
  echo "Pool with name $USER_POOL_NAME already exists in region $REGION, aborting..."
  exit 1
fi

echo "Pool does not already exist, creating it now..."
response=$(aws cognito-idp create-user-pool \
  --pool-name "$USER_POOL_NAME" \
  --cli-input-yaml "$(cat cognito-user-pool-skeleton.yml)" \
  --region="$REGION")
user_pool_id=$(echo "$response" | jq -r '.UserPool.Id')
echo "Created user pool with id $user_pool_id"

# Create user pool domain
response=$(
  aws cognito-idp create-user-pool-domain \
    --user-pool-id "$user_pool_id" \
    --region "$REGION" \
    --domain "$USER_POOL_NAME"
)

# Create one resource server for all clients
response=$(
  aws cognito-idp create-resource-server \
    --user-pool-id "$user_pool_id" \
    --region "$REGION" \
    --identifier client-functions \
    --name client-functions \
    --scopes 'ScopeName="invoke",ScopeDescription="invoke clients"'
)

# Create invoker client and admin client for user signup etc.
response=$(
  aws cognito-idp create-user-pool-client \
    --user-pool-id "$user_pool_id" \
    --region "$REGION" \
    --client-name fedless-client-invoker \
    --generate-secret \
    --supported-identity-providers COGNITO \
    --allowed-o-auth-flows "client_credentials" \
    --allowed-o-auth-scopes "client-functions/invoke"
)
client_id=$(echo "$response" | jq '.UserPoolClient.ClientId')
echo "Created client fedless-client-invoker with id $client_id"

callback_url="https://fedless.andreasgrafberger.de/greeting"

# TODO: Consider allowing "implicit" type o-auth flows instead of code

response=$(
  aws cognito-idp create-user-pool-client \
    --user-pool-id "$user_pool_id" \
    --region "$REGION" \
    --client-name server \
    --callback-urls "$callback_url" \
    --supported-identity-providers COGNITO \
    --generate-secret \
    --allowed-o-auth-flows "implicit" \
    --allowed-o-auth-scopes "phone" "email" "openid" "profile"
)
client_id=$(echo "$response" | jq -r '.UserPoolClient.ClientId')
echo "Created client server with id $client_id"

signup_ui_url="https://$USER_POOL_NAME.auth.$REGION.amazoncognito.com/login?" \
signup_ui_url+="client_id=$client_id&response_type=token&scope=email+openid+phone+profile"
signup_ui_url+="&redirect_uri=$callback_url"

echo "Users can sign up with this url: $signup_ui_url"
