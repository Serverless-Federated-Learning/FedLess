#!/usr/bin/env bash
set -e

serverless_file_content=$(sed -n '/authorizers:/q;p' serverless.yml)
echo "$serverless_file_content" >serverless.all.yml
echo "    authorizers:" >>serverless.all.yml

for i in {1..20}; do
  authorizer_block="""
      client-authorizer-$i:
        identitySource: \$request.header.Authorization
        issuerUrl: https://cognito-idp.$COGNITO_USER_POOL_REGION.amazonaws.com/$COGNITO_USER_POOL_ID
        audience:
          - $COGNITO_INVOKER_CLIENT_ID
"""
  echo "$authorizer_block" >>serverless.all.yml
done

echo "functions:" >>serverless.all.yml
for i in {1..20}; do

  function_config_obj="""
  client-functions-$i:
    image: baseimage
    events:
      - httpApi:
          path: /client-indep-secure-$i
          method: POST
          integration: lambda-proxy
          #private: true
          authorizer:
            name: client-authorizer-$i
            scopes:
              - client-functions/invoke
  """

  echo "$function_config_obj" >>serverless.all.yml
done

sls deploy -c "serverless.all.yml"
