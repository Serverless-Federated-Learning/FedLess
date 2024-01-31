#!/usr/bin/env bash
set -e

serverless_file_content=$(sed -n '/functions:/q;p' serverless.yml)
echo "$serverless_file_content" > serverless.all.yml
echo "functions:" >> serverless.all.yml

for i in {1..20}; do

  function_config_obj="""
  client-function-$i:
    image: baseimage
    events:
      - http:
          path: federated/client-$i
          method: post
          integration: lambda-proxy
          private: true
  """

  echo "$function_config_obj" >> serverless.all.yml
done

sls deploy -c "serverless.all.yml"
