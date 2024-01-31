#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root_directory="$(dirname "$(dirname "$script_dir")")"
echo $script_dir
echo $root_directory
n_clients=100
clients_per_round=20
allowed_stragglers=30
accuracy_threshold=0.99
rounds=30


dataset_name="mnist"
client_timeout=160
base_out_dir="$root_directory/out/mnist-flower-$n_clients-$clients_per_round"
config_dir="$script_dir/mnist-demo.yaml"
echo $base_out_dir


python -m fedless.controller.scripts \
  -d "mnist" \
  -s "fedavg" \
  -c "$config_dir" \
  --clients "$n_clients" \
  --clients-in-round "$clients_per_round" \
  --stragglers "$allowed_stragglers" \
  --max-accuracy "$accuracy_threshold" \
  --out "$base_out_dir" \
  --rounds "$rounds" \
  --timeout "$client_timeout" 
