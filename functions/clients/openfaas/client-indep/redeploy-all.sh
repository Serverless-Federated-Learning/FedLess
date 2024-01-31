#!/usr/bin/env bash
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
config_base="client-indep-fedmd-1" # use ...-lrz-4-cores for 4-core versions

# for i in {1..4}; do
#   faas deploy -f "$config_base-lrz-$i.yml"
# done

faas-cli up -f "$config_base.yml"
