#!/usr/bin/env bash

config_base="client-indep" # use ...-lrz-4-cores for 4-core versions

for i in {1..4}; do
  faas deploy -f "$config_base-lrz-$i.yml"
done

faas deploy -f "$config_base.yml"
