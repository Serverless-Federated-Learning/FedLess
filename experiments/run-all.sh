#!/usr/bin/env bash
set -e

out_directoy="200GPU"

# full list
datasets=("mnist" "femnist" "shakespeare" "speech")
#strategies=("fedavg" "fedlesscan" "fedprox" "fednova" "scaffold" "fedscore")
strategies=("fedavg" "fedlesscan" "fedprox" "fedasync" "fedscore")
# strategies=("fedavg" "fedscore")

# datasets=("femnist" "shakespeare")
datasets=("mnist")
# strategies=("fedlesscan" "fedscore")
# strategies=("fednova" "scaffold")

for ds in "${datasets[@]}"
do
  for st in "${strategies[@]}"
  do
    echo "$ds - $st"
    if [ $st == "fedasync" ] || [ $st == "fedscore" ]
    then
      for ratio in 0.6 0.7 0.8 0.9;
      do
        ./withgpu/run.sh "$ds" "$st" "$out_directoy" "$ratio" || true
      done
    else
      ./withgpu/run.sh "$ds" "$st" "$out_directoy" || true
    fi
    
    #sleep 10
  done
done