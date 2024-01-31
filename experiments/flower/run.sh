#!/usr/bin/env bash

set -e

server_ssh_host="lrz-4xlarge"
server_ip="0.0.0.0"
port="31532"
server_address="$server_ip:$port"
server_cpus="2"    #"4"
server_memory="18g" # "16g"
# rounds=1
# min_num_clients=20
num_clients_total=3

client_cpus=1.0
client_memory="2g"
dataset="shakespeare"
batch_size=10
epochs=5
optimizer="Adam"
lr=0.001

session_id="$RANDOM"

# Syntax='ssh-host;clients'
# workers[0]='invasic;30'
# workers[1]='sk1;15'
# workers[2]='sk2;15'
# workers[3]='sksmall;5'
# workers[4]='lrz-1;5'
# workers[5]='lrz-2;5'
# workers[6]='lrz-3;5'
# workers[0]='lrz-4xlarge;40'
worker='lrz-4xlarge;40' 

echo "Update server and client images (build and push)"
docker build -f server.Dockerfile -t "flower-server" .
docker tag "flower-server" mohamedazab/flower:server
docker push mohamedazab/flower:server
docker build -f client.Dockerfile -t "flower-client" .
docker tag "flower-client" mohamedazab/flower:client
docker push mohamedazab/flower:client
mkdir -p flower-logs

run_experiment() {
  dataset=$1
  min_num_clients=$2
  client_cpus=$3
  client_memory=$4
  batch_size=$5
  epochs=$6
  optimizer=$7
  lr=$8
  rounds=$9
  session_id=${10}

  echo "Experiment: dataset=$dataset, min_num_clients=$min_num_clients, client_cpus=$client_cpus,
 client_memory=$client_memory, dataset=$dataset, batch_size=$batch_size, epochs=$epochs,
 rounds=$rounds, optimizer=$optimizer, lr=$lr"

  echo "Removing running container if it exists..."
  docker stop fl-server || true

  exp_filename="flower-logs/fedless_${dataset}_${min_num_clients}_${num_clients_total}_${epochs}_${session_id}"

  echo "Starting server, results are stored in $exp_filename.out and $exp_filename.err"
  run_cmd="docker run --rm -p $port:$port --name fl-server --network host \
-e https_proxy=\$http_proxy \
-e no_proxy=$server_ip \
--cpus $server_cpus --memory $server_memory \
mohamedazab/flower:server --rounds $rounds --min-num-clients $min_num_clients --dataset=$dataset"
  # shellcheck disable=SC2029
  nohup $run_cmd > $exp_filename.out 2> $exp_filename.err < /dev/null &
  sleep 10
  echo "Deploying and starting clients..."

  current_partition=0

  echo "Starting clients..."
    # shellcheck disable=SC2206
    worker_info=(${worker//;/ })
    ssh_host=${worker_info[0]}
    cores_assigned_to_host=${worker_info[1]}
    # if [[ $current_partition -ge $num_clients_total ]]; then
    #   break
    # fi
    echo "Starting $cores_assigned_to_host clients on $ssh_host"
    for ((i = 1; i <= cores_assigned_to_host; i++)); do
      if [[ $current_partition -ge $num_clients_total ]]; then
        break
      fi
      run_cmd="docker run --rm --name fl-client-$current_partition \
--network host \
mohamedazab/flower:client \
--server $server_address \
--dataset $dataset \
--partition $current_partition \
--batch-size $batch_size \
--epochs $epochs \
--optimizer $optimizer \
--lr $lr \
--clients-total $num_clients_total"
      echo "($ssh_host) $run_cmd"
      # shellcheck disable=SC2029
      nohup $run_cmd > ${exp_filename}_$current_partition.out 2> ${exp_filename}_$current_partition.err < /dev/null &
      current_partition=$((current_partition + 1))
    done

  if ((current_partition >= num_clients_total)); then
    echo "Successfully deployed all clients"
  else
    echo "WARNING: Tried to deploy client partition ($current_partition / $num_clients_total) but no compute left..."
  fi
  sleep 10

## call client.py with docker exec
  while docker ps | grep mohamedazab/flower:server; do
    echo "Not finished yet"
    sleep 20
  done
  sleep 10
  wait
}

#run_experiment dataset min_num_clients client_cpus client_memory batch_size epochs optimizer lr rounds session_id

## MNIST
run_experiment "mnist" 2 1 "2g" 10 5 "Adam" 0.001 4 "$RANDOM"

