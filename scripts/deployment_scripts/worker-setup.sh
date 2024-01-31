#!/usr/bin/env bash

# Script for ubuntu (tested with 18.04) to set up kubernetes and join a network

# Install docker
sudo apt-get update
# install docker and extras
sudo apt-get install -y docker.io
sudo apt-get update && sudo apt-get install apt-transport-https ca-certificates curl software-properties-common gnupg2 -y
# configure docker
sudo mkdir /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
"exec-opts": ["native.cgroupdriver=systemd"],
"log-driver": "json-file",
"log-opts": {
"max-size": "100m"
},
"storage-driver": "overlay2"
}
EOF

sudo systemctl enable docker
sudo systemctl daemon-reload
sudo systemctl restart docker


# Install k8 tools
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

echo "Setup complete. Now join the cluster by running the join command returned during cluster setup"
