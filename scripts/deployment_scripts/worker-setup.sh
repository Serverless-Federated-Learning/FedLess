#!/usr/bin/env bash

# Script for ubuntu (tested with 20.04) to set up kubernetes
# tested with kubernetes version: v1.25.5 (1.25.5-00)
K8S_VERSION=1.25.5-00

sudo apt-get update 
sudo apt-get -y install btrfs-progs pkg-config libseccomp-dev unzip tar libseccomp2 socat util-linux apt-transport-https curl ipvsadm 

# containerd
# https://github.com/containerd/containerd
wget --continue --quiet https://github.com/containerd/containerd/releases/download/v1.6.8/containerd-1.6.8-linux-amd64.tar.gz
sudo tar -C /usr/local -xzf containerd-1.6.8-linux-amd64.tar.gz > /dev/null 2>&1

# runc is a CLI tool 
# for spawning and running containers on Linux according to the OCI specification.
wget --continue --quiet https://github.com/opencontainers/runc/releases/download/v1.1.3/runc.amd64
mv runc.amd64 runc
sudo install -D -m0755 runc /usr/local/sbin/runc

containerd --version || echo "failed to build containerd"

# install k8s
curl --silent --show-error https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo sh -c "echo 'deb http://apt.kubernetes.io/ kubernetes-xenial main' > /etc/apt/sources.list.d/kubernetes.list"
sudo apt-get update >> /dev/null
sudo apt-get -y install cri-tools ebtables ethtool kubeadm=$K8S_VERSION kubectl=$K8S_VERSION kubelet=$K8S_VERSION kubernetes-cni
sudo apt-mark hold kubelet kubeadm kubectl

# Required for kubeadm init
sudo tee /etc/sysctl.d/99-kubernetes-cri.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

sudo sysctl --quiet --system

# Enable containerd with systemd
wget https://github.com/containerd/containerd/archive/refs/tags/v1.6.8.zip
sudo apt-get install unzip
unzip v1.6.8.zip
cd containerd-1.6.8
sudo cp containerd.service /etc/systemd/system/
sudo chmod 664 /etc/systemd/system/containerd.service
sudo systemctl enable containerd
sudo systemctl start containerd

echo "Setup complete. Now join the cluster by running the join command returned during cluster setup"

# Ports need to open
# https://kubernetes.io/docs/reference/networking/ports-and-protocols/
