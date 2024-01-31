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

# modprobe intelligently adds or removes a module from the Linux kernel
sudo modprobe overlay
sudo modprobe br_netfilter

# Required for kubeadm init
sudo tee /etc/sysctl.d/99-kubernetes-cri.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

# Applying the configuration
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

# Setup kubernetes cluster
CRI_SOCK="unix:///run/containerd/containerd.sock"
sudo kubeadm init --cri-socket $CRI_SOCK --pod-network-cidr=10.244.0.0/16

# Make cluster available to non-root users
mkdir -p "$HOME/.kube"
sudo cp -i /etc/kubernetes/admin.conf "$HOME/.kube/config"
sudo chown "$(id -u)":"$(id -g)" "$HOME/.kube/config"

# Deploy pod network
kubectl apply -f \
  https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

# Untaint control-plane
kubectl taint nodes --all node-role.kubernetes.io/control-plane-


# after 24H
# kubeadm token create --print-join-command

# Quick way to print join command instead of scrolling through the output above
discovery_token=$(openssl x509 -in /etc/kubernetes/pki/ca.crt -noout -pubkey |
  openssl rsa -pubin -outform DER 2>/dev/null | sha256sum | cut -d' ' -f1)

join_token=$(kubeadm token list -o jsonpath="{@.token}")
network_ip=$(hostname -I | cut -d' ' -f1)
echo "====================="
echo "Setup complete!"
echo "To join this cluster run the command below on nodes in the same network"
echo "sudo kubeadm join \"$network_ip:6443\" --token \"$join_token\" --discovery-token-ca-cert-hash \"sha256:$discovery_token\""