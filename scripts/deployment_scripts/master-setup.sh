#!/usr/bin/env bash

# Script for ubuntu (tested with 18.04) to set up kubernetes and openwhisk

# Install docker
sudo apt-get update
sudo apt-get install -y docker.io

# Install k8 tools
sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Setup kubernetes cluster
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Make cluster available to non-root users
mkdir -p "$HOME/.kube"
sudo cp -i /etc/kubernetes/admin.conf "$HOME/.kube/config"
sudo chown "$(id -u)":"$(id -g)" "$HOME/.kube/config"

# Deploy pod network
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

# Untaint master node
kubectl taint nodes --all node-role.kubernetes.io/control-plane-

# Quick way to print join command instead of scrolling through the output above
discovery_token=$(openssl x509 -in /etc/kubernetes/pki/ca.crt -noout -pubkey |
  openssl rsa -pubin -outform DER 2>/dev/null | sha256sum | cut -d' ' -f1)
join_token=$(kubeadm token list -o jsonpath="{@.token}")
network_ip=$(hostname -I | cut -d' ' -f1)
echo "====================="
echo "Setup complete!"
echo "To join this cluster run the command below on nodes in the same network"
echo "sudo kubeadm join \"$network_ip:6443\" --token \"$join_token\" --discovery-token-ca-cert-hash \"sha256:$discovery_token\""
