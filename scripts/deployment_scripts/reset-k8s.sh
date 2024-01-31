# Rest kubernetes cluster using kubeadm
sudo kubeadm reset -f
# Remove all the data from all below locations
sudo rm -rf /etc/cni /etc/kubernetes /var/lib/dockershim /var/lib/etcd /var/lib/kubelet /var/run/kubernetes ~/.kube/*

# Flush all the firewall (iptables) rules
sudo iptables -F && sudo iptables -X
sudo iptables -t nat -F && sudo iptables -t nat -X
sudo iptables -t raw -F && sudo iptables -t raw -X
sudo iptables -t mangle -F && sudo iptables -t mangle -X

# sudo systemctl restart docker
sudo systemctl restart containerd

sudo modprobe overlay
sudo modprobe br_netfilter

sudo tee /etc/sysctl.d/99-kubernetes-cri.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.ipv4.ip_forward                 = 1
net.bridge.bridge-nf-call-ip6tables = 1
EOF

# Applying the configuration
sudo sysctl --quiet --system

# Setup kubernetes cluster
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Make cluster available to non-root users
mkdir -p "$HOME/.kube"
sudo cp -i /etc/kubernetes/admin.conf "$HOME/.kube/config"
sudo chown "$(id -u)":"$(id -g)" "$HOME/.kube/config"

# Deploy pod network
kubectl apply -f \
  https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

# Untaint control plane
kubectl taint nodes --all node-role.kubernetes.io/control-plane-