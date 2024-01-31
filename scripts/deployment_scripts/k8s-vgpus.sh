#!/usr/bin/env bash
K8S_VERSION=1.25.5

# Install helm if necessary
if ! command -v helm &>/dev/null; then
  echo "Helm installation not found, installing it now"
  curl https://baltocdn.com/helm/signing.asc | sudo apt-key add -
  sudo apt-get install apt-transport-https --yes
  echo "deb https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
  sudo apt-get update
  sudo apt-get install helm --yes
fi

# Install the nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/libnvidia-container.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit


# install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2


# Configure containerd with a default config.toml configuration file
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
sudo mkdir -p /etc/containerd && sudo containerd config default | sudo tee /etc/containerd/config.toml

# Configure containerd: set up nvidia-container-runtime as the default runtime
# This needs to be done manually.
echo '[MANUAL STEP] Configure containerd: set up nvidia-container-runtime as the default runtime'
cat << EOF
version = 2
[plugins]
  [plugins."io.containerd.grpc.v1.cri"]
    [plugins."io.containerd.grpc.v1.cri".containerd]
      default_runtime_name = "nvidia"

      [plugins."io.containerd.grpc.v1.cri".containerd.runtimes]
        [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia]
          privileged_without_host_devices = false
          runtime_engine = ""
          runtime_root = ""
          runtime_type = "io.containerd.runc.v2"
          [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.nvidia.options]
            BinaryName = "/usr/bin/nvidia-container-runtime"
EOF
read -p "Copy the config above, and press any key to nano the config file... " -n1 -s
sudo nano /etc/containerd/config.toml

sudo systemctl restart containerd

# label all nodes with gpu=on
kubectl label node $(kubectl get nodes -o jsonpath='{.items[0].metadata.name}') gpu=on

# add vGPU repo
helm repo add vgpu-charts https://4paradigm.github.io/k8s-vgpu-scheduler
helm install vgpu vgpu-charts/vgpu --set scheduler.kubeScheduler.imageTag=v$K8S_VERSION -n kube-system

# check if install successful
kubectl get pods -n kube-system

# ---- uninstall on fail
# kubectl delete job vgpu-admission-create -n kube-system
# helm uninstall vgpu -n kube-system

# ---- GPU Test Pod
# cat <<EOF | kubectl apply -f -
# apiVersion: v1
# kind: Pod
# metadata:
#   name: gpu-pod
# spec:
#   containers:
#     - name: ubuntu-container
#       image: ubuntu:18.04
#       command: ["bash", "-c", "sleep 86400"]
#       resources:
#         limits:
#           nvidia.com/gpu: 1 # requesting 2 vGPUs
#           nvidia.com/gpumem: 3000 # Each vGPU contains 3000m device memory （Optional,Integer）
#           nvidia.com/gpucores: 30 # Each vGPU uses 30% of the entire GPU （Optional,Integer)
# EOF
# kubectl exec -it gpu-pod -- /bin/bash
# kubectl delete pod gpu-pod
