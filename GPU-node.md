
# GPU OpenFaaS clients on Kubernetes

As the default Kubernetes architecture does not support GPU splitting or fine-grained
allocation and sharing of GPU resources, we need to use additional plugins. Here, we use [OpenAIOS vGPU scheduler for Kubernetes](https://github.com/4paradigm/k8s-vgpu-scheduler) from 4paradigm.

## 4paradigm's vGPU scheduler for Kubernetes

The following steps are extracted from the [official documentation](https://github.com/4paradigm/k8s-vgpu-scheduler/blob/master/README.md):

### Install the `nvidia-container-toolkit`

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/libnvidia-container.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

### Configure `containerd`
When running *Kubernetes* with *containerd*, edit the config file `nvidia-container-runtime` as the default low-level runtime:
```
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
```

And then restart `containerd`:  

```bash
sudo systemctl daemon-reload && systemctl restart containerd
```

Then, you need to label your GPU nodes which can be scheduled by 4pd-k8s-scheduler by adding "gpu=on", otherwise, it cannot be managed by our scheduler.

```
kubectl label nodes <NODE_NAME> gpu=on
```

You must set the Kubernetes scheduler image version according to your Kubernetes server version during installation.  
(We use the version **1.25.4**)

```
helm repo add vgpu-charts https://4paradigm.github.io/k8s-vgpu-scheduler
helm install vgpu vgpu-charts/vgpu --set scheduler.kubeScheduler.imageTag=v1.25.4 -n kube-system
```

To test if everything works, you could run a simple GPU pod:

```
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
    - name: ubuntu-container
      image: ubuntu:18.04
      command: ["bash", "-c", "sleep 86400"]
      resources:
        limits:
          nvidia.com/gpu: 1
```

It is essential that the value `nvidia.com/gpu` cannot exceed 1 if there is only one card on the cluster. But multiple pods can all have the value 1, and the GPU will be shared among the pods.

## OpenFaaS

### Create a Profile (Optional)

If there are both CPU and GPU clients on the machine, we should create a Profile using taints and affinity to place functions on the node with a GPU.

The following steps are extracted from the [official documentation](https://docs.openfaas.com/reference/profiles/):

First, Label and Taint the node with the GPU:

```bash
kubectl label nodes <NODE_NAME> gpu=on # should be already done in the previous step
kubectl taint nodes <NODE_NAME> gpu:NoSchedule
```

Create a Profile that allows functions to run on this node.

```yaml
kubectl apply -f- << EOF
kind: Profile
apiVersion: openfaas.com/v1
metadata:
  name: withgpu
  namespace: openfaas
spec:
    tolerations:
    - key: "gpu"
      operator: "Exists"
      effect: "NoSchedule"
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
          - matchExpressions:
            - key: gpu
              operator: In
              values:
              - on
EOF
```

Then we could add the following line in the config yaml file for the OpenFaaS function:

```yaml
com.openfaas.profile: withgpu
```

### Deploy a GPU client

Before deploying it to OpenFaaS, we need to prepare the docker image specifically for the GPU clients.

#### Creating the docker image for GPU clients

We would need the following [base image](/docker_images/openfaas/GPU/gpu-base-image/) for GPU clients:

```bash
FROM --platform=${TARGETPLATFORM:-linux/amd64} tensorflow/tensorflow:2.10.0-gpu
COPY requirements.txt .
COPY test_requirements.txt .
RUN pip install -r requirements.txt # tensorflow could be excluded, as already present
RUN pip install -r test_requirements.txt
CMD ["python"]
```

The exact config and Dockerfile for GPU clients and base image is listed [here](/docker_images/openfaas/GPU/).

#### Deploying to OpenFaaS

To deploy a GPU client on OpenFaaS, we need to the requests and limits for the GPU in the YAML file as shown in the following example:

```yaml
version: 1.0
provider:
    name: openfaas
    gateway: http://127.0.0.1:8080
functions:
    gpu−client:
        lang: python3−http−debian
        handler: ./client
        image: fedless−gpu−client
        limits:
            cpu: 2000m
            memory: 3000Mi
            nvidia.com/gpu: 1
            nvidia.com/gpucores: 18
            nvidia.com/gpumem−percentage: 18
        requests:
            nvidia.com/gpu: 1
            nvidia.com/gpucores: 18
            nvidia.com/gpumem−percentage: 18
        environment:
            gpu_memory_fraction: 0.18 # this is for limiting memory usage on the tensorflow level
```

`gpu_memory_fraction` is defined and used in the [function handler](/docker_images/openfaas/GPU/gpu-client/function/handler.py).