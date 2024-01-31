#!/usr/bin/env bash

# install GPU drivers
# https://cloud.google.com/compute/docs/gpus/install-drivers-gpu

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# Verifying the GPU driver install
sudo nvidia-smi