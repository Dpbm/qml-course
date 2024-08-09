#!/bin/bash

FILE=https://github.com/NVIDIA/cuda-quantum/releases/download/0.8.0/install_cuda_quantum.x86_64

curl -L -O ${FILE}

sudo -E bash install_cuda_quantum.$(uname -m) --accept
. /etc/profile

export PATH="/opt/nvidia/cudaq/bin:$PATH"
