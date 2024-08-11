#!/bin/bash

echo "Installing Dependencies..."

sudo apt update && sudo apt upgrade -y
sudo apt install -y openmpi-bin \
                 libopenmpi-dev \
                 make \
                 curl \
                 gcc \
                 build-essential \
                 libcublas-12-6 \
                 libcublas-dev-12-6
