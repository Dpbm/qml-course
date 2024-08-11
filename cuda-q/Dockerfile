FROM nvcr.io/nvidia/nightly/cuda-quantum:latest

WORKDIR bell
COPY . .
RUN sudo apt update && \
    sudo apt upgrade -y && \
    sudo apt install make -y && \
    make
ENTRYPOINT ["/home/cudaq/bell/bell"]
