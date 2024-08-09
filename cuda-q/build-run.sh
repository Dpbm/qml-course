#!/bin/bash

docker build . -t cuda-q
docker run cuda-q
