#!/bin/bash

yes | docker container prune
docker image rm cuda-q
