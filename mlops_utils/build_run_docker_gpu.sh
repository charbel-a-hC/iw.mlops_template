#!/bin/bash

if [[ "$(docker images -q vunet:latest 2> /dev/null)" == "" ]]; then
    docker build -t vunet . --build-arg CUDA=11.2.1 --build-arg CUDNN=8
fi

docker run --rm -it -v ${PWD}:/VUnet --gpus all --shm-size 16G vunet