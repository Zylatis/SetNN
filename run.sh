#!/bin/bash
# NVIDIA Docker image to use TF with CUDA
TAG="18.11-py3"

nvidia-docker run -it --rm -v  /home/graeme_gossel/SetNN:/workspace/SetNN \
nvcr.io/nvidia/tensorflow:${TAG} 
    
exit 0

