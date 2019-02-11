#!/bin/bash
#
# mnist_example.sh                             10/15/2017 
#
# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Example code to show how to run a container with nvidia-docker.
# This example does an MNIST training run.

    TAG="18.11-py3"

        nvidia-docker run -it --rm -v  /home/graeme_gossel/SetNN:/workspace/SetNN \
        nvcr.io/nvidia/tensorflow:${TAG} 
        #~ \
            #~ -w /opt/tensorflow/tensorflow/examples/tutorials/mnist \
            #~ nvcr.io/nvidia/tensorflow:${TAG} \
            #~ python /home/graeme_gossel/ML_dft/src/train.py
  

exit 0

