#!/bin/bash

# change gcc to a version compatible with pytorch
source /opt/rh/devtoolset-8/enable

# add nvcc path
PATH=$PATH:/usr/local/cuda-10.2/bin/

# check that you can access nvcc
nvcc --version

/home/arigo/mambaforge/envs/lost2/bin/python setup.py build develop

