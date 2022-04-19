#!/bin/bash

source /opt/rh/devtoolset-8/enable

PATH=$PATH:/usr/local/cuda-10.2/bin/

nvcc --version

/home/arigo/miniforge3/envs/lost/bin/python setup.py build develop

