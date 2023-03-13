#!/bin/sh

# Run the following commands to
# sudo find / -name 'nvcc'  # Path to binaries
# sudo find / -name 'libcublas.so.*'  # Path to libraries

# disable pre-allocation of memory for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# add the src folder to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}://src"

# set the LD_LIBRARY_PATH to include the CUDA libraries
# https://www.tensorflow.org/install/pip#linux
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
