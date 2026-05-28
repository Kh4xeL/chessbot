#!/bin/bash
set -e

# Download LibTorch into project root
wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip -O libtorch.zip
unzip -q libtorch.zip
rm libtorch.zip

# Build C++ engine
mkdir -p build && cd build
cmake .. 
make -j$(nproc)

# Make engine executable
chmod +x engine