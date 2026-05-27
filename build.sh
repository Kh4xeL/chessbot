#!/bin/bash
set -e

# Download LibTorch into project root (where CMakeLists.txt expects it)
wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip -O libtorch.zip
unzip -q libtorch.zip        # extracts as ./libtorch/
rm libtorch.zip

# Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)