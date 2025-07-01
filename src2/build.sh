#!/bin/bash

# Build script for PETSc test project
set -e

echo "Building PETSc test project..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc

# Build the project
echo "Building the project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "To run the test: mpirun -np 2 ./petsc_test" 