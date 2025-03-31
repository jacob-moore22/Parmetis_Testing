#!/bin/bash

# Guard against sourcing
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed
    : # continue with script
else
    echo "This script should be executed, not sourced"
    echo "Please run: ./build.sh -t <build_type>"
    return 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [-t build_type] [-d]"
    echo "build_type options: serial, openmp, pthreads, cuda, hip"
    echo "  -t    Specify build type (required)"
    echo "  -d    Enable debug build (optional)"
    exit 1
}

# Parse command line arguments
while getopts "t:d" opt; do
    case ${opt} in
        t )
            build_type=$OPTARG
            ;;
        d )
            debug=true
            ;;
        \? )
            usage
            ;;
    esac
done

# Validate build type
if [ -z "$build_type" ]; then
    echo "Error: Build type (-t) is required"
    usage
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
KOKKOS_INSTALL_SCRIPT="../kokkos-install.sh"
BUILD_DIR="${SCRIPT_DIR}/build_${build_type}"
KOKKOS_INSTALL_DIR="${SCRIPT_DIR}/kokkos_${build_type}"

# Create build directory
mkdir -p "${BUILD_DIR}"

# Add after KOKKOS_INSTALL_SCRIPT definition
if [ ! -f "${KOKKOS_INSTALL_SCRIPT}" ]; then
    echo "Error: Could not find kokkos-install.sh at ${KOKKOS_INSTALL_SCRIPT}"
    exit 1
fi

# First, install Kokkos with the specified build type
echo "Installing Kokkos with ${build_type} backend..."
cd "${SCRIPT_DIR}"  # Ensure we're in the right directory
if [ "$debug" = "true" ]; then
    bash "${KOKKOS_INSTALL_SCRIPT}" -t "${build_type}" -d -p "${SCRIPT_DIR}/install"
else
    bash "${KOKKOS_INSTALL_SCRIPT}" -t "${build_type}" -p "${SCRIPT_DIR}/install"
fi

# Source the Kokkos environment
source install/setup_env.sh

# Create CMakeLists.txt for the example
cat > "${SCRIPT_DIR}/CMakeLists.txt" << EOF
cmake_minimum_required(VERSION 3.16)
project(MATARExample1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Find Kokkos
find_package(Kokkos REQUIRED)

add_definitions(-DHAVE_KOKKOS=1)

# Set multiple potential include paths to find MATAR
include_directories(
  "${SCRIPT_DIR}/../MATAR"
  "${CMAKE_CURRENT_SOURCE_DIR}/../MATAR"
  "${CMAKE_SOURCE_DIR}/../MATAR"
  "${CMAKE_BINARY_DIR}/../MATAR"
)

include_directories(
  "${SCRIPT_DIR}/../include"
  "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  "${CMAKE_SOURCE_DIR}/../include"
  "${CMAKE_BINARY_DIR}/../include"
)

message(STATUS "CMAKE_SOURCE_DIR absolute path: ${CMAKE_SOURCE_DIR}")
message(STATUS "SCRIPT_DIR absolute path: ${SCRIPT_DIR}")
message(STATUS "Primary MATAR include path: ${SCRIPT_DIR}/../MATAR")

# Uncomment to debug if MATAR directory is not found
# if(NOT EXISTS "${SCRIPT_DIR}/../MATAR")
#     message(FATAL_ERROR "MATAR directory not found at: ${SCRIPT_DIR}/../MATAR")
# endif()

# Create the executable
add_executable(parmetis_test parmetis_test.cpp)
target_link_libraries(parmetis_test Kokkos::kokkos)

# Check if MPI was found
if(MPI_FOUND)
  message(STATUS "MPI found")
  # Include directories for MPI
  include_directories(\${MPI_INCLUDE_PATH})
  include_directories(\${MPI_CXX_INCLUDE_PATH})

  # Link your target with MPI libraries
  target_link_libraries(parmetis_test ${MPI_CXX_LIBRARIES})
endif()

EOF

# Build the example
echo "Building parmetis_test example..."
cd "${BUILD_DIR}"

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH="${SCRIPT_DIR}/install" -DCMAKE_INCLUDE_PATH="${SCRIPT_DIR}/../MATAR" ..

# Build
make -j$(nproc)

echo "Build completed!"
echo "The executable can be found at: ${BUILD_DIR}/parmetis_test"
echo ""
echo "To run the example:"
echo "cd ${BUILD_DIR}"
echo "./parmetis_test" 