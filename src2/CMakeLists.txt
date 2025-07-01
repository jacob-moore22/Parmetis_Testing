cmake_minimum_required(VERSION 3.16)
project(PETScExample CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable ExternalProject module
include(ExternalProject)
include(FetchContent)



# Set backend options (same as in kokkos-install.sh)
set(Kokkos_BACKEND "cuda" CACHE STRING "Kokkos backend (serial, openmp, pthreads, cuda, hip)")
set_property(CACHE Kokkos_BACKEND PROPERTY STRINGS serial openmp pthreads cuda hip)

message(STATUS "Building with Kokkos backend: ${Kokkos_BACKEND}")

# Basic Kokkos options (always enabled)
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable Serial backend")
set(Kokkos_ARCH_NATIVE ON CACHE BOOL "Enable native architecture optimizations")
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Build Kokkos tests")
set(BUILD_TESTING OFF CACHE BOOL "Build tests")

# Backend-specific options
if(Kokkos_BACKEND STREQUAL "cuda")
    # Find CUDA
    find_package(CUDAToolkit REQUIRED)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable CUDA backend")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA Lambda support")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR ON CACHE BOOL "Enable CUDA Constexpr support")
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Enable CUDA RDC")
elseif(Kokkos_BACKEND STREQUAL "hip")
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable HIP backend")
    set(Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "Enable HIP RDC")
    set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE)
elseif(Kokkos_BACKEND STREQUAL "openmp")
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP backend")
elseif(Kokkos_BACKEND STREQUAL "pthreads")
    set(Kokkos_ENABLE_THREADS ON CACHE BOOL "Enable Threads backend")
elseif(Kokkos_BACKEND STREQUAL "serial")
    # Serial is already enabled by default
    message(STATUS "Using Serial backend only")
else()
    message(FATAL_ERROR "Invalid Kokkos_BACKEND value: ${Kokkos_BACKEND}")
endif()

# Additional debug build option
option(DEBUG_BUILD "Enable debug build" OFF)
if(DEBUG_BUILD)
    set(CMAKE_BUILD_TYPE Debug)
    message(STATUS "Debug build enabled")
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Download and configure Kokkos
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master # You can specify a specific tag/release here
)

# Make Kokkos available
FetchContent_MakeAvailable(kokkos)

# Define that we're using Kokkos
add_definitions(-DHAVE_KOKKOS=1)









# Set PETSc installation directory
set(PETSC_INSTALL_DIR ${CMAKE_BINARY_DIR}/petsc-install)
set(PETSC_SOURCE_DIR ${CMAKE_BINARY_DIR}/petsc-src)

# Find required dependencies
find_package(MPI REQUIRED)
find_program(MAKE_EXECUTABLE NAMES gmake make)

# Download and build PETSc
ExternalProject_Add(
    petsc_external
    GIT_REPOSITORY https://gitlab.com/petsc/petsc.git
    GIT_TAG main
    GIT_SHALLOW ON
    SOURCE_DIR ${PETSC_SOURCE_DIR}
    CONFIGURE_COMMAND ${PETSC_SOURCE_DIR}/configure
        --prefix=${PETSC_INSTALL_DIR}
        --with-mpi-dir=/apps/spack-managed/gcc-11.3.1/openmpi-4.1.4-7kjpiw7r6ws7snr4snw35e4yrk74x3fq
        --with-blas-lapack
        --download-f2cblaslapack=1
        --with-fc=0
        --download-ptscotch
        --with-debugging=0
        COPTFLAGS=-O3
        CXXOPTFLAGS=-O3
    BUILD_COMMAND ${MAKE_EXECUTABLE} PETSC_DIR=${PETSC_SOURCE_DIR} PETSC_ARCH=arch-linux-c-opt all
    INSTALL_COMMAND ${MAKE_EXECUTABLE} PETSC_DIR=${PETSC_SOURCE_DIR} PETSC_ARCH=arch-linux-c-opt install
    BUILD_IN_SOURCE 1
)

# --download-metis
# --download-parmetis

# Set multiple potential include paths to find MATAR
include_directories(
    "/var/tmp/repos/Parmetis_Testing/src/../MATAR"
    "/var/tmp/repos/Parmetis_Testing/src/../include"
)


# Create the main executable
add_executable(petsc_test petsc_test.cpp)

# Add dependency on PETSc build
add_dependencies(petsc_test petsc_external)

# Link against PETSc and MPI
target_link_libraries(petsc_test 
    PRIVATE 
    Kokkos::kokkos
    ${PETSC_INSTALL_DIR}/lib/libpetsc.so
    ${MPI_CXX_LIBRARIES}
)

target_include_directories(petsc_test 
    PRIVATE 
    ${PETSC_INSTALL_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_compile_definitions(petsc_test PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} PETSC_HAVE_MPI)
target_compile_options(petsc_test PRIVATE ${MPI_CXX_COMPILE_FLAGS})

# Custom target to show build information
add_custom_target(petsc_info
    COMMAND echo "PETSc installation directory: ${PETSC_INSTALL_DIR}"
    COMMAND echo "PETSc include directory: ${PETSC_INSTALL_DIR}/include"
    COMMAND echo "PETSc library directory: ${PETSC_INSTALL_DIR}/lib"
    VERBATIM
)

# Print some helpful information
message(STATUS "PETSc will be downloaded and built automatically")
message(STATUS "PETSc install directory: ${PETSC_INSTALL_DIR}")
message(STATUS "Make sure you have MPI installed on your system")