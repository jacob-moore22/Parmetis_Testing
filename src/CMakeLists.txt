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
  "/var/tmp/repos/Parmetis_Testing/src/../MATAR"
  "/../MATAR"
  "/../MATAR"
  "/../MATAR"
)

include_directories(
  "/var/tmp/repos/Parmetis_Testing/src/../include"
  "/../include"
  "/../include"
  "/../include"
)

message(STATUS "CMAKE_SOURCE_DIR absolute path: ")
message(STATUS "SCRIPT_DIR absolute path: /var/tmp/repos/Parmetis_Testing/src")
message(STATUS "Primary MATAR include path: /var/tmp/repos/Parmetis_Testing/src/../MATAR")

# Uncomment to debug if MATAR directory is not found
# if(NOT EXISTS "/var/tmp/repos/Parmetis_Testing/src/../MATAR")
#     message(FATAL_ERROR "MATAR directory not found at: /var/tmp/repos/Parmetis_Testing/src/../MATAR")
# endif()

# Create the executable
add_executable(parmetis_test parmetis_test.cpp)
target_link_libraries(parmetis_test Kokkos::kokkos)

# Check if MPI was found
if(MPI_FOUND)
  message(STATUS "MPI found")
  # Include directories for MPI
  include_directories(${MPI_INCLUDE_PATH})
  include_directories(${MPI_CXX_INCLUDE_PATH})

  # Link your target with MPI libraries
  target_link_libraries(parmetis_test )
endif()

