cmake_minimum_required(VERSION 3.16)
project(MATARExample1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MPI
find_package(MPI REQUIRED)

# Set MPI compiler wrappers
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_C_COMPILER mpicc)

# Check if MPI was found
if(MPI_FOUND)
    message(STATUS "MPI found")
    include_directories(${MPI_INCLUDE_PATH})
    include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

# Find Kokkos
find_package(Kokkos REQUIRED)

add_definitions(-DHAVE_KOKKOS=1)

# Set multiple potential include paths to find MATAR
include_directories(
    "/var/tmp/repos/Parmetis_Testing/src/../MATAR"
    "/var/tmp/repos/Parmetis_Testing/src/../include"
    "/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_parmetis/include"
    "/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_metis/include"
    "/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_gklib/include"
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

# Link libraries - note the order matters
target_link_libraries(parmetis_test 
    PRIVATE
    MPI::MPI_CXX
    Kokkos::kokkos
    -L/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_parmetis/lib -lparmetis
    -L/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_metis/lib -lmetis
    -L/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_gklib/lib -lGKlib
)

# Set rpath for runtime library loading
set_target_properties(parmetis_test PROPERTIES
    INSTALL_RPATH "/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_parmetis/lib:/var/tmp/repos/Parmetis_Testing/src/../parmetis/install_metis/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)
