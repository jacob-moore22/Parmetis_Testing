#!/bin/bash

# Set installation prefix to install directory in current directory
GKLIB_INSTALL_PREFIX=${1:-$(pwd)/install_gklib}
METIS_INSTALL_PREFIX=${2:-$(pwd)/install_metis}
PARMETIS_INSTALL_PREFIX=${3:-$(pwd)/install_parmetis}

# Number of parallel jobs for compilation
NUM_THREADS=$(nproc)

# Install dependencies (user must ensure required packages are installed)
echo "Ensure you have cmake, build-essential, and openmpi installed."

# Create a workspace
WORKSPACE_DIR="$(pwd)/clone_repos"
mkdir -p $WORKSPACE_DIR && cd $WORKSPACE_DIR

# Clone repositories
git clone https://github.com/KarypisLab/GKlib.git
git clone https://github.com/KarypisLab/METIS.git
git clone https://github.com/KarypisLab/ParMETIS.git

# Build GKlib
cd GKlib
mkdir build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=$GKLIB_INSTALL_PREFIX
make -j$NUM_THREADS
make install
cd ../..

# Build METIS
cd METIS
make config gklib_path=$GKLIB_INSTALL_PREFIX prefix=$METIS_INSTALL_PREFIX
make -j$NUM_THREADS
make install
cd ..



# Build ParMETIS
cd ParMETIS
mkdir build && cd build
cmake .. -DGKLIB_PATH=$GKLIB_INSTALL_PREFIX -DMETIS_PATH=$METIS_INSTALL_PREFIX -DCMAKE_INSTALL_PREFIX=$PARMETIS_INSTALL_PREFIX \
         -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
         CFLAGS="$(mpicc --showme:compile)" \
         LDFLAGS="$(mpicc --showme:link)"
make -j$NUM_THREADS
make install
cd ../..

# Cleanup
echo "ParMETIS installation complete! Consider adding $INSTALL_PREFIX/lib to your LD_LIBRARY_PATH if needed."
echo "export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:\$LD_LIBRARY_PATH"


# ParMETIS installation complete! Consider adding /var/tmp/repos/Parmetis_Testing/parmetis/install/lib to your LD_LIBRARY_PATH if needed.
# export LD_LIBRARY_PATH=/var/tmp/repos/Parmetis_Testing/parmetis/install/lib:$LD_LIBRARY_PATH


# make config gklib_path=/var/tmp/repos/Parmetis_Testing/parmetis/install/lib prefix=/var/tmp/repos/Parmetis_Testing/parmetis/install