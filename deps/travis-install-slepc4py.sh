#!/bin/sh

# Required system packages...
#     gfortran mpich valgrind bison flex cmake libtool autoconf
#     gcc g++ build-essential perl m4 git

set -e

export INSTALL_DIR=$HOME/petsc_and_slepc

mkdir $INSTALL_DIR
cd $INSTALL_DIR

# Download required repositories
git clone --depth 1 https://github.com/xianyi/OpenBLAS.git
git clone --depth 1 https://bitbucket.org/petsc/petsc.git
git clone --depth 1 https://bitbucket.org/slepc/slepc.git
# git clone --depth 1 https://bitbucket.org/mpi4py/mpi4py.git
git clone --depth 1 https://bitbucket.org/petsc/petsc4py.git
git clone --depth 1 https://bitbucket.org/slepc/slepc4py.git

# if [ ! -d "$INSTALL_DIR" ]; then
# else
#   echo 'Using cached petsc_and_slepc directory.';
# fi

# Build Openblas
cd $INSTALL_DIR/OpenBLAS
git pull
make -s

# Build PETSc
export PETSC_DIR=$INSTALL_DIR/petsc
export PETSC_ARCH=arch-linux2-c-release-openblas
cd $PETSC_DIR
git pull
python2 ./configure \
  --with-mpi=0  \  # THIS IS FOR TRAVIS ONLY
  --with-blas-lapack-lib=$INSTALL_DIR/OpenBLAS/libopenblas.a \
  --with-scalar-type=complex  \
  --download-mumps  \
  --download-scalapack  \
  --download-parmetis  \
  --download-metis --download-ptscotch  \
  --with-debugging=0  \
  COPTFLAGS='-O3 -march=native -mtune=native'  \
  CXXOPTFLAGS='-O3 -march=native -mtune=native'  \
  FOPTFLAGS='-O3 -march=native -mtune=native'
make -s all
make test
make streams

# Build SLEPc
export SLEPC_DIR=$INSTALL_DIR/slepc
cd $SLEPC_DIR
git pull
python2 ./configure
make -s
make test

# ----------------------- #
# Install python packages #
# ----------------------- #

# cd $INSTALL_DIR/mpi4py
# git pull
# pip install --no-deps .

cd $INSTALL_DIR/petsc4py
git pull
pip install --no-deps .

cd $INSTALL_DIR/slepc4py
git pull
pip install --no-deps .
