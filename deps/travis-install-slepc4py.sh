#!/bin/sh

# Required system packages...
#     gfortran mpich valgrind bison flex cmake libtool autoconf
#     gcc g++ build-essential perl m4 git

set -e

mkdir $HOME/petsc_and_slepc
cd $HOME/petsc_and_slepc
# Download required repositories
git clone --depth 1 https://github.com/xianyi/OpenBLAS.git
git clone --depth 1 https://bitbucket.org/petsc/petsc.git
git clone --depth 1 https://bitbucket.org/slepc/slepc.git
git clone --depth 1 https://bitbucket.org/mpi4py/mpi4py.git
git clone --depth 1 https://bitbucket.org/petsc/petsc4py.git
git clone --depth 1 https://bitbucket.org/slepc/slepc4py.git

# if [ ! -d "$HOME/petsc_and_slepc" ]; then
# else
#   echo 'Using cached petsc_and_slepc directory.';
# fi

# Build Openblas
cd $HOME/petsc_and_slepc/OpenBLAS
git pull
make

# Build PETSc
export PETSC_DIR=$HOME/petsc_and_slepc/petsc
export PETSC_ARCH=arch-linux2-c-release-openblas
cd $PETSC_DIR
git pull
python2 ./configure \
  --with-blas-lapack-lib=$HOME/petsc_and_slepc/OpenBLAS/libopenblas.a \
  --with-scalar-type=complex \
  --download-mumps \
  --download-scalapack \
  --download-parmetis \
  --download-metis --download-ptscotch \
  --with-debugging=0 \
  COPTFLAGS='-O3 -march=native -mtune=native' \
  CXXOPTFLAGS='-O3 -march=native -mtune=native' \
  FOPTFLAGS='-O3 -march=native -mtune=native'
make all
make test
make streams

# Build SLEPc
export SLEPC_DIR=$HOME/petsc_and_slepc/slepc
cd $SLEPC_DIR
git pull
python2 ./configure
make
make test

# Install python packages
cd $HOME/petsc_and_slepc/mpi4py
git pull
pip install --no-deps .

cd $HOME/petsc_and_slepc/petsc4py
git pull
pip install --no-deps .

cd $HOME/petsc_and_slepc/slepc4py
git pull
pip install --no-deps .
