#!/bin/sh
set -ex

export INSTALL_DIR=$HOME/petsc_and_slepc

if [ ! -d "$INSTALL_DIR/petsc" ]; then
  if [ -d $INSTALL_DIR ]; then
    rm -rf $INSTALL_DIR
  fi
  mkdir $INSTALL_DIR
  cd $INSTALL_DIR

  # ------------------------------ #
  # Download required repositories #
  # ------------------------------ #
  git clone --depth 1 https://bitbucket.org/petsc/petsc.git
  git clone --depth 1 https://bitbucket.org/slepc/slepc.git
  git clone --depth 1 https://bitbucket.org/mpi4py/mpi4py.git
  git clone --depth 1 https://bitbucket.org/petsc/petsc4py.git
  git clone --depth 1 https://bitbucket.org/slepc/slepc4py.git


  # ---------- #
  # BUILD ALL  #
  # ---------- #
  # PETSc
  export PETSC_DIR=$INSTALL_DIR/petsc
  export PETSC_ARCH=arch-linux2-c-release
  cd $PETSC_DIR
  git pull
  python2 ./configure \
    --download-openmpi \
    --with-scalar-type=complex \
    --download-mumps \
    --download-scalapack \
    --download-parmetis \
    --download-metis \
    --download-ptscotch \
    --with-fortran-kernels=generic \
    --with-debugging=1
  make -s all
  make test
  make streams NPMAX=2

  # SLEPc
  export SLEPC_DIR=$INSTALL_DIR/slepc
  cd $SLEPC_DIR
  git pull
  python2 ./configure
  make -s
  make test


  # ----------------------- #
  # Install python packages #
  # ----------------------- #
  cd $INSTALL_DIR/mpi4py
  export PATH="$PATH:$INSTALL_DIR/petsc/arch-linux2-c-release/bin"
  git pull
  pip install --no-deps .

  cd $INSTALL_DIR/petsc4py
  git pull
  pip install --no-deps .

  cd $INSTALL_DIR/slepc4py
  git pull
  pip install --no-deps .
else
  echo 'Using cached petsc_and_slepc directory.';
fi
