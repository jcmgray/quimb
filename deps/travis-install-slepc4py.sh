#!/bin/sh
set -ex

export INSTALL_DIR=$HOME/petsc_and_slepc

if [ pip list | grep slepc4py &>/dev/null ]; then
  echo 'Using cached petsc_and_slepc directory.';
  exit 0
fi

if [ -d $INSTALL_DIR ]; then
  rm -rf $INSTALL_DIR
fi
mkdir $INSTALL_DIR
cd $INSTALL_DIR

# ------------------------------ #
# Download required repositories #
# ------------------------------ #
git clone --depth 5 https://bitbucket.org/petsc/petsc.git
git clone --depth 5 https://bitbucket.org/slepc/slepc.git
git clone --depth 5 https://bitbucket.org/petsc/petsc4py.git
git clone --depth 5 https://bitbucket.org/slepc/slepc4py.git


# ---------- #
# BUILD ALL  #
# ---------- #

export PETSC_DIR=$INSTALL_DIR/petsc
export PETSC_ARCH=arch-linux2-c-release

# PETSc
cd $PETSC_DIR
git pull

python2 ./configure \
  --download-mumps \
  --download-scalapack \
  --download-parmetis \
  --download-metis \
  --with-scalar-type=complex

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
cd $INSTALL_DIR/petsc4py
git pull
pip install --no-deps -U .

cd $INSTALL_DIR/slepc4py
git pull
pip install --no-deps -U .

