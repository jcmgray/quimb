#!/bin/sh
set -ex

# Check for pre existing mpi installation
if [ pip list | grep mpi4py &>/dev/null ]; then
    echo "mpi4py already installed"
    exit 0
fi

# install directory for mpi4py and maybe mpi itself
export BUILD_DIR="$HOME/mpi_stuff"
mkdir -p $BUILD_DIR
export INSTALL_DIR="$BUILD_DIR/openmpi_install"


# # download and extract openmpi
OPENMPI_VER=${OPENMPI_VER:-"openmpi-1.10.7"}
if [ ${OPENMPI_VER:11:1} = '.' ]; then
  DOWNLOAD_PREFIX=v${OPENMPI_VER:8:3}
else
  DOWNLOAD_PREFIX=v${OPENMPI_VER:8:4}
fi

TAR_REMOTE="https://www.open-mpi.org/software/ompi/$DOWNLOAD_PREFIX/downloads/$OPENMPI_VER.tar.gz"
wget $TAR_REMOTE -P $BUILD_DIR
tar xzf "$BUILD_DIR/$OPENMPI_VER.tar.gz" -C $BUILD_DIR
cd "$BUILD_DIR/$OPENMPI_VER"

# compile and install
./configure \
    COPTFLAGS='-O0' \
    CXXOPTFLAGS='-O0' \
    FOPTFLAGS='-O0' \
    --prefix=$INSTALL_DIR \
    --disable-dlopen
make
make install

# install python package
MPI4PY_REPO=${MPI4PY_REPO:-"https://bitbucket.org/mpi4py/mpi4py.git"}
cd $BUILD_DIR
if [ -d "$BUILD_DIR/mpi4py" ]; then
    git clone $MPI4PY_REPO
fi
cd $BUILD_DIR/mpi4py
git pull
pip install --no-deps -U .
