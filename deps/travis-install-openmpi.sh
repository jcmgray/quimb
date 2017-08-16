#!/bin/sh
set -ex

export LOCAL=${LOCAL:-"$HOME/local"}

# Check for pre existing mpi installation
if [ pip list | grep mpi4py &>/dev/null ]; then
    echo "mpi already installed"
    exit 0
fi

# setup the source locations etc
SRC_DIR=${SRC_DIR:-"$LOCAL/../src"}
OPENMPI_VER=${OPENMPI_VER:-"openmpi-2.1.1"}
MPI4PY_REPO=${MPI4PY_REPO:-"https://bitbucket.org/mpi4py/mpi4py.git"}

# make folders
mkdir -p $LOCAL
mkdir -p $SRC_DIR

# # download and extract openmpi
# wget "https://www.open-mpi.org/software/ompi/v${OPENMPI_VER:8:3}/downloads/$OPENMPI_VER.tar.gz" -P $SRC_DIR
# tar xzf "$SRC_DIR/$OPENMPI_VER.tar.gz" -C $SRC_DIR
# cd "$SRC_DIR/$OPENMPI_VER"

# # compile and install
# ./configure --prefix=$LOCAL COPTFLAGS='-O0' CXXOPTFLAGS='-O0' FOPTFLAGS='-O0'
# make
# make install

# install python package
cd $SRC_DIR
git clone $MPI4PY_REPO
cd $SRC_DIR/mpi4py
python setup.py build
python setup.py install
