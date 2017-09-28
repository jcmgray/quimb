#!/bin/sh
set -ex

# install directory for mpi4py and maybe mpi itself
export INSTALL_DIR=$HOME/mpi_stuff

# Check for pre existing mpi installation
if [ pip list | grep mpi4py &>/dev/null ]; then
    echo "mpi already installed"
    exit 0
fi

# make folders
mkdir -p $INSTALL_DIR

# # download and extract openmpi
# OPENMPI_VER=${OPENMPI_VER:-"openmpi-2.1.1"}
# wget "https://www.open-mpi.org/software/ompi/v${OPENMPI_VER:8:3}/downloads/$OPENMPI_VER.tar.gz" -P $INSTALL_DIR
# tar xzf "$INSTALL_DIR/$OPENMPI_VER.tar.gz" -C $INSTALL_DIR
# cd "$INSTALL_DIR/$OPENMPI_VER"

# # compile and install
# ./configure --prefix=$LOCAL COPTFLAGS='-O0' CXXOPTFLAGS='-O0' FOPTFLAGS='-O0'
# make
# make install

# install python package
MPI4PY_REPO=${MPI4PY_REPO:-"https://bitbucket.org/mpi4py/mpi4py.git"}
cd $INSTALL_DIR
git clone $MPI4PY_REPO
cd $INSTALL_DIR/mpi4py
python setup.py build
python setup.py install
