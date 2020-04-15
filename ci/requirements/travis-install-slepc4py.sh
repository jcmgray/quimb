#!/bin/sh
set -ex

export PETSC_CONFIGURE_OPTIONS='--download-mumps --download-scalapack --download-parmetis --download-metis --with-scalar-type=complex'
pip install Cython
pip install petsc petsc4py --no-binary :all:
pip install slepc slepc4py --no-binary :all:
