#!/bin/sh
set -ex

export PETSC_CONFIGURE_OPTIONS='--download-mumps --download-scalapack --download-parmetis --download-metis --with-scalar-type=complex'
pip install Cython
travis_wait 30 pip install petsc petsc4py --no-binary :all:
travis_wait 30 pip install slepc slepc4py --no-binary :all:
