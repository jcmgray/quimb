#!/bin/sh
set -ex

export PETSC_CONFIGURE_OPTIONS='--download-mumps --download-scalapack --download-parmetis --download-metis --with-scalar-type=complex'
pip install petsc petsc4py
pip install slepc slepc4py
