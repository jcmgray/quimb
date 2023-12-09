# Installation

`quimb` is available on both [pypi](https://pypi.org/project/quimb/) and
[conda-forge](https://anaconda.org/conda-forge/quimb). While `quimb` is
pure python and has no direct dependencies itself, the recommended distribution
would be [mambaforge](https://github.com/conda-forge/miniforge#mambaforge)
for installing the various backend array libraries and their dependencies.

**Installing with `pip`:**
```bash
pip install quimb
```

**Installing with `conda`:**
```bash
conda install -c conda-forge quimb
```

**Installing with `mambaforge`:**
```bash
mamba install quimb
```
```{hint}
Mamba is a faster version of `conda`, and the -forge distritbution comes
pre-configured with only the `conda-forge` channel, which further simplifies
and speeds up installing dependencies.
```

**Installing the latest version directly from github:**

If you want to checkout the latest version of features and fixes, you can
install directly from the github repository:
```bash
pip install -U git+https://github.com/jcmgray/quimb.git
```

**Installing a local, editable development version:**

If you want to make changes to the source code and test them out, you can
install a local editable version of the package:
```bash
git clone https://github.com/jcmgray/quimb.git
pip install --no-deps -U -e quimb/
```

## Required Dependencies

The core packages `quimb` requires are:

- python 3.8+
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [numba](http://numba.pydata.org/)
- [cytoolz](https://github.com/pytoolz/cytoolz)
- [tqdm](https://github.com/tqdm/tqdm)
- [psutil](https://github.com/giampaolo/psutil)

For ease and performance (i.e. mkl compiled libraries), [conda](https://conda.io/miniconda.html/) is the recommended distribution with which to install these.

In addition, the tensor network library, {mod}`quimb.tensor`, requires:

- [cotengra](https://cotengra.readthedocs.io)
- [autoray](https://autoray.readthedocs.io)

`cotengra` efficiently optimizes and performs tensor contraction expressions. It can be installed with `pip` or from [conda-forge](https://conda-forge.org) and is a required dependency since various bits of the core `quimb` module now make use tensor-network functionality behind the scenes.
`autoray` allows backend agnostic numeric code for various tensor network operations so that many libraries other than `numpy` can be used. It can be installed via `pip` from [pypi](https://pypi.org/project/autoray/) or via `conda` [from conda-forge](https://anaconda.org/conda-forge/autoray).

## Optional Dependencies

Plotting tensor networks as colored graphs with weighted edges requires:

- [matplotlib](https://matplotlib.org/)
- [networkx](https://networkx.github.io/)
- [pygraphviz](https://pygraphviz.github.io/) (optional, for faster layouts)

Fast, multi-threaded random number generation no longer (with `numpy>1.17`) requires [randomgen](https://github.com/bashtage/randomgen) though its bit generators can still be used.

Finally, fast and optionally distributed partial eigen-solving, SVD, exponentiation etc. can be accelerated with `slepc4py` and its dependencies:

- [slepc4py](https://bitbucket.org/slepc/slepc4py)
- [slepc](http://slepc.upv.es/)
- [petsc4py](https://bitbucket.org/petsc/petsc4py)
- [petsc](http://www.mcs.anl.gov/petsc/)
- [mpi4py](http://mpi4py.readthedocs.io/en/latest/) (v2.1.0+)
- An MPI implementation ([OpenMPI](https://www.open-mpi.org/) recommended, the 1.10.x series seems most robust for spawning processes).

To install these from conda-forge, with complex dtype specified for example, use:
```bash
mamba install -c conda-forge mpi4py petsc=*=*complex* petsc4py slepc=*=*complex* slepc4py
```

For best performance of some routines, (e.g. shift invert eigen-solving), petsc must be configured with certain options. Pip can handle this compilation and installation, for example the following script installs everything necessary on Ubuntu:

```bash
#!/bin/bash

# install build tools, OpenMPI, and OpenBLAS
sudo apt install -y openmpi-bin libopenmpi-dev gfortran bison flex cmake valgrind curl autoconf libopenblas-base libopenblas-dev

# optimization flags, e.g. for intel you might want "-O3 -xHost"
export OPTFLAGS="-O3 -march=native -s -DNDEBUG"

# petsc options, here configured for real
export PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --download-mumps --download-scalapack --download-parmetis --download-metis --COPTFLAGS='$OPTFLAGS' --CXXOPTFLAGS='$OPTFLAGS' --FOPTFLAGS='$OPTFLAGS'"

# make sure using all the same version
export PETSC_VERSION=3.14.0
pip install petsc==$PETSC_VERSION --no-binary :all:
pip install petsc4py==$PETSC_VERSION --no-binary :all:
pip install slepc==$PETSC_VERSION --no-binary :all:
pip install slepc4py==$PETSC_VERSION --no-binary :all:
```

:::{note}
For the most control and best performance it is recommended to compile and install these (apart from MPI if you are e.g. on a cluster) manually - see the [PETSc instructions](https://www.mcs.anl.gov/petsc/documentation/installation.html).
It is possible to compile several versions of PETSc/SLEPc side by side, for example a `--with-scalar-type=complex` and/or a `--with-precision=single` version, naming them with different values of `PETSC_ARCH`. When loading PETSc/SLEPc, `quimb` respects `PETSC_ARCH` if it is set, but it cannot dynamically switch between them.
:::
