============
Installation
============

:mod:`quimb` itself is a pure python package and can now be found on PyPI:

.. code-block:: bash

    pip install quimb

However it is recommended to first install the main dependencies using e.g. ``conda``, as below.
The code is hosted on `github <https://github.com/jcmgray/quimb>`_ and if the dependencies are satisfied, a development version can be installed with pip directly from there:

.. code-block:: bash

    pip install --no-deps -U git+https://github.com/jcmgray/quimb.git@main


Required Dependencies
=====================

The core packages ``quimb`` requires are:

* python 3.6+
* `numpy <http://www.numpy.org/>`_
* `scipy <https://www.scipy.org/>`_
* `numba <http://numba.pydata.org/>`_
* `cytoolz <https://github.com/pytoolz/cytoolz>`_
* `tqdm <https://github.com/tqdm/tqdm>`_
* `psutil <https://github.com/giampaolo/psutil>`_

For ease and performance (i.e. mkl compiled libraries), `conda <https://conda.io/miniconda.html/>`_ is the recommended distribution with which to install these.

In addition, the tensor network library, :mod:`quimb.tensor`, requires:

* `opt_einsum <https://github.com/dgasmith/opt_einsum>`_
* `autoray <https://github.com/jcmgray/autoray>`_

``opt_einsum`` efficiently optimizes tensor contraction expressions. It can be installed with ``pip`` or from `conda-forge <https://conda-forge.org>`_ and is a required dependency since various bits of the core ``quimb`` module now make use tensor-network functionality behind the scenes.
``autoray`` allows backend agnostic numeric code for various tensor network operations so that many libraries other than ``numpy`` can be used. It can be installed via ``pip`` from `pypi <https://pypi.org/project/autoray/>`_ or via ``conda`` `from conda-forge <https://anaconda.org/conda-forge/autoray>`_.


Optional Dependencies
=====================

Plotting tensor networks as colored graphs with weighted edges requires:

* `matplotlib <https://matplotlib.org/>`_
* `networkx <https://networkx.github.io/>`_

Fast, multi-threaded random number generation no longer (with `numpy>1.17`) requires `randomgen <https://github.com/bashtage/randomgen>`_ though its bit generators can still be used.

Finally, fast and optionally distributed partial eigen-solving, SVD, exponentiation etc. can be accelerated with ``slepc4py`` and its dependencies:

* `slepc4py <https://bitbucket.org/slepc/slepc4py>`_
* `slepc <http://slepc.upv.es/>`_
* `petsc4py <https://bitbucket.org/petsc/petsc4py>`_
* `petsc <http://www.mcs.anl.gov/petsc/>`_
* `mpi4py <http://mpi4py.readthedocs.io/en/latest/>`_ (v2.1.0+)
* An MPI implementation (`OpenMPI <https://www.open-mpi.org/>`_ recommended, the 1.10.x series seems most robust for spawning processes)

For best performance of some routines, (e.g. shift invert eigen-solving), petsc must be configured with certain options.
Pip can handle this compilation and installation, for example the following script installs everything necessary on Ubuntu:

.. code-block:: bash

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

.. note::

    For the most control and best performance it is recommended to compile and install these (apart from MPI if you are e.g. on a cluster) manually - see the `PETSc instructions <https://www.mcs.anl.gov/petsc/documentation/installation.html>`_.
    It is possible to compile several versions of PETSc/SLEPc side by side, for example a ``--with-scalar-type=complex`` and/or a ``--with-precision=single`` version, naming them with different values of ``PETSC_ARCH``. When loading PETSc/SLEPc, ``quimb`` respects ``PETSC_ARCH`` if it is set, but it cannot dynamically switch between them.
