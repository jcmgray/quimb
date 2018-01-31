############
Installation
############

``quimb`` is hosted on `github <https://github.com/jcmgray/quimb>`_. If the dependencies are satisfied, it can be installed with pip directly from there:

.. code-block:: bash

    pip install git+git://github.com/jcmgray/quimb.git@develop

Required Dependencies
=====================

The core packages ``quimb`` requires are:

    * python 3.5+
    * `numpy <http://www.numpy.org/>`_
    * `scipy <https://www.scipy.org/>`_
    * `numba <http://numba.pydata.org/>`_
    * `numexpr <https://github.com/pydata/numexpr>`_
    * `cytoolz <https://github.com/pytoolz/cytoolz>`_

For ease and performance (i.e. mkl compiled libraries), `conda <https://conda.io/miniconda.html/>`_ is the recommended distribution with which to install these.


Optional Dependencies
=====================

The optional dependencies can improve performance considerably in a number of situations.

The tensor network routines benefit greatly from:

  * `opt_einsum <https://github.com/dgasmith/opt_einsum>`_

which efficiently optimizes tensor contraction expressions.

Fast and optionally distributed partial eigen-solving, SVD, exponentiation etc. can be accelerated with ``slepc4py`` and its dependencies:

    * `slepc4py <https://bitbucket.org/slepc/slepc4py>`_
    * `slepc <http://slepc.upv.es/>`_
    * `petsc4py <https://bitbucket.org/petsc/petsc4py>`_
    * `petsc <http://www.mcs.anl.gov/petsc/>`_
    * `mpi4py <http://mpi4py.readthedocs.io/en/latest/>`_ (v2.1.0+)
    * An MPI implementation (`OpenMPI <https://www.open-mpi.org/>`_ recommended, the 1.10.x series seems most robust for spawning processes)

It is recommended to compile and install these (apart from MPI if you are e.g. on a cluster) yourself (see below).

To show real-time progress on certain operations (evolutions, DMRG, ...), `tqdm <https://github.com/tqdm/tqdm>`_ is needed.

For best performance of some routines, (e.g. shift invert eigen-solving), petsc must be configured with certain options. Here is a rough overview of the steps to installing the above in a directory ``$SRC_DIR``, with MPI and ``mpi4py`` already installed. ``$PATH_TO_YOUR_BLAS_LAPACK_LIB`` should point to e.g. `OpenBLAS <https://github.com/xianyi/OpenBLAS>`_ (``libopenblas.so``) or the MKL library (``libmkl_rt.so``). ``$COMPILE_FLAGS`` should be optimizations chosen for your compiler, e.g. for ``gcc`` ``"-O3 -march=native -s -DNDEBUG"``, or for ``icc`` ``"-O3 -xHost"`` etc.


Build PETSC
~~~~~~~~~~~

.. code-block:: bash

    cd $SRC_DIR
    git clone https://bitbucket.org/petsc/petsc.git

    export PETSC_DIR=$SRC_DIR/petsc
    export PETSC_ARCH=arch-auto-complex

    cd petsc
    python2 ./configure \
      --download-mumps \
      --download-scalapack \
      --download-parmetis \
      --download-metis \
      --download-ptscotch \
      --with-debugging=0 \
      --with-blas-lapack-lib=$PATH_TO_YOUR_BLAS_LAPACK_LIB \
      COPTFLAGS="$COMPILE_FLAGS" \
      CXXOPTFLAGS="$COMPILE_FLAGS" \
      FOPTFLAGS="$COMPILE_FLAGS" \
      --with-scalar-type=complex
    make all
    make test
    make streams NPMAX=4


Build SLEPC
~~~~~~~~~~~

.. code-block:: bash

    cd $SRC_DIR
    git clone https://bitbucket.org/slepc/slepc.git
    export SLEPC_DIR=$SRC_DIR/slepc
    cd slepc
    python2 ./configure
    make
    make test


Build the python interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd $SRC_DIR
    git clone https://bitbucket.org/petsc/petsc4py.git
    git clone https://bitbucket.org/slepc/slepc4py.git

    cd $SRC_DIR/petsc4py
    python setup.py build
    python setup.py install

    cd $SRC_DIR/slepc4py
    python setup.py build
    python setup.py install


.. note ::

    It is possible to compile several versions of PETSc/SLEPc side by side, for example a ``--with-scalar-type=real`` version, naming them with different values of ``PETSC_ARCH``. When loading PETSc/SLEPc, ``quimb`` respects ``PETSC_ARCH`` if it is set, but it cannot dynamically switch bewteen them.
