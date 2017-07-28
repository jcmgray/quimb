############
Installation
############


Required Dependencies
---------------------

The core packages ``quimb`` requires are:

    * python 3.5+
    * `numpy <http://www.numpy.org/>`_
    * `scipy <https://www.scipy.org/>`_
    * `numba <http://numba.pydata.org/>`_
    * `numexpr <https://github.com/pydata/numexpr>`_

For ease and performance (i.e. mkl compiled libraries), `conda <http://conda.pydata.org/miniconda.html/>`_ is the recommended distribution with which to install these.


Optional Dependencies
---------------------

The optional dependencies mainly allow high performance, distributed eigen-solving. For sparse systems this functionality is provided by ``slepc4py``, along with its dependencies:

    * `slepc4py <https://bitbucket.org/slepc/slepc4py>`_
    * `slepc <http://slepc.upv.es/>`_
    * `petsc4py <https://bitbucket.org/petsc/petsc4py>`_
    * `petsc <http://www.mcs.anl.gov/petsc/>`_
    * `mpi4py <http://mpi4py.readthedocs.io/en/latest/>`_ (v2.1.0+)
    * An MPI implementation (`OpenMPI <https://www.open-mpi.org/>`_ recommended)

For distributed, dense eigensolving (but probably slower than ``numpy``) the ``scalapy`` package interfaces with ``scalapack``.

    * `scalapack <http://www.netlib.org/scalapack/>`_
    * `scalapy <https://github.com/jrs65/scalapy>`_

