.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/develop/docs/_static/quimb_logo_title.png" width="450px">

.. image:: https://github.com/jcmgray/quimb/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/jcmgray/quimb/actions/workflows/tests.yml
  :alt: Tests
.. image:: https://codecov.io/gh/jcmgray/quimb/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/jcmgray/quimb
  :alt: Code Coverage
.. image:: https://app.codacy.com/project/badge/Grade/3c7462a3c45f41fd9d8f0a746a65c37c
  :target: https://www.codacy.com/gh/jcmgray/quimb/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jcmgray/quimb&amp;utm_campaign=Badge_Grade
  :alt: Code Quality
.. image:: https://readthedocs.org/projects/quimb/badge/?version=latest
  :target: http://quimb.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: http://joss.theoj.org/papers/10.21105/joss.00819/status.svg
  :target: https://doi.org/10.21105/joss.00819
  :alt: JOSS Paper
.. image:: https://img.shields.io/pypi/v/quimb?color=teal
   :target: https://pypi.org/project/quimb/
   :alt: PyPI


----------------------------------------------------------------------------------

`quimb <https://github.com/jcmgray/quimb>`_ is an easy but fast python library for quantum information and many-body calculations, including with tensor networks. The code is hosted on `github <https://github.com/jcmgray/quimb>`_, do please submit any issues or pull requests there. It is also thoroughly unit-tested and the tests might be the best place to look for detailed documentation.

The **core** ``quimb`` module:

* Uses straight ``numpy`` and ``scipy.sparse`` matrices as quantum objects
* Accelerates and parallelizes many operations using `numba <https://numba.pydata.org>`_.
* Makes it easy to construct operators in large tensor spaces (e.g. 2D lattices)
* Uses efficient methods to compute various quantities including entanglement measures
* Has many built-in states and operators, including those based on fast, parallel random number generation
* Can perform evolutions with several methods, computing quantities on the fly
* Has an optional `slepc4py <https://bitbucket.org/slepc/slepc4py>`_ interface for easy distributed (MPI) linear algebra. This can massively increase the performance when seeking, for example, mid-spectrum eigenstates

The **tensor network** submodule ``quimb.tensor``:

* Uses a geometry free representation of tensor networks
* Uses `opt_einsum <https://github.com/dgasmith/opt_einsum>`_ to find efficient contraction orders for hundreds or thousands of tensors
* Can perform those contractions on various backends, including with a GPU
* Can plot any network, color-coded, with bond size represented
* Can treat any network as a scipy ``LinearOperator``, allowing many decompositions
* Can perform DMRG1, DMRG2 and DMRGX, in matrix product state language
* Has tools to efficiently address periodic problems (transfer matrix compression and pseudo-orthogonalization)
* Can perform MPS time evolutions with TEBD
* Can optimize arbitrary tensor networks with ``tensorflow``, ``pytorch``, ``jax`` or ``autograd``

.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/develop/docs/_static/montage.png" width="800px">

The **full documentation** can be found at: `<http://quimb.readthedocs.io/en/latest/>`_.
Contributions of any sort are very welcome - please see the `contributing guide <https://github.com/jcmgray/quimb/blob/develop/.github/CONTRIBUTING.md>`_.
`Issues <https://github.com/jcmgray/quimb/issues>`_ and `pull requests <https://github.com/jcmgray/quimb/pulls>`_ are hosted on `github <https://github.com/jcmgray/quimb>`_.
For other questions and suggestions, please use the `dicusssions page <https://github.com/jcmgray/quimb/discussions>`_.
