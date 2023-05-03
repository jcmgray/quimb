.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/HEAD/docs/_static/logo-banner.png" width="800px">


.. image:: https://github.com/jcmgray/quimb/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/jcmgray/quimb/actions/workflows/tests.yml
  :alt: Tests
.. image:: https://codecov.io/gh/jcmgray/quimb/branch/main/graph/badge.svg
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

``quimb`` is an easy but fast python library
for *'quantum information many-body'* calculations, focusing primarily on **tensor
networks**. The code is hosted on `github <https://github.com/jcmgray/quimb>`_,
and docs are hosted on `readthedocs <http://quimb.readthedocs.io/en/latest/>`_.
Functionality is split in two:

----------------------------------------------------------------------------------

The ``quimb.tensor`` module contains tools for working with **tensors
and tensor networks**. It has a particular focus on automatically
handling arbitrary geometry, e.g. beyond 1D and 2D lattices. With this
you can:

* construct and manipulate arbitrary (hyper) graphs of tensor networks
* automatically contract, optimize and draw networks
* use various backend array libraries such as
  `jax <https://jax.readthedocs.io>`_ and
  `torch <https://pytorch.org/>`_ via
  `autoray <https://github.com/jcmgray/autoray/>`_
* run specific MPS, PEPS, MERA and quantum circuit algorithms, such as DMRG &
  TEBD

.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/HEAD/docs/_static/rand-tensor.svg" width="300px">

----------------------------------------------------------------------------------

The core ``quimb`` module contains tools for reference
**'exact'** quantum calculations, where the states and operator are
represented as either ``numpy.ndarray`` or ``scipy.sparse``
**matrices**. With this you can:

* construct operators in complicated tensor spaces
* find groundstates, excited states and do time evolutions, including
  with `slepc <https://slepc.upv.es/>`_
* compute various quantities including entanglement measures
* take advantage of `numba <https://numba.pydata.org>`_ accelerations
* stochastically estimate $\\mathrm{Tr}f(X)$ quantities

.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/HEAD/docs/_static/rand-herm-matrix.svg" width="300px">

----------------------------------------------------------------------------------

The **full documentation** can be found at:
`quimb.readthedocs.io <https://quimb.readthedocs.io>`_.
Contributions of any sort are very welcome - please see the
`contributing guide <https://github.com/jcmgray/quimb/blob/main/.github/CONTRIBUTING.md>`_.
`Issues <https://github.com/jcmgray/quimb/issues>`_ and
`pull requests <https://github.com/jcmgray/quimb/pulls>`_ are hosted on
`github <https://github.com/jcmgray/quimb>`_.
For other questions and suggestions, please use the
`discussions page <https://github.com/jcmgray/quimb/discussions>`_.
