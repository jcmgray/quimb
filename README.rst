.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/develop/docs/_static/quimb_logo_title.png" width="450px">

.. image:: https://img.shields.io/travis/jcmgray/quimb/stable.svg
    :target: https://travis-ci.org/jcmgray/quimb
.. image:: https://img.shields.io/codecov/c/github/jcmgray/quimb/develop.svg
  :target: https://codecov.io/gh/jcmgray/quimb
.. image:: https://api.codacy.com/project/badge/Grade/490e11dea3984e25aae1f915865f2c3f
   :target: https://www.codacy.com/app/jcmgray/quimb?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jcmgray/quimb&amp;utm_campaign=Badge_Grade
.. image:: https://landscape.io/github/jcmgray/quimb/develop/landscape.svg?style=flat
   :target: https://landscape.io/github/jcmgray/quimb/develop
   :alt: Code Health
.. image:: https://img.shields.io/readthedocs/quimb/stable.svg
   :target: http://quimb.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

`quimb <https://github.com/jcmgray/quimb>`_ is an easy but fast python library for quantum information and many-body calculations. The main docs are hosted on `readthedocs <http://quimb.readthedocs.io>`_. It is also thoroughly unit-tested and the tests are probably the best place to look for detailed documentation.

The core package:

* Uses `numpy` and `scipy.sparse` matrices as quantum objects
* Accelerates many operations using `numba` and `numexpr`
* Makes it easy to construct operators in large tensor spaces
* Uses efficient methods to compute various quantities including entanglement measures
* Can generate a variety of random states and operators
* Can perform evolutions with several methods, computing quantities on the fly
* Has an optional `slepc4py` interface for easy distributed (MPI) linear algebra. This can massively increase the performance when seeking, for example, mid-spectrum eigenstates.

The tensor network module ``quimb.tensor``:

* Uses a geometry free representation of tensor networks
* Can plot any network, color-coded, with bond size represented
* Can treat any network as a scipy ``LinearOperator``, allowing many decompositions
* Can perform DMRG1, DMRG2 and DMRGX, in matrix product state language.
* Has tools to efficiently address periodic problems (transfer matrix compression and pseudo-orthognolization)

.. raw:: html

    <img src="https://github.com/jcmgray/quimb/blob/develop/docs/_static/mps_en_overlap_cyclic_compressed.png" width="300px">

The full documentation can be found at: `<http://quimb.readthedocs.io/en/latest/>`_.
