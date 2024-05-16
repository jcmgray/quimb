![quimb logo](https://github.com/jcmgray/quimb/blob/HEAD/docs/_static/logo-banner.png?raw=true)

[![Tests](https://github.com/jcmgray/quimb/actions/workflows/tests.yml/badge.svg)](https://github.com/jcmgray/quimb/actions/workflows/tests.yml)
[![Code Coverage](https://codecov.io/gh/jcmgray/quimb/branch/main/graph/badge.svg)](https://codecov.io/gh/jcmgray/quimb)
[![Code Quality](https://app.codacy.com/project/badge/Grade/3c7462a3c45f41fd9d8f0a746a65c37c)](https://www.codacy.com/gh/jcmgray/quimb/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jcmgray/quimb&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/quimb/badge/?version=latest)](http://quimb.readthedocs.io/en/latest/?badge=latest)
[![JOSS Paper](http://joss.theoj.org/papers/10.21105/joss.00819/status.svg)](https://doi.org/10.21105/joss.00819)
[![PyPI](https://img.shields.io/pypi/v/quimb?color=teal)](https://pypi.org/project/quimb/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/quimb/badges/version.svg)](https://anaconda.org/conda-forge/quimb)

[`quimb`](https://github.com/jcmgray/quimb) is an easy but fast python library for *'quantum information many-body'* calculations, focusing primarily on **tensor networks**. The code is hosted on [github](https://github.com/jcmgray/quimb), and docs are hosted on [readthedocs](http://quimb.readthedocs.io/en/latest/). Functionality is split in two:

---

The `quimb.tensor` module contains tools for working with **tensors and tensor networks**. It has a particular focus on automatically handling arbitrary geometry, e.g. beyond 1D and 2D lattices. With this you can:

- construct and manipulate arbitrary (hyper) graphs of tensor networks
- automatically [contract](https://cotengra.readthedocs.io), optimize and draw networks
- use various backend array libraries such as [jax](https://jax.readthedocs.io) and [torch](https://pytorch.org/) via [autoray](https://github.com/jcmgray/autoray/)
- run specific MPS, PEPS, MERA and quantum circuit algorithms, such as DMRG & TEBD

![tensor pic](https://github.com/jcmgray/quimb/blob/HEAD/docs/_static/rand-tensor.svg?raw=true)

---

The core `quimb` module contains tools for reference **'exact'** quantum calculations, where the states and operator are represented as either `numpy.ndarray` or `scipy.sparse` **matrices**. With this you can:

- construct operators in complicated tensor spaces
- find groundstates, excited states and do time evolutions, including with [slepc](https://slepc.upv.es/)
- compute various quantities including entanglement measures
- take advantage of [numba](https://numba.pydata.org) accelerations
- stochastically estimate $\mathrm{Tr}f(X)$ quantities

![matrix pic](https://github.com/jcmgray/quimb/blob/HEAD/docs/_static/rand-herm-matrix.svg?raw=true)

---

The **full documentation** can be found at: [quimb.readthedocs.io](https://quimb.readthedocs.io). Contributions of any sort are very welcome - please see the [contributing guide](https://github.com/jcmgray/quimb/blob/main/.github/CONTRIBUTING.md). [Issues](https://github.com/jcmgray/quimb/issues) and [pull requests](https://github.com/jcmgray/quimb/pulls) are hosted on [github](https://github.com/jcmgray/quimb). For other questions and suggestions, please use the [discussions page](https://github.com/jcmgray/quimb/discussions).