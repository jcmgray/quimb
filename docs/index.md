<img src="_static/kagome-contract-treeset-2.png" alt="quimb" width="800px">

# Welcome to quimb's documentation!

[![Tests](https://github.com/jcmgray/quimb/actions/workflows/tests.yml/badge.svg)](https://github.com/jcmgray/quimb/actions/workflows/tests.yml)
[![Code Coverage](https://codecov.io/gh/jcmgray/quimb/branch/main/graph/badge.svg)](https://codecov.io/gh/jcmgray/quimb)
[![Code Quality](https://app.codacy.com/project/badge/Grade/3c7462a3c45f41fd9d8f0a746a65c37c)](https://www.codacy.com/gh/jcmgray/quimb/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jcmgray/quimb&amp;utm_campaign=Badge_Grade)
[![Documentation Status](https://readthedocs.org/projects/quimb/badge/?version=latest)](http://quimb.readthedocs.io/en/latest/?badge=latest)
[![JOSS Paper](http://joss.theoj.org/papers/10.21105/joss.00819/status.svg)](https://doi.org/10.21105/joss.00819)
[![PyPI](https://img.shields.io/pypi/v/quimb?color=teal)](https://pypi.org/project/quimb/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/quimb/badges/version.svg)](https://anaconda.org/conda-forge/quimb)

[`quimb`](https://github.com/jcmgray/quimb) is an easy but fast python library for *'quantum information many-body'* calculations, focusing primarily on **tensor networks**. The code is hosted on [github](https://github.com/jcmgray/quimb), and docs are hosted on [readthedocs](http://quimb.readthedocs.io/en/latest/). Functionality is split in two:

::::{grid} 2

:::{grid-item-card} Tensor module
:img-bottom: _static/rand-tensor.svg

The `quimb.tensor` module contains tools for working with **tensors and tensor networks**. It has a particular focus on automatically handling arbitrary geometry, e.g. beyond 1D and 2D lattices. With this you can:

- construct and manipulate arbitrary (hyper) graphs of tensor networks
- automatically [contract](https://cotengra.readthedocs.io), optimize and draw networks
- use various backend array libraries such as [jax](https://jax.readthedocs.io) and [torch](https://pytorch.org/) via [autoray](https://github.com/jcmgray/autoray/)
- run specific MPS, PEPS, MERA and quantum circuit algorithms, such as DMRG & TEBD
:::

:::{grid-item-card} Matrix module
:img-bottom: _static/rand-herm-matrix.svg

The core `quimb` module contains tools for reference **'exact'** quantum calculations, where the states and operator are represented as either `numpy.ndarray` or `scipy.sparse` **matrices**. With this you can:

- construct operators in complicated tensor spaces
- find groundstates, excited states and do time evolutions, including with [slepc](https://slepc.upv.es/)
- compute various quantities including entanglement measures
- take advantage of [numba](https://numba.pydata.org) accelerations
- stochastically estimate $\mathrm{Tr}f(X)$ quantities
:::

::::


(guides)=
## User Guide

The following guides give a basic introduction to the various parts:

```{toctree}
:caption: Guides
:maxdepth: 1
installation
```
```{toctree}
:maxdepth: 2
index_tn
index_matrix
```


(examples)=
## Examples

The following examples, generated from the notebooks in ``docs/examples``,
demonstrate some more advanced features or complete usage:

```{toctree}
:maxdepth: 2
index_examples
```


(citing)=
## Citing

``quimb`` is published in the Journal of Open Source Software
[here](https://doi.org/10.21105/joss.00819) - if it's ever useful in research
please consider supporting development by citing it!

```latex
@article{gray2018quimb,
    title={quimb: a python library for quantum information and many-body calculations},
    author={Gray, Johnnie},
    journal={Journal of Open Source Software},
    year = {2018},
    volume={3}, number={29}, pages={819},
    doi={10.21105/joss.00819},
}
```


(development)=
## Notes

Development notes, including on contributing to ``quimb`` and release details
can be found below:

```{toctree}
:caption: Development

changelog
develop
```

```{toctree}
:hidden:

GitHub Repository <https://github.com/jcmgray/quimb>
```