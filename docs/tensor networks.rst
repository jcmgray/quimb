###############
Tensor Networks
###############


The tensor functionality is stored in ``quimb.tensor`` and not imported by default.

.. code-block:: python

    >>> from quimb import *
    >>> from quimb.tensor import *

The core functions of note are :class:`~quimb.tensor.tensor_core.Tensor`, :class:`~quimb.tensor.tensor_core.TensorNetwork`, :func:`~quimb.tensor.tensor_core.tensor_contract`, and :func:`~quimb.tensor.tensor_core.tensor_split`.


Requirements
~~~~~~~~~~~~

To perform groups of tensor contractions efficiently and decomposed into BLAS operations, `opt_einsum <https://github.com/dgasmith/opt_einsum>`_ is required (possibly a recent github version), although this functionality should be in ``numpy`` soon (v1.14+).


Features
~~~~~~~~

- Auto optimized tensor contractions using BLAS for expressions using 100s of tensors - no need to ever think about contraction order, best scaling etc.
- Completely geometry free underlying representation of networks.
- Plot the graph of any tensor network - with bond dimension edge weighting and other coloring.
- Multiple Tensors and Networks can be views of the same underlying data to save memory.
- Treat any network as a :class:`scipy.sparse.linalg.LinearOperator` for solving, factorizing etc.
- Form compressed density matrices from PBC or OBC MPS states, and thus compute entanglement measures (e.g. :meth:`~quimb.tensor.tensor_1d.MatrixProductState.logneg_subsys`).
- Fully unit tested.


Currently implemented algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static:

    - 1-site :class:`~quimb.tensor.tensor_algo_static.DMRG1`
    - 2-site :class:`~quimb.tensor.tensor_algo_static.DMRG2`
    - 1-site :class:`~quimb.tensor.tensor_algo_static.DMRGX`

Should be fairly easily to implement / planned:

    - 2-site DMRG-X
    - 1-site TDVP
    - 2-site TDVP
