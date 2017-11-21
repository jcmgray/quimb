###############
Tensor Networks
###############


Requirements
~~~~~~~~~~~~

To perform groups of tensor contractions efficiently and decomposed into BLAS operations, `opt_einsum <https://github.com/dgasmith/opt_einsum>`_ is highly recommended, although this functionality should be in ``numpy`` soon (v1.14+).


Features
~~~~~~~~

- Auto optimized tensor contractions using BLAS for small to medium groups of tensors - no need to ever think about contraction order, best scaling etc.
- Completely geometry free representation of networks.
- Multiple Tensors and Networks can be views of the same underlying data to save memory.
- Fully unit tested.


Currently implemented algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static:

    - 1-site :class:`~quimb.tensor.DMRG1`
    - 1-site :class:`~quimb.tensor.DMRGX`

Should be fairly easily to implement / planned:

    - 2-site DMRG
    - 2-site DMRG-X
    - 1-site TDVP
    - 2-site TDVP


Basic Manipulations
-------------------

pass


Building Hamiltonians
---------------------

pass
