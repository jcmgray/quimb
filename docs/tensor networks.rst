###############
Tensor Networks
###############


Requirements
~~~~~~~~~~~~

To perform groups of tensor contractions efficiently and decomposed into BLAS operations, `opt_einsum <https://github.com/dgasmith/opt_einsum>`_ is highly recommended, although this functionality should be in ``numpy`` soon (v1.14+).


Features
~~~~~~~~

- Auto optimized tensor contractions using BLAS for small to medium groups of tensors - no need to ever think about contraction order, best scaling etc.
- Completely geometry free underlying representation of networks.
- Multiple Tensors and Networks can be views of the same underlying data to save memory.
- Fully unit tested.


Currently implemented algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Static:

    - 1-site :class:`~quimb.tensor.DMRG1`
    - 2-site :class:`~quimb.tensor.DMRG2`
    - 1-site :class:`~quimb.tensor.DMRGX`

Should be fairly easily to implement / planned:

    - 2-site DMRG-X
    - 1-site TDVP
    - 2-site TDVP


Basic Manipulations
-------------------

pass


Building Hamiltonians
---------------------

pass


Example of DMRG2 calcuation
---------------------------

.. code-block:: python

    In [1]: from quimb.tensor import *

    In [2]: builder = MPOSpinHam(S=1)

    In [3]: builder.add_term(0.5, '+', '-')

    In [4]: builder.add_term(0.5, '-', '+')

    In [5]: builder.add_term(1.0, 'Z', 'Z')

    In [6]: ham = builder.build(n=100)

    In [7]: dmrg = DMRG2(ham, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)

    In [8]: dmrg.solve(tol=1e-6, verbose=True)
    SWEEP-1, direction=R, max_bond=10, cutoff:1e-10
    100%|███████████████████████████████████████████| 99/99 [00:01<00:00, 75.66it/s]
    Energy: -138.73797893126138 ... not converged
    SWEEP-2, direction=R, max_bond=20, cutoff:1e-10
    100%|██████████████████████████████████████████| 99/99 [00:00<00:00, 442.40it/s]
    Energy: -138.93684387336182 ... not converged
    SWEEP-3, direction=R, max_bond=100, cutoff:1e-10
    100%|███████████████████████████████████████████| 99/99 [00:01<00:00, 53.31it/s]
    Energy: -138.9400480376106 ... not converged
    SWEEP-4, direction=R, max_bond=100, cutoff:1e-10
    100%|███████████████████████████████████████████| 99/99 [00:09<00:00, 10.24it/s]
    Energy: -138.9400856058551 ... not converged
    SWEEP-5, direction=R, max_bond=200, cutoff:1e-10
    100%|███████████████████████████████████████████| 99/99 [00:15<00:00,  6.36it/s]
    Energy: -138.9400860644765 ... converged!
    Out[8]: True

    In [9]: dmrg.state.show()
         3 9 27 55 65 74 79 84 87 89 91 93 94 95 95 95 95 94 94 94 93 93 92 92 91 91 90 90 90 90 90 90 90 90 90 90 90 90 90 90 9
        >->->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->- ...
        | | |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
                                                              ...
        0 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 91 91 90 91 91 91 9
    ... ->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->- ...
         |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
                                                              ...
        2 95 96 96 96 96 96 95 92 90 87 83 78 73 64 53 27 9 3
    ... ->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->->-o
         |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | | |
