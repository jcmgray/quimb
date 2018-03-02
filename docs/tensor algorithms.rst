#################
Tensor Algorithms
#################


1D tensor networks
------------------

Generate a random :class:`~quimb.tensor.tensor_1d.MatrixProductState`, and contract its inner product:

.. code-block:: python

    >>> p = MPS_rand_state(n=30, bond_dim=50)
    >>> p.H @ p
    1.0000000000000009
    >>> p.show()
     50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
    o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o--o
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

    >>> p.left_canonize()
    >>> p.show()
     2 4 8 16 32 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
    >->->->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->--o
    | | | |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

    >>> p.H @ p
    0.9999999999999991

Add MPS and compress:

.. code-block:: python

    >>> p2 = (p + p) / 2
    >>> p2.show()
     100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 10
    o===o===o===o===o===o===o===o===o===o===o===o===o===o===o===o===o===o===o===o== ...
    |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |

    >>> p2.compress(form=20)
    >>> p2.show()
     2 4 8 16 32 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 32 16 8 4 2
    >->->->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->-->--o--<--<--<--<--<--<-<-<-<
    | | | |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | | | |

    >>> p2.H @ p2
    0.9999999999999998

Find the overlap with a random hermitian :class:`~quimb.tensor.tensor_1d.MatrixProductOperator`

.. code-block:: python

    >>> A = MPO_rand_herm(30, bond_dim=7)
    >>> pH = p.H
    >>> align_TN_1D(pH, A, p, inplace=True);
    >>> (pH & A & p) ^ ...
    -1.2069781127179028e-29


Building Hamiltonians
---------------------

See :class:`~quimb.tensor.tensor_gen.MPOSpinHam`.


Example of DMRG2 calcuation
---------------------------

Build a Hamiltonian term by term and setup a DMRG solver:

.. code-block:: python

    In [1]: from quimb.tensor import *

    In [2]: builder = MPOSpinHam(S=1)

    In [3]: builder.add_term(1/2, '+', '-')

    In [4]: builder.add_term(1/2, '-', '+')

    In [5]: builder.add_term(1, 'Z', 'Z')

    In [6]: ham = builder.build(n=100)

    In [7]: dmrg = DMRG2(ham, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)


Now solve to a certain relative energy tolerance, showing progress and a schematic of the final state:

.. code-block:: guess

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


Performance tips
----------------

    1. Make sure numpy is linked to a fast BLAS (e.g. MKL version that comes with conda).
    2. Install slepc4py, to use as the iterative eigensolver, it's faster than scipy.
    3. If the hamiltonian is real, compile and use a real version of SLEPC (set the environment variable PETSC_ARCH before launch).
