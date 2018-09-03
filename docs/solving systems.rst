####################################
Eigen-Solving & Other Linear Algebra
####################################


Dense / full decomposition
--------------------------

Currently full decompositions use numpy. They are as follows:

    - :func:`~quimb.linalg.base_linalg.eig`
    - :func:`~quimb.linalg.base_linalg.eigh`
    - :func:`~quimb.linalg.base_linalg.eigvals`
    - :func:`~quimb.linalg.base_linalg.eigvalsh`
    - :func:`~quimb.linalg.base_linalg.eigvecs`
    - :func:`~quimb.linalg.base_linalg.eigvecsh`
    - :func:`~quimb.linalg.base_linalg.eigensystem`
    - :func:`~quimb.linalg.base_linalg.svd`


Partial decomposition
---------------------

Partial decompositions are mostly just specified by  supplying the ``k`` kwarg to the above functions. These also take a ``backend`` argument which can be one of:

- ``'scipy'``: is generally reliable
- ``'numpy'``: can be faster for small or dense problems
- ``'lobpcg'``: useful for fast, low accruacy generalized eigenproblems (like periodic DMRG)
- ``'slepc'``: Usually the fastest for large problems, with many options. Will either spawn MPI
  workers or should be used in ``syncro`` mode.
- ``'slepc-nompi'``: like ``'slepc'``, but performs computation in the single, main process.
- ``'AUTO'`` - choose a good backend, the default.

The possible partical decompositions are:

    - :func:`~quimb.linalg.base_linalg.eig` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigh` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvals` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvalsh` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigenvectors` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvecs` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvecsh` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.groundstate`
    - :func:`~quimb.linalg.base_linalg.groundenergy`
    - :func:`~quimb.linalg.base_linalg.eigensystem_partial`
    - :func:`~quimb.linalg.base_linalg.svds`

So for example the :func:`~quimb.linalg.base_linalg.groundstate` function
for a Hamiltonian ``H`` is an alias to:

.. code:: python

    psi = eigvecsh(H, k=1, which='sa')

[find eigenvectors, Hermitian operator (``h`` post-fix), get ``k=1`` eigenstate,
and target the '(s)mallest (a)lgebraic' eigenvalue].


Interior eigen-solving
~~~~~~~~~~~~~~~~~~~~~~

SLEPc is highly recommended for performing these using 'shift-invert'.
See the following functions:

    - ``eigh(..., k=k, sigma=x)`` with ``k > 0`` etc., or
    - :func:`~quimb.linalg.base_linalg.eigh_window`
    - :func:`~quimb.linalg.base_linalg.eigvalsh_window`
    - :func:`~quimb.linalg.base_linalg.eigvecsh_window`

With the last three allowing the specification of a window *relative* to the total spectrum of the operator.


Fast Randomized Linear Algebra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``quimb`` has an implementation of a fast randomized SVD - :func:`~quimb.linalg.rand_linalg.rsvd` -
that can be significantly quicker than :func:`~quimb.linalg.base_linalg.svd` or :func:`~quimb.linalg.base_linalg.svds`,
especially for large ``k``. This might be useful for e.g. tensor network linear operator decompositions.
It can perform the SVD rank-adaptively, which allows the efficient estimation of an operator's rank,
see :func:`~quimb.linalg.rand_linalg.estimate_rank`.
