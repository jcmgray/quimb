###############
Solving Systems
###############


Dense / full decomposition
--------------------------

Finding full eigen-spectrum:

    - :func:`~quimb.linalg.base_linalg.eigensystem`
    - :func:`~quimb.linalg.base_linalg.eig`
    - :func:`~quimb.linalg.base_linalg.eigh`
    - :func:`~quimb.linalg.base_linalg.eigvals`
    - :func:`~quimb.linalg.base_linalg.eigvalsh`
    - :func:`~quimb.linalg.base_linalg.eigvecs`
    - :func:`~quimb.linalg.base_linalg.eigvecsh`
    - :func:`~quimb.linalg.base_linalg.svd`


Partial decomposition
---------------------

Finding extremal eigen-pairs / groundstates:

    - :func:`~quimb.linalg.base_linalg.eigensystem_partial`
    - :func:`~quimb.linalg.base_linalg.eig` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigh` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvals` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvalsh` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigenvectors` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvecs` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.eigvecsh` with ``(k > 0)``
    - :func:`~quimb.linalg.base_linalg.groundstate`
    - :func:`~quimb.linalg.base_linalg.groundenergy`
    - :func:`~quimb.linalg.base_linalg.svds`


Internal eigen-solving
~~~~~~~~~~~~~~~~~~~~~~

Targeting mid-spectrum eigen-pairs:

    - ``eigh(..., k > 0, sigma=x)`` etc, or
    - :func:`~quimb.linalg.base_linalg.eigh_window`
    - :func:`~quimb.linalg.base_linalg.eigvalsh_window`
    - :func:`~quimb.linalg.base_linalg.eigvecsh_window`
