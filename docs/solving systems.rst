###############
Solving Systems
###############


Dense / full decomposition
--------------------------

Finding full eigen-spectrum:

    - :func:`~quimb.linalg.base_linalg.eigsys`
    - :func:`~quimb.linalg.base_linalg.eigvals`
    - :func:`~quimb.linalg.base_linalg.eigvecs`
    - :func:`~quimb.linalg.base_linalg.svd`


Partial decomposition
---------------------

Finding extremal eigen-pairs / groundstates:

    - :func:`~quimb.linalg.base_linalg.seigsys`
    - :func:`~quimb.linalg.base_linalg.seigvals`
    - :func:`~quimb.linalg.base_linalg.seigvecs`
    - :func:`~quimb.linalg.base_linalg.groundstate`
    - :func:`~quimb.linalg.base_linalg.groundenergy`
    - :func:`~quimb.linalg.base_linalg.svds`


Internal eigen-solving
~~~~~~~~~~~~~~~~~~~~~~

Targeting mid-spectrum eigen-pairs:

    - ``seigsys(..., sigma=x)`` etc, or
    - :func:`~quimb.linalg.base_linalg.eigsys_window`
    - :func:`~quimb.linalg.base_linalg.eigvals_window`
    - :func:`~quimb.linalg.base_linalg.eigvecs_window`
