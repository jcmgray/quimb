######################
Calculating Quantities
######################


Approximate Spectral Functions
==============================

The module :py:mod:`~quimb.linalg.approx_spectral`, contains a Lanczos method for estimating any quantities of the form ``tr(fn(A))``. Where ``A`` is any operator that implements a dot product with a vector. For example, estimating the trace of the sqrt of a matrix would naievly require diagonalising it:

.. code:: python

    >>> rho = rand_rho(2**12)
    >>> np.sum(np.sqrt(eigvals(rho)))
    54.324631408257559

    >>> tr_sqrt_approx(rho)
    54.27572830646708

Diagonalization has a cost of ``O(n^3)``, which is essentially reduced to ``O(k * n^2)`` for this stochastic method. For a general function :func:`~quimb.linalg.approx_spectral.approx_spectral_function` can be used.

However, the real advantage occurs when the full matrix does not need to be fully represented, e.g. in the case of 'partial trace states'. One can then calculate quantities for subsystems that would not be possible to explicitly represent.

For example, the partial trace, followed by partial transpose, followed by vector multiplication can be 'lazily' evaluated as a tensor contraction (see :py:func:`~quimb.linalg.approx_spectral.lazy_ptr_ppt_dot`). In this way the logarithmic negativity of subsytems can be efficiently calculated:

.. code:: python

    >>> psi = rand_ket(2**20)
    >>> dims = [2**8, 2**4, 2**8]
    >>> logneg_subsys_approx(psi, dims, sysa=0, sysb=2)
    5.742612642373124

The above takes a few seconds, but explicitly diagonalising the 16 qubit reduced state (a 65536x65536 matrix) would take hours if not days.
