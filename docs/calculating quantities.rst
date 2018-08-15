######################
Calculating Quantities
######################

There are various built-in functions to calculate quantities, not limited to:

- :func:`~quimb.calc.fidelity`
- :func:`~quimb.calc.purify`
- :func:`~quimb.calc.dephase`
- :func:`~quimb.calc.entropy`
- :func:`~quimb.calc.mutinf`
- :func:`~quimb.calc.mutinf_subsys`
- :func:`~quimb.calc.schmidt_gap`
- :func:`~quimb.calc.tr_sqrt`
- :func:`~quimb.calc.partial_transpose`
- :func:`~quimb.calc.logneg`
- :func:`~quimb.calc.logneg_subsys`
- :func:`~quimb.calc.negativity`
- :func:`~quimb.calc.concurrence`
- :func:`~quimb.calc.one_way_classical_information`
- :func:`~quimb.calc.quantum_discord`
- :func:`~quimb.calc.trace_distance`
- :func:`~quimb.calc.decomp`
- :func:`~quimb.calc.correlation`
- :func:`~quimb.calc.pauli_correlations`
- :func:`~quimb.calc.ent_cross_matrix`
- :func:`~quimb.calc.is_degenerate`
- :func:`~quimb.calc.is_eigenvector`
- :func:`~quimb.calc.page_entropy`
- :func:`~quimb.calc.heisenberg_energy`


Approximate Spectral Functions
==============================

The module :py:mod:`~quimb.linalg.approx_spectral`, contains a Lanczos method for estimating any quantities of the form ``tr(fn(A))``. Where ``A`` is any operator that implements a dot product with a vector. For example, estimating the trace of the sqrt of a matrix would naievly require diagonalising it:

.. code-block:: py3

    >>> rho = rand_rho(2**12)
    >>> np.sum(np.sqrt(eigvalsh(rho)))
    54.324631408257559

    >>> tr_sqrt_approx(rho)
    54.27572830646708

Diagonalization has a cost of ``O(n^3)``, which is essentially reduced to ``O(k * n^2)`` for this stochastic method. For a general function :func:`~quimb.linalg.approx_spectral.approx_spectral_function` can be used.

However, the real advantage occurs when the full matrix does not need to be fully represented, e.g. in the case of 'partial trace states'. One can then calculate quantities for subsystems that would not be possible to explicitly represent.

For example, the partial trace, followed by partial transpose, followed by vector multiplication can be 'lazily' evaluated as a tensor contraction (see :py:func:`~quimb.linalg.approx_spectral.lazy_ptr_ppt_dot`). In this way the logarithmic negativity of subsytems can be efficiently calculated:

.. code-block:: py3

    >>> psi = rand_ket(2**20)
    >>> dims = [2**8, 2**4, 2**8]
    >>> logneg_subsys_approx(psi, dims, sysa=0, sysb=2)
    5.742612642373124

The above takes a few seconds, but explicitly diagonalising the 16 qubit reduced state (a 65536x65536 matrix) would take hours if not days.

Under the hood, this method makes use of :mod:`quimb.tensor` functionality, which allows `various tensor contraction backends <http://optimized-einsum.readthedocs.io/en/latest/backends.html>`_ to be used (see :func:`~quimb.tensor.tensor_core.set_tensor_linop_backend`). These types of computation are particularly suited to the GPU and therefore if `cupy <https://cupy.chainer.org>`_ is installed it will be used automatically:

.. code-block:: py3

    >>> %timeit qu.logneg_subsys_approx(psi, dims, sysa=0, sysb=2, backend='numpy')
    726 ms ± 223 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> %timeit qu.logneg_subsys_approx(psi, dims, sysa=0, sysb=2, backend='cupy')
    85.9 ms ± 22.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
