Generating Objects
==================

States
------

- :func:`~quimb.gen.states.basis_vec`
- :func:`~quimb.gen.states.up`
- :func:`~quimb.gen.states.down`
- :func:`~quimb.gen.states.plus`
- :func:`~quimb.gen.states.minus`
- :func:`~quimb.gen.states.yplus`
- :func:`~quimb.gen.states.yminus`
- :func:`~quimb.gen.states.bloch_state`
- :func:`~quimb.gen.states.bell_state`
- :func:`~quimb.gen.states.singlet`
- :func:`~quimb.gen.states.thermal_state`
- :func:`~quimb.gen.states.computational_state`
- :func:`~quimb.gen.states.neel_state`
- :func:`~quimb.gen.states.singlet_pairs`
- :func:`~quimb.gen.states.werner_state`
- :func:`~quimb.gen.states.ghz_state`
- :func:`~quimb.gen.states.w_state`
- :func:`~quimb.gen.states.levi_civita`
- :func:`~quimb.gen.states.perm_state`
- :func:`~quimb.gen.states.graph_state_1d`


Operators
---------

**Gate operators**:

- :func:`~quimb.gen.operators.pauli`
- :func:`~quimb.gen.operators.hadamard`
- :func:`~quimb.gen.operators.phase_gate`
- :func:`~quimb.gen.operators.T_gate`
- :func:`~quimb.gen.operators.S_gate`
- :func:`~quimb.gen.operators.U_gate`
- :func:`~quimb.gen.operators.rotation`
- :func:`~quimb.gen.operators.Rx`
- :func:`~quimb.gen.operators.Ry`
- :func:`~quimb.gen.operators.Rz`
- :func:`~quimb.gen.operators.Xsqrt`
- :func:`~quimb.gen.operators.Ysqrt`
- :func:`~quimb.gen.operators.Zsqrt`
- :func:`~quimb.gen.operators.Wsqrt`
- :func:`~quimb.gen.operators.phase_gate`
- :func:`~quimb.gen.operators.swap`
- :func:`~quimb.gen.operators.iswap`
- :func:`~quimb.gen.operators.fsim`
- :func:`~quimb.gen.operators.fsimg`
- :func:`~quimb.gen.operators.controlled`
- :func:`~quimb.gen.operators.CNOT`
- :func:`~quimb.gen.operators.cX`
- :func:`~quimb.gen.operators.cY`
- :func:`~quimb.gen.operators.cZ`

Most of these are cached (and immutable), so can be called repeatedly without creating any new objects:

.. code-block:: py3

    >>> pauli('Z') is pauli('Z')
    True


**Hamiltonians and related operators**:

- :func:`~quimb.gen.operators.spin_operator`
- :func:`~quimb.gen.operators.ham_heis`
- :func:`~quimb.gen.operators.ham_heis_2D`
- :func:`~quimb.gen.operators.ham_ising`
- :func:`~quimb.gen.operators.ham_XY`
- :func:`~quimb.gen.operators.ham_XXZ`
- :func:`~quimb.gen.operators.ham_j1j2`
- :func:`~quimb.gen.operators.ham_mbl`
- :func:`~quimb.gen.operators.zspin_projector`
- :func:`~quimb.gen.operators.create`
- :func:`~quimb.gen.operators.destroy`
- :func:`~quimb.gen.operators.num`
- :func:`~quimb.gen.operators.ham_hubbard_hardcore`

.. note::

    The Hamiltonians are generally defined using spin operators rather than
    Pauli matrices. Thus for example, the following spin-1/2 Hamiltonians would
    be equivalent

    - in spin-operators:

    .. math::

        \hat{H} = \sum J S^X_i S^X_{i + 1} + B S^Z_i

    - and in Pauli operators (with :math:`S^X=\dfrac{\sigma^X}{2}` etc.):

    .. math::

        \hat{H} = \sum \dfrac{J}{4} \sigma^X_i \sigma^X_{i + 1} + \dfrac{B}{2} \sigma^Z_{i}

    note that interaction terms are scaled different than the single site terms.


Random States & Operators
-------------------------

**Random pure states**:

- :func:`~quimb.gen.rand.rand_ket`
- :func:`~quimb.gen.rand.rand_haar_state`
- :func:`~quimb.gen.rand.gen_rand_haar_states`
- :func:`~quimb.gen.rand.rand_product_state`
- :func:`~quimb.gen.rand.rand_matrix_product_state`
- :func:`~quimb.gen.rand.rand_mera`

**Random operators**:

- :func:`~quimb.gen.rand.rand_matrix`
- :func:`~quimb.gen.rand.rand_herm`
- :func:`~quimb.gen.rand.rand_pos`
- :func:`~quimb.gen.rand.rand_rho`
- :func:`~quimb.gen.rand.rand_uni`
- :func:`~quimb.gen.rand.rand_mix`
- :func:`~quimb.gen.rand.rand_seperable`
- :func:`~quimb.gen.rand.rand_iso`

All of these functions accept a ``seed`` argument for replicability:

.. code-block:: py3

    >>> rand_rho(2, seed=42)
    qarray([[ 0.196764+7.758223e-19j, -0.08442 +2.133635e-01j],
            [-0.08442 -2.133635e-01j,  0.803236-2.691589e-18j]])


    >>> rand_rho(2, seed=42)
    qarray([[ 0.196764+7.758223e-19j, -0.08442 +2.133635e-01j],
            [-0.08442 -2.133635e-01j,  0.803236-2.691589e-18j]])


For some applications, generating random numbers with a single thread can be a bottleneck, though
since version 1.17 ``numpy`` itself enables parallel streams of random numbers to be generated.
``quimb`` handles setting up the bit generators and multi-threading the creation of random arrays, with potentially large performance gains. While the random number sequences can be still replicated using the ``seed`` argument, they also depend (deterministically) on the number of threads used, so may vary across machines unless this is set (e.g. with ``'OMP_NUM_THREADS'``).

.. note::

    Previously, `randomgen <https://github.com/bashtage/randomgen>`_ was needed for this functionality, and its `bit generators <https://bashtage.github.io/randomgen/bit_generators/index.html>`_ can still be specified to :func:`~quimb.gen.rand.set_rand_bitgen` if installed.

The following gives a quick idea of the speed-ups possible. First random, complex, normally distributed array generation with a naive ``numpy`` method:

.. code-block:: py3

    >>> import numpy as np
    >>> %timeit np.random.randn(2**22) + 1j * np.random.randn(2**22)
    394 ms ± 2.93 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


And generation with ``quimb``:

.. code-block:: py3

    >>> import quimb as qu
    >>> %timeit qu.randn(2**22, dtype=complex)
    45.8 ms ± 2.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    >>> # try a randomgen bit generator
    >>> qu.set_rand_bitgen('Xoshiro256')
    >>> %timeit qu.randn(2**22, dtype=complex)
    41.2 ms ± 2.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    >>> # use the default numpy bit generator
    >>> qu.set_rand_bitgen(None)
