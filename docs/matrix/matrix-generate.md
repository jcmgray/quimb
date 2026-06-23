# Generating Objects

## States

- {func}`.basis_vec`
- {func}`.up`
- {func}`.down`
- {func}`.plus`
- {func}`.minus`
- {func}`.yplus`
- {func}`.yminus`
- {func}`.bloch_state`
- {func}`.bell_state`
- {func}`.singlet`
- {func}`.thermal_state`
- {func}`.computational_state`
- {func}`.neel_state`
- {func}`.singlet_pairs`
- {func}`.werner_state`
- {func}`.ghz_state`
- {func}`.w_state`
- {func}`.levi_civita`
- {func}`.perm_state`
- {func}`.graph_state_1d`

## Operators

**Gate operators**:

- {func}`.pauli`
- {func}`.hadamard`
- {func}`.phase_gate`
- {func}`.T_gate`
- {func}`.S_gate`
- {func}`.U_gate`
- {func}`.rotation`
- {func}`.Rx`
- {func}`.Ry`
- {func}`.Rz`
- {func}`.Xsqrt`
- {func}`.Ysqrt`
- {func}`.Zsqrt`
- {func}`.Wsqrt`
- {func}`.phase_gate`
- {func}`.swap`
- {func}`.iswap`
- {func}`.fsim`
- {func}`.fsimg`
- {func}`.controlled`
- {func}`.CNOT`
- {func}`.cX`
- {func}`.cY`
- {func}`.cZ`

Most of these are cached (and immutable), so can be called repeatedly without creating any new objects:

```python
>>> pauli('Z') is pauli('Z')
True
```

**Hamiltonians and related operators**:

- {func}`.spin_operator`
- {func}`.ham_heis`
- {func}`.ham_heis_2D`
- {func}`.ham_ising`
- {func}`.ham_XY`
- {func}`.ham_XXZ`
- {func}`.ham_j1j2`
- {func}`.ham_mbl`
- {func}`.zspin_projector`
- {func}`.create`
- {func}`.destroy`
- {func}`.num`
- {func}`.ham_hubbard_hardcore`

:::{note}
The Hamiltonians are generally defined using spin operators rather than
Pauli matrices. Thus for example, the following spin-1/2 Hamiltonians would
be equivalent

- in spin-operators:

$$
\hat{H} = \sum J S^X_i S^X_{i + 1} + B S^Z_i
$$

- and in Pauli operators (with $S^X=\dfrac{\sigma^X}{2}$ etc.):

$$
\hat{H} = \sum \dfrac{J}{4} \sigma^X_i \sigma^X_{i + 1} + \dfrac{B}{2} \sigma^Z_{i}
$$

note that interaction terms are scaled different than the single site terms.
:::

## Random States & Operators

**Random pure states**:

- {func}`.rand_ket`
- {func}`.rand_haar_state`
- {func}`.gen_rand_haar_states`
- {func}`.rand_product_state`
- {func}`.rand_matrix_product_state`
- {func}`.rand_mera`

**Random operators**:

- {func}`.rand_matrix`
- {func}`.rand_herm`
- {func}`.rand_pos`
- {func}`.rand_rho`
- {func}`.rand_uni`
- {func}`.rand_mix`
- {func}`.rand_seperable`
- {func}`.rand_iso`

All of these functions accept a `seed` argument for replicability:

```python
>>> rand_rho(2, seed=42)
qarray([[ 0.196764+7.758223e-19j, -0.08442 +2.133635e-01j],
        [-0.08442 -2.133635e-01j,  0.803236-2.691589e-18j]])


>>> rand_rho(2, seed=42)
qarray([[ 0.196764+7.758223e-19j, -0.08442 +2.133635e-01j],
        [-0.08442 -2.133635e-01j,  0.803236-2.691589e-18j]])
```

For some applications, generating random numbers with a single thread can be a bottleneck, though
since version 1.17 `numpy` itself enables parallel streams of random numbers to be generated.
`quimb` handles setting up the bit generators and multi-threading the creation of random arrays, with potentially large performance gains. While the random number sequences can be still replicated using the `seed` argument, they also depend (deterministically) on the number of threads used, so may vary across machines unless this is set (e.g. with `'OMP_NUM_THREADS'`).

:::{note}
Previously, [randomgen](https://github.com/bashtage/randomgen) was needed for this functionality, and its [bit generators](https://bashtage.github.io/randomgen/bit_generators/index.html) can still be specified to {func}`.set_rand_bitgen` if installed.
:::

The following gives a quick idea of the speed-ups possible. First random, complex, normally distributed array generation with a naive `numpy` method:

```python
>>> import numpy as np
>>> %timeit np.random.randn(2**22) + 1j * np.random.randn(2**22)
394 ms ± 2.93 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

And generation with `quimb`:

```python
>>> import quimb as qu
>>> %timeit qu.randn(2**22, dtype=complex)
45.8 ms ± 2.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> # try a randomgen bit generator
>>> qu.set_rand_bitgen('Xoshiro256')
>>> %timeit qu.randn(2**22, dtype=complex)
41.2 ms ± 2.02 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> # use the default numpy bit generator
>>> qu.set_rand_bitgen(None)
```
