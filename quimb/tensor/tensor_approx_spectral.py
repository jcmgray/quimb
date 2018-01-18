import numpy as np

from .tensor_gen import MPO_rand, MPO_zeros_like


def construct_lanczos_tridiag_MPO(
        A,
        K,
        v0=None,
        initial_bond_dim=None,
        beta_tol=1e-6,
        max_bond=None,
        seed=None):
    """
    """
    if initial_bond_dim is None:
        initial_bond_dim = 2
    if max_bond is None:
        max_bond = int(A.max_bond()**1)

    if v0 is None:
        V = MPO_rand(A.nsites, initial_bond_dim, phys_dim=A.phys_dim(0))
    else:  # normalize
        V = v0 / (v0.H @ v0)**0.5
    Vm1 = MPO_zeros_like(V)

    alpha = np.zeros(K + 1)
    beta = np.zeros(K + 2)

    bsz = A.phys_dim()**A.nsites
    beta[1] = A.phys_dim()**A.nsites

    compress_kws = {'max_bond': max_bond}

    for j in range(1, K + 1):

        Vt = A.apply(V, compress=True, **compress_kws)
        Vt.add_MPO(-beta[j] * Vm1, inplace=True, compress=True, **compress_kws)
        alpha[j] = (V.H @ Vt).real
        Vt.add_MPO(-alpha[j] * V, inplace=True, compress=True, **compress_kws)
        beta[j + 1] = (Vt.H @ Vt)**0.5

        # check for convergence
        if abs(beta[j + 1]) < beta_tol:
            yield alpha[1:j + 1], beta[2:j + 2], beta[1]**2 / bsz
            break

        Vm1 = V.copy()
        V = Vt / beta[j + 1]

        if j > 3:
            yield (np.copy(alpha[1:j + 1]),
                   np.copy(beta[2:j + 2]),
                   np.copy(beta[1])**2 / bsz)
