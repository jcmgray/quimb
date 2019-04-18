import numpy as np
import scipy.linalg as scla
import scipy.sparse.linalg as spla
import scipy.linalg.interpolative as sli
from autoray import do

from ..core import njit
from ..linalg.rand_linalg import rsvd, estimate_rank


@njit(['i4(f4[:], f4, i4)', 'i4(f8[:], f8, i4)'])  # pragma: no cover
def _trim_singular_vals(s, cutoff, cutoff_mode):
    """Find the number of singular values to keep of ``s`` given ``cutoff`` and
    ``cutoff_mode``.

    Parameters
    ----------
    s : array
        Singular values.
    cutoff : float
        Cutoff.
    cutoff_mode : {1, 2, 3}
        How to perfrm the trim:

            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
    """
    if cutoff_mode == 1:
        n_chi = np.sum(s > cutoff)

    elif cutoff_mode == 2:
        n_chi = np.sum(s > cutoff * s[0])

    elif (cutoff_mode == 3) or (cutoff_mode == 4):

        target = cutoff
        if cutoff_mode == 4:
            target *= np.sum(s**2)

        n_chi = s.size
        s2s = 0.0
        for i in range(s.size - 1, -1, -1):
            s2 = s[i]**2
            if not np.isnan(s2):
                s2s += s2
            if s2s > target:
                break
            n_chi -= 1
    else:
        raise ValueError("``cutoff_mode`` not one of {1, 2, 3, 4}.")

    return max(n_chi, 1)


@njit(['f4(f4[:], i4)', 'f8(f8[:], i4)'])  # pragma: no cover
def _renorm_singular_vals(s, n_chi):
    """Find the normalization constant for ``s`` such that the new sum squared
    of the ``n_chi`` largest values equals the sum squared of all the old ones.
    """
    s_tot_keep = 0.0
    s_tot_lose = 0.0
    for i in range(s.size):
        s2 = s[i]**2
        if not np.isnan(s2):
            if i < n_chi:
                s_tot_keep += s2
            else:
                s_tot_lose += s2
    return ((s_tot_keep + s_tot_lose) / s_tot_keep)**0.5


@njit  # pragma: no cover
def _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb):
    if cutoff > 0.0:
        n_chi = _trim_singular_vals(s, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            norm = _renorm_singular_vals(s, n_chi)
            s = s[:n_chi] * norm
            U = U[..., :n_chi]
            V = V[:n_chi, ...]

    s = np.ascontiguousarray(s)

    if absorb == -1:
        U *= s.reshape((1, -1))
    elif absorb == 1:
        V *= s.reshape((-1, 1))
    else:
        s **= 0.5
        U *= s.reshape((1, -1))
        V *= s.reshape((-1, 1))

    return U, V


@njit  # pragma: no cover
def _svd_nb(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition.
    """
    U, s, V = np.linalg.svd(x, full_matrices=False)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _svd_alt(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decompt using alternate scipy driver.
    """
    U, s, V = scla.svd(x, full_matrices=False, lapack_driver='gesvd')
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _svd(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    args = (x, cutoff, cutoff_mode, max_bond, absorb)

    try:
        return _svd_nb(*args)

    except (scla.LinAlgError, ValueError) as e:  # pragma: no cover

        if isinstance(e, scla.LinAlgError) or 'converge' in str(e):
            import warnings
            warnings.warn("TN SVD failed, trying again with alternate driver.")

            return _svd_alt(*args)

        raise e


def _svdvals(x):
    """SVD-decomposition, but return singular values only.
    """
    return np.linalg.svd(x, full_matrices=False, compute_uv=False)


@njit  # pragma: no cover
def dag(x):
    """Hermitian conjugate.
    """
    return np.conjugate(x).T


@njit  # pragma: no cover
def _eig(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-split via eigen-decomposition.
    """
    if x.shape[0] > x.shape[1]:
        # Get sU, V
        s2, V = np.linalg.eigh(dag(x) @ x)
        U = x @ V
        V = dag(V)

        # Check if want U, sV
        if absorb == 1:
            s = s2**0.5
            U /= s.reshape((1, -1))
            V *= s.reshape((-1, 1))
        # Or sqrt(s)U, sqrt(s)V
        elif absorb == 0:
            sqrts = s2**0.25
            U /= sqrts.reshape((1, -1))
            V *= sqrts.reshape((-1, 1))

    else:
        # Get U, sV
        s2, U = np.linalg.eigh(x @ dag(x))
        V = dag(U) @ x

        # Check if want sU, V
        if absorb == -1:
            s = s2**0.5
            U *= s.reshape((1, -1))
            V /= s.reshape((-1, 1))

        # Or sqrt(s)U, sqrt(s)V
        elif absorb == 0:
            sqrts = s2**0.25
            U *= sqrts.reshape((1, -1))
            V /= sqrts.reshape((-1, 1))

    # eigh produces ascending eigenvalue order -> slice opposite to svd
    if cutoff > 0.0:
        s = s2[::-1]**0.5
        n_chi = _trim_singular_vals(s, cutoff, cutoff_mode)

        if max_bond > 0:
            n_chi = min(n_chi, max_bond)

        if n_chi < s.size:
            norm = _renorm_singular_vals(s, n_chi)
            U = U[..., -n_chi:]
            V = V[-n_chi:, ...]

            if absorb == -1:
                U *= norm
            elif absorb == 0:
                U *= norm**0.5
                V *= norm**0.5
            else:
                V *= norm

    return U, V


@njit
def _svdvals_eig(x):  # pragma: no cover
    """SVD-decomposition via eigen, but return singular values only.
    """
    if x.shape[0] > x.shape[1]:
        s2 = np.linalg.eigvalsh(dag(x) @ x)
    else:
        s2 = np.linalg.eigvalsh(x @ dag(x))
    return s2**0.5


@njit  # pragma: no cover
def _eigh(x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition, using hermitian eigen-decomposition, only works if
    ``x`` is hermitian.
    """
    s, U = np.linalg.eigh(x)
    s, U = s[::-1], U[:, ::-1]  # make sure largest singular value first

    V = np.sign(s).reshape(-1, 1) * dag(U)
    s = np.abs(s)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


@njit  # pragma: no cover
def _numba_cholesky(x, cutoff=-1, cutoff_mode=3, max_bond=-1, absorb=0):
    """SVD-decomposition, using cholesky decomposition, only works if
    ``x`` is positive definite.
    """
    L = np.linalg.cholesky(x)
    return L, dag(L)


def _cholesky(x, cutoff=-1, cutoff_mode=3, max_bond=-1, absorb=0):
    try:
        return _numba_cholesky(x, cutoff, cutoff_mode, max_bond, absorb)
    except np.linalg.LinAlgError as e:
        if cutoff < 0:
            raise e
        # try adding cutoff identity - assuming it is approx allowable error
        xi = x + 2 * cutoff * np.eye(x.shape[0])
        return _numba_cholesky(xi, cutoff, cutoff_mode, max_bond, absorb)


def _choose_k(x, cutoff, max_bond):
    """Choose the number of singular values to target.
    """
    d = min(x.shape)

    if cutoff != 0.0:
        k = estimate_rank(x, cutoff, k_max=None if max_bond < 0 else max_bond)
    else:
        k = min(d, max_bond)

    # if computing more than half of spectrum then just use dense method
    return 'full' if k > d // 2 else k


def _svds(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using iterative methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == 'full':
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return _svd(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, V = spla.svds(x, k=k)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _isvd(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using interpolative matrix random methods. Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == 'full':
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return _svd(x, cutoff, cutoff_mode, max_bond, absorb)

    U, s, V = sli.svd(x, k)
    V = dag(V)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _rsvd(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using randomized methods (due to Halko). Allows the
    computation of only a certain number of singular values, e.g. max_bond,
    from the get-go, and is thus more efficient. Can also supply
    ``scipy.sparse.linalg.LinearOperator``.
    """
    U, s, V = rsvd(x, cutoff)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


def _eigsh(x, cutoff=0.0, cutoff_mode=2, max_bond=-1, absorb=0):
    """SVD-decomposition using iterative hermitian eigen decomp, thus assuming
    that ``x`` is hermitian. Allows the computation of only a certain number of
    singular values, e.g. max_bond, from the get-go, and is thus more
    efficient. Can also supply ``scipy.sparse.linalg.LinearOperator``.
    """
    k = _choose_k(x, cutoff, max_bond)

    if k == 'full':
        if not isinstance(x, np.ndarray):
            x = x.to_dense()
        return _eigh(x, cutoff, cutoff_mode, max_bond, absorb)

    s, U = spla.eigsh(x, k=k)
    s, U = s[::-1], U[:, ::-1]  # make sure largest singular value first
    V = np.sign(s).reshape(-1, 1) * dag(U)
    s = np.abs(s)
    return _trim_and_renorm_SVD(U, s, V, cutoff, cutoff_mode, max_bond, absorb)


@njit  # pragma: no cover
def _qr_numba(x):
    """QR-decomposition.
    """
    Q, R = np.linalg.qr(x)
    return Q, R


def _qr(x):
    if isinstance(x, np.ndarray):
        return _qr_numba(x)
    return do('linalg.qr', x)


@njit  # pragma: no cover
def _lq_numba(x):
    """LQ-decomposition.
    """
    Q, L = np.linalg.qr(x.T)
    return L.T, Q.T


def _lq(x):
    if isinstance(x, np.ndarray):
        return _lq_numba(x)
    Q, L = do('linalg.qr', do('transpose', x))
    return do('transpose', L), do('transpose', Q)
