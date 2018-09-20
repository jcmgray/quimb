"""Randomized iterative methods for decompositions.
"""
from numbers import Integral

import numpy as np
import scipy.linalg as sla
from cytoolz import identity

from ..gen.rand import randn
from ..core import dag, dot, njit


def lu_orthog(X):
    return sla.lu(X, permute_l=True, overwrite_a=True, check_finite=False)[0]


def qr_orthog(X):
    return sla.qr(X, mode='economic', overwrite_a=True, check_finite=False)[0]


def orthog(X, lu=False):
    if lu:
        return lu_orthog(X)
    return qr_orthog(X)


def QB_to_svd(Q, B, compute_uv=True):
    UsV = sla.svd(B, full_matrices=False, compute_uv=compute_uv,
                  overwrite_a=True, check_finite=False)

    if not compute_uv:
        return UsV

    U, s, V = UsV
    return dot(Q, U), s, V


def trim(arrays, k):
    if isinstance(arrays, tuple) and len(arrays) == 3:
        U, s, VH = arrays
        U, s, VH = U[:, :k], s[:k], VH[:k, :]
        return U, s, VH
    if isinstance(arrays, tuple) and len(arrays) == 2:
        # Q, B factors
        Q, B = arrays
        return Q[:, :k], B[:k, :]
    else:
        # just singular values
        return arrays[:k]


def possibly_extend_randn(G, k, p, A):
    # make sure we are using block of the right size by removing or adding
    kG = G.shape[1]
    if kG > k + p:
        # have too many columns
        G = G[:, :k + p]
    elif kG < k + p:
        # have too few columns
        G_extra = randn((A.shape[1], k + p - kG), dtype=A.dtype)
        G = np.concatenate((G, G_extra), axis=1)
    return G


def isstring(x, s):
    if not isinstance(x, str):
        return False
    return x == s


def rsvd_qb(A, k, q, p, state, AH=None):

    if AH is None:
        AH = dag(A)

    # generate first block
    if isstring(state, 'begin-qb'):
        G = randn((A.shape[1], k + p), dtype=A.dtype)
    # block already supplied
    elif len(state) == 1:
        G, = state
    # mid-way through adaptive algorithm in QB mode
    if len(state) == 3:
        Q, B, G = state
    else:
        Q = np.empty((A.shape[0], 0), dtype=A.dtype)
        B = np.empty((0, A.shape[1]), dtype=A.dtype)

    QH, BH = dag(Q), dag(B)
    G = possibly_extend_randn(G, k, p, A)

    Qi = orthog(dot(A, G) - dot(Q, dot(B, G)), lu=q > 0)

    for i in range(1, q + 1):
        Qi = orthog(dot(AH, Qi) - dot(BH, dot(QH, Qi)), lu=True)
        Qi = orthog(dot(A, Qi) - dot(Q, dot(B, Qi)), lu=i != q)

    Qi = orthog(Qi - dot(Q, dot(QH, Qi)))
    Bi = dag(dot(AH, Qi)) - dot(dot(dag(Qi), Q), B)

    if p > 0:
        Qi, Bi = trim((Qi, Bi), k)

    Q = np.concatenate((Q, Qi), axis=1)
    B = np.concatenate((B, Bi), axis=0)

    return Q, B, G


def rsvd_core(A, k, compute_uv=True, q=2, p=0, state=None, AH=None):
    """Core R3SVD algorithm.

    Parameters
    ----------
    A : linear operator, shape (m, n)
        Operator to decompose, assumed m >= n.
    k : int
        Number of singular values to find.
    compute_uv : bool, optional
        Return the left and right singular vectors.
    q : int, optional
        Number of power iterations.
    p : int, optional
        Over sampling factor.
    state : {None, array_like, (), (G0,), (U0, s0, VH0, G0)}, optional
        Iterate based on these previous results:

            - None: basic mode.
            - array_like: use this as the initial subspace.
            - 'begin-svd': begin block iterations, return U, s, VH, G
            - (G0,) : begin block iterations with this subspace
            - (U0, s0, VH0, G0): continue block iterations, return G

    """
    iterating = isinstance(state, (tuple, str))
    maybe_project_left = maybe_project_right = identity

    if AH is None:
        AH = dag(A)

    # generate first block
    if state is None or isstring(state, 'begin-svd'):
        G = randn((A.shape[1], k + p), dtype=A.dtype)
    # initial block supplied
    elif hasattr(state, 'shape'):
        G = state
    elif len(state) == 1:
        G, = state
    # mid-way through adaptive algorithm in SVD mode
    elif len(state) == 4:
        U0, s0, VH0, G = state
        UH0, V0 = dag(U0), dag(VH0)

        def maybe_project_left(X):
            X -= dot(U0, dot(UH0, X))
            return X

        def maybe_project_right(X):
            X -= dot(V0, dot(VH0, X))
            return X

    G = possibly_extend_randn(G, k, p, A)
    G = maybe_project_right(G)

    Q = dot(A, G)
    Q = maybe_project_left(Q)
    Q = orthog(Q, lu=q > 0)

    # power iterations with stabilization
    for i in range(1, q + 1):
        Q = dot(AH, Q)
        Q = maybe_project_right(Q)
        Q = orthog(Q, lu=True)

        Q = dot(A, Q)
        Q = maybe_project_left(Q)
        Q = orthog(Q, lu=i < q)

    B = dag(dot(AH, Q))
    UsVH = QB_to_svd(Q, B, compute_uv=compute_uv or iterating)
    if p > 0:
        UsVH = trim(UsVH, k)

    if not iterating:
        return UsVH

    U, s, VH = UsVH

    if isstring(state, 'begin-svd') or len(state) == 1:
        # first run -> don't need to project or concatenate anything
        return U, s, VH, G

    U = orthog(maybe_project_left(U))
    VH = dag(orthog(maybe_project_right(dag(VH))))

    U = np.concatenate((U0, U), axis=1)
    s = np.concatenate((s0, s))
    VH = np.concatenate((VH0, VH), axis=0)

    return U, s, VH, G


@njit
def is_sorted(x):  # pragma: no cover
    for i in range(x.size - 1):
        if x[i + 1] < x[i]:
            return False
    return True


def gen_k_steps(start, incr=1.4):
    yield start
    step = start
    while True:
        yield step
        step = round(incr * step)


def rsvd_iterate(A, eps, compute_uv=True, q=2, p=0, G0=None,
                 k_max=None, k_start=2, k_incr=1.4, AH=None, use_qb=20):
    """Handle rank-adaptively calling ``rsvd_core``.
    """

    if AH is None:
        AH = dag(A)

    # perform first iteration and set initial rank
    k_steps = gen_k_steps(k_start, k_incr)
    rank = next(k_steps)

    if use_qb:
        Q, B, G = rsvd_qb(A, rank, q=q, p=p, AH=AH,
                          state='begin-qb' if G0 is None else (G0,))
        U, s, VH = QB_to_svd(Q, B)
        G -= dot(dag(VH), dot(VH, G))
    else:
        U, s, VH, G = rsvd_core(A, rank, q=q, p=p, AH=AH,
                                state='begin-svd' if G0 is None else (G0,))

    # perform randomized SVD in small blocks
    while (s[-1] > eps * s[0]) and (rank < k_max):

        # only step k as far as k_max
        new_k = min(next(k_steps), k_max - rank)
        rank += new_k

        if (rank < use_qb) or (use_qb is True):
            Q, B, G = rsvd_qb(A, new_k, q=q, p=p, state=(Q, B, G), AH=AH)
            U, s, VH = QB_to_svd(Q, B)
            G -= dot(dag(VH), dot(VH, G))
        else:
            # concatenate new U, s, VH orthogonal to current U, s, VH
            U, s, VH, G = rsvd_core(A, new_k, q=q, p=p,
                                    state=(U, s, VH, G), AH=AH)

    # make sure singular values always sorted in decreasing order
    if not is_sorted(s):
        so = np.argsort(s)[::-1]
        U, s, VH = U[:, so], s[so], VH[so, :]

    return U, s, VH if compute_uv else s


@njit
def count_svdvals_needed(s, eps):  # pragma: no cover
    n = s.size
    thresh = eps * s[0]
    for i in range(n - 1, 0, -1):
        if s[i - 1] < thresh:
            n -= 1
        else:
            break
    return n


def isdouble(dtype):
    """Check if ``dtype`` is double precision.
    """
    return dtype in ('float64', 'complex128')


def estimate_rank(A, eps, k_max=None, use_sli=True, k_start=2, k_incr=1.4,
                  q=0, p=0, get_vectors=False, G0=None, AH=None, use_qb=20):
    """Estimate the rank of an linear operator. Uses a low quality random
    SVD with a resolution of ~ 10.

    Parameters
    ----------
    A : linear operator
        The operator to find rank of.
    eps : float
        Find rank to this relative (compared to largest singular value)
        precision.
    k_max : int, optional
        The maximum rank to find.
    use_sli : bool, optional
        Whether to use :func:`scipy.linalg.interpolative.estimate_rank` if
        possible (double precision and no ``k_max`` set).
    k_start : int, optional
        Begin the adaptive SVD with a block of this size.
    k_incr : float, optional
        Adaptive rank increment factor. Increase the k-step (from k_start) by
        this factor each time. Set to 1 to use a constant step.
    q : int, optional
        Number of power iterations.
    get_vectors : bool, optional
        Return the right singular vectors found in the pass.
    G0 : , optional

    Returns
    -------
    rank : int
        The rank.
    VH : array
        The (adjoint) right singular vectors if ``get_vectors=True``.
    """
    if k_max is None:
        k_max = min(A.shape)
    if eps <= 0.0:
        return k_max

    use_sli = (use_sli and (k_max == min(A.shape)) and
               isdouble(A.dtype) and not get_vectors)
    if use_sli:
        return sla.interpolative.estimate_rank(A, eps)

    if A.shape[0] < A.shape[1]:
        A = A.T
        if get_vectors:
            raise ValueError
    if AH is None:
        AH = dag(A)

    _, s, VH = rsvd_iterate(A, eps, q=q, p=p, G0=G0, AH=AH, use_qb=use_qb,
                            k_start=k_start, k_max=k_max, k_incr=k_incr)

    rank = count_svdvals_needed(s, eps)

    if get_vectors:
        return rank, VH[:rank, :]
    return rank


def maybe_flip(UsV, flipped):
    # if only singular values or only tranposing do nothing
    if not (isinstance(UsV, tuple) and flipped):
        return UsV
    U, s, V = UsV
    return V.T, s, U.T


def rsvd(A, eps_or_k, compute_uv=True, mode='adapt+block', use_qb=20,
         q=2, p=0, k_max=None, k_start=2, k_incr=1.4, G0=None, AH=None):
    """Fast, randomized, iterative SVD. Adaptive variant of method due
    originally to Halko. This scales as ``log(k)`` rather than ``k`` so can be
    more efficient.

    Parameters
    ----------
    A : operator, shape (m, n)
        The operator to decompose.
    eps_or_k : float or int
        Either the relative precision or the number of singular values to
        target. If precision, this is relative to the largest singular value.
    compute_uv : bool, optional
        Whether to return the left and right singular vectors.
    mode : {'adapt+block', 'adapt', 'block'}, optional
        How to perform the randomized SVD. If ``eps_or_k`` is an integer then
        this is implicitly 'block' and ignored. Else:

            - 'adapt+block', perform an initial low quality pass to estimate
              the rank of ``A``, then use the subspace and rank from that to
              perform an accurate fully blocked RSVD.
            - 'adapt', just perform the adaptive randomized SVD.

    q : int, optional
        The number of power iterations, increase for accuracy at the expense
        of runtime.
    p : int, optional
        Oversampling factor. Perform projections with this many extra columns
        and then throw then away.
    k_max : int, optional
        Maximum adaptive rank. Default: ``min(A.shape)``.
    k_start : int, optional
        Initial k when increasing rank adaptively.
    k_incr : float, optional
        Adaptive rank increment factor. Increase the k-step (from k_start) by
        this factor each time. Set to 1 to use a constant step.
    G0 : array_like, shape (n, k), optional
        Initial subspace to start iterating on. If not given a random one will
        be generated.

    Returns
    -------
    U, array, shape (m, k)
        Left singular vectors, if ``compute_uv=True``.
    s, array, shape (k,)
        Singular values.
    V, array, shape (k, n)
        Right singular vectors, if ``compute_uv=True``.
    """

    flipped = A.shape[0] < A.shape[1]
    if flipped:
        A = A.T

    # 'block' mode -> just perform single pass random SVD
    if isinstance(eps_or_k, Integral):
        UsV = rsvd_core(A, eps_or_k, q=q, p=p, state=G0, compute_uv=compute_uv)
        return maybe_flip(UsV, flipped)

    if k_max is None:
        k_max = min(A.shape)
    k_max = min(max(1, k_max), min(A.shape))

    if AH is None:
        AH = dag(A)

    adaptive_opts = {'k_start': k_start, 'k_max': k_max, 'k_incr': k_incr,
                     'use_qb': use_qb, 'AH': AH, 'G0': G0}

    # 'adapt' mode -> rank adaptively perform SVD to low accuracy
    if mode == 'adapt':
        UsV = rsvd_iterate(A, eps_or_k, q=q, p=p,
                           compute_uv=compute_uv, **adaptive_opts)

    # 'adapt+block' mode -> use first pass to find rank, then use blocking mode
    elif mode == 'adapt+block':

        # estimate both rank and get approximate spanning vectors
        k, VH = estimate_rank(A, eps_or_k, get_vectors=True, **adaptive_opts)

        # reuse vectors to effectively boost number of power iterations by one
        UsV = rsvd_core(A, k, q=max(q - 1, 0), p=p, AH=AH,
                        state=dag(VH), compute_uv=compute_uv)

    else:
        raise ValueError("``mode`` must  be one of {'adapt+block', 'adapt'} or"
                         " ``k`` should be a integer to use 'block' mode.")

    return maybe_flip(UsV, flipped)
