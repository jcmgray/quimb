"""Functions for more advanced calculations of quantities and properties of
quantum objects.
"""
from math import sin, cos, pi, log, log2, sqrt
import numbers
import collections
import itertools
import functools

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from .accel import (
    accel,
    dot_dense,
    ldmul,
    issparse,
    isop,
    zeroify,
    realify,
    prod,
    isvec,
    dot,
)
from .core import (
    qu,
    kron,
    eye,
    eyepad,
    tr,
    ptr,
    infer_size,
    overlap,
    dop,
)
from .solve.base_solver import (
    eigsys,
    eigvals,
    norm,
    seigvals,
)
from .solve.mpi_spawner import (
    mfn_multiply_slepc_spawn,
)
from .gen.operators import sig
from .gen.states import (
    basis_vec,
    bell_state,
    bloch_state,
)


def expm(a, herm=True):
    """Matrix exponential, can be accelerated if explicitly hermitian.

    Parameters
    ----------
    a : dense or sparse matrix
        Matrix to exponentiate.
    herm : bool, optional
        If True (the default), and ``a`` is dense, digonalize the matrix
        in order to perform the exponential.

    Returns
    -------
    matrix
    """
    if issparse(a):
        # convert to and from csc to suppress scipy warning
        return spla.expm(a.tocsc()).tocsr()
    elif not herm:
        return np.asmatrix(spla.expm(a))
    else:
        evals, evecs = eigsys(a)
        return dot_dense(evecs, ldmul(np.exp(evals), evecs.H))


_EXPM_MULTIPLY_METHODS = {
    'SCIPY': spla.expm_multiply,
    'SLEPC': functools.partial(mfn_multiply_slepc_spawn, fntype='exp'),
}


def expm_multiply(mat, vec, backend="AUTO", **kwargs):
    """Compute the action of ``expm(mat)`` on ``vec``.

    Parameters
    ----------
    mat : matrix-like
        Matrix to exponentiate.
    vec : vector-like
        Vector to act with exponential of matrix on.
    backend : {'AUTO', 'SCIPY', 'SLEPC'}, optional
        Which backend to use.
    kwargs
        Supplied to backend function.

    Returns
    -------
    vector
        Result of ``expm(mat) @ vec``.
    """
    return _EXPM_MULTIPLY_METHODS[backend.upper()](mat, vec, **kwargs)


def sqrtm(a, herm=True):
    """Matrix square root, can be accelerated if explicitly hermitian.

    Parameters
    ----------
    a : dense or sparse matrix
        Matrix to take square root of.
    herm : bool, optional
        If True (the default), and ``a`` is dense, digonalize the matrix
        in order to take the square root.

    Returns
    -------
    matrix
    """
    if issparse(a):
        raise NotImplementedError("No sparse sqrtm available.")
    elif not herm:
        return np.asmatrix(sla.sqrtm(a))
    else:
        evals, evecs = eigsys(a)
        return dot_dense(evecs, ldmul(np.sqrt(evals.astype(complex)),
                                      evecs.H))


def fidelity(p1, p2):
    """Fidelity between two quantum states.

    Parameters
    ----------
    p1 : vector or matrix
        First state.
    p2 : vector or matrix
        Second state.

    Returns
    -------
    float
    """
    if isvec(p1) or isvec(p2):
        return overlap(p1, p2)
    else:
        sqrho = sqrtm(p1)
        return tr(sqrtm(dot(sqrho, dot(p2, sqrho))))
        # return norm(sqrtm(p1) @ sqrtm(p2), "tr")


def purify(rho):
    """Take state rho and purify it into a wavefunction of squared
    dimension.

    Parameters
    ----------
    rho : matrix
        Density matrix to purify.

    Returns
    -------
    vector :
        The purified ket.
    """
    # TODO: trim zeros?
    d = rho.shape[0]
    evals, vs = eigsys(rho)
    evals = np.sqrt(np.clip(evals, 0, 1))
    psi = np.zeros(shape=(d**2, 1), dtype=complex)
    for i, evals in enumerate(evals.flat):
        psi += evals * kron(vs[:, i], basis_vec(i, d))
    return qu(psi)


@zeroify
def entropy(a, rank=None):
    """Compute the (von Neumann) entropy.

    Parameters
    ----------
    a : matrix or 1d array
        Positive operator or list of positive eigenvalues.
    rank : int (optional)
        If operator has known rank, then a partial decomposition can be
        used to accelerate the calculation.

    Returns
    -------
    float
        The von Neumann entropy.
    """
    a = np.asarray(a)
    if np.ndim(a) == 1:
        evals = a
    else:
        if rank is None:
            evals = eigvals(a)
        else:  # know that not all eigenvalues needed
            evals = seigvals(a, k=rank, which='LM', backend='AUTO')

    evals = evals[evals > 0.0]
    return np.sum(-evals * np.log2(evals))


@zeroify
def mutual_information(p, dims=(2, 2), sysa=0, sysb=1, rank=None):
    """Find the mutual information between two subystems of a state.

    That is, ``H(A) + H(B) - H(AB)``, for von Neumann entropy ``H``, and two
    subsystems A and B.

    Parameters
    ----------
    p : vector or matrix
        State, can be vector or operator.
    dims : tuple(int), optional
        Internal dimensions of state.
    sysa : int, optional
        Index of first subsystem, A.
    sysb : int, optional
        Index of second subsystem, B.
    rank : int, optional
        If known, the rank of rho_ab, to speed calculation of ``H(AB)`` up.
        For example, if ``p`` comes from tracing out three qubits from a
        system, then its rank is 2^3 = 8 etc.

    Returns
    -------
    float
    """
    if np.size(dims) > 2:
        if rank == 'AUTO':
            rank = prod(dims) // (dims[sysa] * dims[sysb])
        p = ptr(p, dims, (sysa, sysb))
        dims = (dims[sysa], dims[sysb])
    if isop(p):  # mixed combined system
        hab = entropy(p, rank=rank)
        rhoa = ptr(p, dims, 0)
        rhob = ptr(p, dims, 1)
        ha, hb = entropy(rhoa), entropy(rhob)
    else:  # pure combined system
        hab = 0.0
        rhoa = ptr(p, dims, sysa)
        ha = hb = entropy(rhoa)
    return ha + hb - hab


mutinf = mutual_information


def partial_transpose(p, dims=(2, 2)):
    """Partial transpose of a density matrix.

    Parameters
    ----------
    p : matrix or vector
        The state to partially transpose.
    dims : tuple(int), optional
        The two internal dimensions describing a bipartition of the state.

    Returns
    -------
    matrix
    """
    p = qu(p, "dop")
    p = np.array(p)  \
        .reshape((*dims, *dims))  \
        .transpose((2, 1, 0, 3))  \
        .reshape((prod(dims), prod(dims)))
    return qu(p)


@zeroify
def negativity(p, dims=(2, 2), sysa=0, sysb=1):
    """Compute negativity between two subsytems.

    This is defined as ( | rho_{AB}^{T_B} | - 1) / 2. If ``len(dims) > 2``,
    then the non-target dimensions will be traced out first.

    Parameters
    ----------
    p : matrix or vector
        State to compute negativity for.
    dims : tuple(int), optional
        The internal dimensions of ``p``.
    sysa : int, optional
        Index of the first subsystem, A, relative to ``dims``.
    sysb : int, optional
        Index of the first subsystem, B, relative to ``dims``.

    Returns
    -------
    float
    """
    if isvec(p):
        p = qu(p, qtype='dop')
    if len(dims) > 2:
        p = ptr(p, dims, (sysa, sysb))
        dims = (dims[sysa], dims[sysb])
    n = (norm(partial_transpose(p, dims=dims), "tr") - 1.0) / 2.0
    return max(0.0, n)


@zeroify
def logarithmic_negativity(p, dims=(2, 2), sysa=0, sysb=1):
    """Compute logarithmic negativity between two subsytems.

    This is defined as  log_2( | rho_{AB}^{T_B} | ). If ``len(dims) > 2``,
    then the non-target dimensions will be traced out first.

    Parameters
    ----------
    p : matrix or vector
        State to compute logarithmic negativity for.
    dims : tuple(int), optional
        The internal dimensions of ``p``.
    sysa : int, optional
        Index of the first subsystem, A, relative to ``dims``.
    sysb : int, optional
        Index of the first subsystem, B, relative to ``dims``.

    Returns
    -------
    float
    """
    if isvec(p) and len(dims) == 2:  # pure bipartition, easier to calc
        smaller_system = 0 if dims[0] <= dims[1] else 1
        rhoa = ptr(p, dims, smaller_system)
        e = 2 * log2(sum(np.sqrt(np.clip(eigvals(rhoa, sort=False), 0, 1))))
    else:
        if len(dims) > 2:  #
            p = ptr(p, dims, (sysa, sysb))
            dims = (dims[sysa], dims[sysb])
        e = log2(norm(partial_transpose(p, dims), "tr"))
    return max(0.0, e)


logneg = logarithmic_negativity


@zeroify
def concurrence(p, dims=(2, 2), sysa=0, sysb=1):
    """Concurrence of two-qubit state.

    If ``len(dims) > 2``, then the non-target dimensions will be traced out
    first.

    Parameters
    ----------
    p : matrix or vector
        State to compute concurrence for.
    dims : tuple(int), optional
        The internal dimensions of ``p``.
    sysa : int, optional
        Index of the first subsystem, A, relative to ``dims``.
    sysb : int, optional
        Index of the first subsystem, B, relative to ``dims``.

    Returns
    -------
    float
    """
    if len(dims) > 2:
        p = ptr(p, dims, (sysa, sysb))
    if isop(p):
        pt = dot(kron(sig(2), sig(2)), dot(p.conj(), kron(sig(2), sig(2))))
        evals = (nla.eigvals(dot(p, pt)).real**2)**0.25
        return max(0, 2 * np.max(evals) - np.sum(evals))
    else:
        pt = dot(kron(sig(2), sig(2)), p.conj())
        c = np.real(abs(dot(p.H, pt))).item(0)
        return max(0, c)


def one_way_classical_information(p_ab, prjs, precomp_func=False):
    """One way classical information for two qubit density matrix.

    Parameters
    ----------
    p_ab : matrix
        State of two qubits
    prjs : sequence of matrices
       The POVMs.
    precomp_func : bool, optional
        Whether to return a pre-computed function, closed over the actual
        state.

    Returns
    -------
    float or callable
        The one-way classical information or the function to compute it for
        the given state which takes a set of POVMs as its single argument.
    """
    p_a = ptr(p_ab, (2, 2), 0)
    s_a = entropy(p_a)

    def owci(prjs):
        def gen_paj():
            for prj in prjs:
                p_ab_j = dot((eye(2) & prj), p_ab)
                prob = tr(p_ab_j)
                p_a_j = ptr(p_ab_j, (2, 2), 0) / prob
                yield prob, p_a_j
        return s_a - sum(p * entropy(rho) for p, rho in gen_paj())

    return owci if precomp_func else owci(prjs)


@zeroify
def quantum_discord(p, dims=(2, 2), sysa=0, sysb=1):
    """Quantum Discord for two qubit density matrix.

    If ``len(dims) > 2``, then the non-target dimensions will be traced out
    first.

    Parameters
    ----------
    p : matrix or vector
        State to compute quantum discord for.
    dims : tuple(int), optional
        The internal dimensions of ``p``.
    sysa : int, optional
        Index of the first subsystem, A, relative to ``dims``.
    sysb : int, optional
        Index of the first subsystem, B, relative to ``dims``.

    Returns
    -------
    float
    """
    if len(dims) > 2:
        p = ptr(p, dims, (sysa, sysb))
    else:
        p = qu(p, "dop")
    iab = mutual_information(p)
    owci = one_way_classical_information(p, None, precomp_func=True)

    def trial_qd(a):
        ax, ay, az = sin(a[0]) * cos(a[1]), sin(a[0]) * sin(a[1]), cos(a[0])
        prja = bloch_state(ax, ay, az)
        prjb = eye(2) - prja
        return iab - owci((prja, prjb))

    opt = minimize(trial_qd, (pi / 2, pi),
                   method="SLSQP", bounds=((0, pi), (0, 2 * pi)))
    if opt.success:
        return opt.fun
    else:  # pragma: no cover
        raise ValueError(opt.message)


@zeroify
def trace_distance(p1, p2):
    """Trace distance between two states.

    Parameters
    ----------
    p1 : vector or matrix
        The first state.
    p2 : vector or matrix
        The second state.

    Returns
    -------
    float
    """
    p1_is_op, p2_is_op = isop(p1), isop(p2)

    # If both are pure kets then special case
    if (not p1_is_op) and (not p2_is_op):
        return sqrt(1 - overlap(p1, p2))

    # Otherwise do full calculation
    return 0.5 * norm((p1 if p1_is_op else dop(p1)) -
                      (p2 if p2_is_op else dop(p2)), "tr")


def decomp(a, fn, fn_args, fn_d, nmlz_func, mode="p", tol=1e-3):
    """Decomposes an operator via the Hilbert-schmidt inner product.

    Can both print the decomposition or return it.

    Parameters
    ----------
    a : vector or matrix
        Operator to decompose.
    fn : callable
        Function to generate operator/state to decompose with.
    fn_args :
        Sequence of args whose permutations will be supplied to ``fn``.
    fn_d : int
        The dimension of the operators that `fn` produces.
    nmlz_func : callable
        Function to produce a normlization coefficient given the ``n``
        permutations of operators that will be produced.
    mode :
        String, include ``'p'`` to print the decomp and/or ``'c'`` to
        return OrderedDict, sorted by size of contribution.
    tol :
        Print operators with contirbution above ``tol`` only.

    Returns
    -------
    None or OrderedDict:
        Pauli operator name and overlap with ``a``.

    See Also
    --------
    pauli_decomp, bell_decomp
    """
    if isvec(a):
        a = qu(a, "dop")  # make sure operator
    n = infer_size(a, base=fn_d)

    # define generator for inner product to iterate over efficiently
    def calc_name_and_overlap():
        for perm in itertools.product(fn_args, repeat=n):
            op = kron(*(fn(x, sparse=True) for x in perm)) * nmlz_func(n)
            cff = overlap(a, op)
            yield "".join(str(x) for x in perm), cff

    names_cffs = list(calc_name_and_overlap())
    # sort by descending overlap and turn into OrderedDict
    names_cffs.sort(key=lambda pair: -abs(pair[1]))
    names_cffs = collections.OrderedDict(names_cffs)
    # Print decomposition
    if "p" in mode:
        for x, cff in names_cffs.items():
            if abs(cff) < 0.01:
                break
            dps = int(round(0.5 - np.log10(1.001 * tol)))  # decimal places
            print(x, "{: .{prec}f}".format(cff, prec=dps))
    # Return full calculation
    if "c" in mode:
        return names_cffs


pauli_decomp = functools.partial(decomp,
                                 fn=sig,
                                 fn_args='IXYZ',
                                 fn_d=2,
                                 nmlz_func=lambda n: 2**-n)

bell_decomp = functools.partial(decomp,
                                fn=bell_state,
                                fn_args=(0, 1, 2, 3),
                                fn_d=4,
                                nmlz_func=lambda x: 1)


def correlation(p, opa, opb, sysa, sysb, dims=None, sparse=None,
                precomp_func=False):
    """Calculate the correlation between two sites given two operators.

    Parameters
    ----------
    p : vector or matrix
        State to compute correlations for, ignored if ``precomp_func=True``.
    opa : matrix
        Operator to act on first subsystem.
    opb : matrix
        Operator to act on second subsystem.
    sysa : int
        Index of first subsystem.
    sysb : int
        Index of second subsystem.
    dims : tuple of int, optional
        Internal dimensions of ``p``, will be assumed to be qubits if not
        given.
    sparse : bool, optional
        Whether to compute with sparse operators.
    precomp_func : bool, optional
        Whether to return result or single arg function closed
        over precomputed operator.

    Returns
    -------
    float or callable
        The correlation, <ab> - <a><b>, or a function to compute for an
        arbitrary state.
    """
    if dims is None:
        sz_p = infer_size(p)
        dims = (2,) * sz_p
    if sparse is None:
        sparse = issparse(opa) or issparse(opb)

    opts = {'sparse': sparse,
            'coo_build': sparse,
            'stype': 'csr' if sparse else None}
    opab = eyepad((opa, opb), dims, (sysa, sysb), **opts)
    opa = eyepad((opa,), dims, sysa, **opts)
    opb = eyepad((opb,), dims, sysb, **opts)

    @realify
    def corr(state):
        return overlap(opab, state) - overlap(opa, state) * overlap(opb, state)

    return corr if precomp_func else corr(p)


def pauli_correlations(p, ss=("xx", "yy", "zz"), sysa=0, sysb=1,
                       sum_abs=False, precomp_func=False):
    """Calculate the correlation between sites for a list of operator pairs
    choisen from the pauli matrices.

    Parameters
    ----------
    p : vector or matrix
        State to compute correlations for. Ignored if ``precomp_func=True``.
    ss : tuple or str
        List of pairs specifiying pauli matrices.
    sysa : int, optional
        Index of first site.
    sysb : int, optional
        Index of second site.
    sum_abs : bool, optional
        Whether to sum over the absolute values of each correlation
    precomp_func : bool, optional
        whether to return the values or a single argument
        function closed over precomputed operators etc.

    Returns
    -------
    list of float, list of callable, float or callable
        Either the value(s) of each correlation or the function(s) to compute
        the correlations for an arbitrary state, depending on ``sum_abs`` and
        ``precomp_func``.
    """
    def gen_corr_list():
        for s1, s2 in ss:
            yield correlation(p, sig(s1), sig(s2), sysa, sysb,
                              precomp_func=precomp_func)

    if sum_abs:

        if precomp_func:
            return lambda p: sum((abs(corr(p)) for corr in gen_corr_list()))

        return sum((abs(corr) for corr in gen_corr_list()))

    return tuple(gen_corr_list())


def ent_cross_matrix(p, sz_blc=1, ent_fn=logneg, calc_self_ent=True):
    """Calculate the pair-wise function ent_fn  between all sites or blocks
    of a state.

    Parameters
    ----------
    p : matrix-like
        State.
    sz_blc : int
        Size of the blocks to partition the state into. If the number of
        individual sites is not a multiple of this then the final (smaller)
        block will be ignored.
    ent_fn : callable
        Bipartite function, notionally entanglement
    calc_self_ent : bool
        Whether to calculate the function for each site
        alone, purified. If the whole state is pure then this is the
        entanglement with the whole remaining system.

    Returns
    -------
    2d-numpy.ndarray
        matrix of pairwise ent_fn results.
    """
    sz_p = infer_size(p)
    dims = (2,) * sz_p
    n = sz_p // sz_blc
    ents = np.empty((n, n))

    ispure = isvec(p)
    if ispure and sz_blc * 2 == sz_p:  # pure bipartition
        ent = ent_fn(p, dims=(2**sz_blc, 2**sz_blc)) / sz_blc
        ents[:, :] = ent
        if not calc_self_ent:
            for i in range(n):
                ents[i, i] = np.nan
        return ents

    # Range over pairwise blocks
    for i in range(0, sz_p - sz_blc + 1, sz_blc):
        for j in range(i, sz_p - sz_blc + 1, sz_blc):
            if i == j:
                if calc_self_ent:
                    rhoa = ptr(p, dims, [i + b for b in range(sz_blc)])
                    psiap = purify(rhoa)
                    ent = ent_fn(psiap, dims=(2**sz_blc, 2**sz_blc)) / sz_blc
                else:
                    ent = np.nan
            else:
                rhoab = ptr(p, dims, [i + b for b in range(sz_blc)] +
                                     [j + b for b in range(sz_blc)])
                ent = ent_fn(rhoab, dims=(2**sz_blc, 2**sz_blc)) / sz_blc
            ents[i // sz_blc, j // sz_blc] = ent
            ents[j // sz_blc, i // sz_blc] = ent
    return ents


def qid(p, dims, inds, precomp_func=False, sparse_comp=True,
        norm_func=norm, pow=2, coeff=1):
    # Check inputs
    inds = (inds,) if isinstance(inds, numbers.Number) else inds

    # Construct operators
    ops_i = tuple(tuple(eyepad(sig(s), dims, ind, sparse=sparse_comp)
                        for s in "xyz")
                  for ind in inds)

    # Define function closed over precomputed operators
    def qid_func(x):
        if isvec(x):
            x = dop(x)
        return tuple(sum(coeff * norm_func(dot(x, op) - dot(op, x))**pow
                         for op in ops)
                     for ops in ops_i)

    return qid_func if precomp_func else qid_func(p)


def is_degenerate(op, tol=1e-12):
    """Check if operator has any degenerate eigenvalues, determined relative
    to mean spacing of all eigenvalues.

    Parameters
    ----------
    op : matrix or 1d-array
        Operator or assumed eigenvalues to check degeneracy for.
    tol : float
        How much closer than evenly spaced the eigenvalue gap has to be
        to count as degenerate.

    Returns
    -------
    n_dgen : int
        Number of degenerate eigenvalues.
    """
    op = np.asarray(op)
    if op.ndim != 1:
        evals = eigvals(op)
    else:
        evals = op
    l_gaps = evals[1:] - evals[:-1]
    l_tol = tol * (evals[-1] - evals[0]) / op.shape[0]
    return np.count_nonzero(abs(l_gaps) < l_tol)


@accel
def page_entropy(sz_subsys, sz_total):  # pragma: no cover
    """Calculate the page entropy, i.e. expected entropy for a subsytem
    of a random state in Hilbert space.

    Parameters
    ----------
    sz_subsys : int
        Dimension of subsystem.
    sz_total : int
        Dimension of total system.

    Returns
    -------
    s : float
        Entropy in bits.
    """
    n = sz_total // sz_subsys

    if sz_subsys > n:
        raise ValueError("Subsystem size must be <= 'environment' size.")

    s = 0
    for k in range(n + 1, sz_total + 1):
        s += 1 / k
    s -= (sz_subsys - 1) / (2 * n)

    # Normalize into bits of entropy
    return s / log(2)
