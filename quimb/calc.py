"""Functions for more advanced calculations of quantities and properties of
quantum objects.
"""
import numbers
import itertools
import functools
import collections
from math import sin, cos, pi, log, log2, sqrt

import numpy as np
import numpy.linalg as nla
from scipy.optimize import minimize

from .core import (
    njit, issparse, isop, zeroify, realify, prod, isvec, dot, dag,
    qu, kron, eye, ikron, tr, ptr, infer_size, expec, dop, ensure_qarray,
)
from .linalg.base_linalg import (
    eigh, eigvalsh, norm, sqrtm,
)
from .linalg.approx_spectral import (
    entropy_subsys_approx, tr_sqrt_subsys_approx,
    logneg_subsys_approx, gen_bipartite_spectral_fn,
)
from .gen.operators import pauli
from .gen.states import (
    basis_vec, bell_state, bloch_state,
)
from .utils import int2tup


def fidelity(p1, p2):
    """Fidelity between two quantum states.

    Parameters
    ----------
    p1 : vector or operator
        First state.
    p2 : vector or operator
        Second state.

    Returns
    -------
    float
    """
    if isvec(p1) or isvec(p2):
        return expec(p1, p2)
    else:
        sqrho = sqrtm(p1)
        return tr(sqrtm(dot(sqrho, dot(p2, sqrho))))
        # return norm(sqrtm(p1) @ sqrtm(p2), "tr")


def purify(rho):
    """Take state rho and purify it into a wavefunction of squared
    dimension.

    Parameters
    ----------
    rho : operator
        Density operator to purify.

    Returns
    -------
    vector :
        The purified ket.
    """
    d = rho.shape[0]
    evals, vs = eigh(rho)
    evals = np.sqrt(np.clip(evals, 0, 1))
    psi = np.zeros(shape=(d**2, 1), dtype=complex)
    for i, evals in enumerate(evals.flat):
        psi += evals * kron(vs[:, [i]], basis_vec(i, d))
    return qu(psi)


def dephase(rho, p, rand_rank=None):
    """Dephase ``rho`` by amount ``p``, that is, mix it
    with the maximally mixed state:

        rho -> (1 - p) * rho + p * I / d

    Parameters
    ----------
    rho : operator
        The state.
    p : float
        The final proportion of identity.
    rand_rank : int or float, optional
        If given, dephase with a random diagonal operator with this many
        non-zero entries. If float, proportion of full size.

    Returns
    -------
    rho_dephase : operator
        The dephased density operator.
    """
    d = rho.shape[0]

    if (rand_rank is None) or (rand_rank == d) or (rand_rank == 1.0):
        dephaser = eye(d) / d

    else:
        if not isinstance(rand_rank, numbers.Integral):
            rand_rank = int(rand_rank * d)
        rand_rank = min(max(1, rand_rank), d)

        dephaser = np.zeros((d, d))
        dephaser_diag = np.einsum("aa->a", dephaser)
        nnz = np.random.choice(np.arange(d), size=rand_rank, replace=False)
        dephaser_diag[nnz] = 1 / rand_rank

    return (1 - p) * rho + p * dephaser


@zeroify
def entropy(a, rank=None):
    """Compute the (von Neumann) entropy.

    Parameters
    ----------
    a : operator or 1d array
        Positive operator or list of positive eigenvalues.
    rank : int (optional)
        If operator has known rank, then a partial decomposition can be
        used to accelerate the calculation.

    Returns
    -------
    float
        The von Neumann entropy.

    See Also
    --------
    mutinf, entropy_subsys, entropy_subsys_approx
    """
    a = np.asarray(a)
    if np.ndim(a) == 1:
        evals = a
    else:
        if rank is None:
            evals = eigvalsh(a)
        else:  # know that not all eigenvalues needed
            evals = eigvalsh(a, k=rank, which='LM', backend='AUTO')

    evals = evals[evals > 0.0]
    return np.sum(-evals * np.log2(evals))


entropy_subsys = gen_bipartite_spectral_fn(entropy, entropy_subsys_approx, 0.0)
"""Calculate the entropy of a pure states' subsystem, optionally switching
to an approximate lanczos method when the subsystem is very large.

Parameters
----------
psi_ab : vector
    Bipartite state.
dims : sequence of int
    The sub-dimensions of the state.
sysa :  sequence of int
    The indices of which dimensions to calculate the entropy for.
approx_thresh : int, optional
    The size of sysa at which to switch to the approx method. Set to
    ``None`` to never use the approximation.
**approx_opts
    Supplied to :func:`entropy_subsys_approx`, if used.

Returns
-------
float
    The subsytem entropy.

See Also
--------
entropy, entropy_subsys_approx, mutinf_subsys
"""


@zeroify
def mutinf(p, dims=(2, 2), sysa=0, rank=None):
    """Find the mutual information for a bipartition of a state.

    That is, ``H(A) + H(B) - H(AB)``, for von Neumann entropy ``H``, and two
    subsystems A and B.

    Parameters
    ----------
    p : vector or operator
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

    See Also
    --------
    entropy, mutinf_subsys, entropy_subsys_approx
    """
    sysa = int2tup(sysa)

    # mixed combined system
    if isop(p):
        # total
        hab = entropy(p, rank=rank)

        # subsystem a
        rhoa = ptr(p, dims, sysa)
        ha = entropy(rhoa)

        # need subsystem b as well
        sysb = tuple(i for i in range(len(dims)) if i not in sysa)
        rhob = ptr(p, dims, sysb)
        hb = entropy(rhob)

    # pure combined system
    else:
        hab = 0.0
        ha = hb = entropy_subsys(p, dims, sysa)

    return ha + hb - hab


mutual_information = mutinf


def check_dims_and_indices(dims, *syss):
    """Make sure all indices found in the tuples ``syss`` are in
    ``range(len(dims))``.
    """
    nsys = len(dims)
    all_sys = sum(syss, ())

    if not all(0 <= i < nsys for i in all_sys):
        raise ValueError("Indices specified in `sysa` and `sysb` must be "
                         "in range({}) for dims {}.".format(nsys, dims))


def mutinf_subsys(psi_abc, dims, sysa, sysb, approx_thresh=2**13,
                  **approx_opts):
    """Calculate the mutual information of two subsystems of a pure state,
    possibly using an approximate lanczos method for large subsytems.

    Parameters
    ----------
    psi_abc : vector
        Tri-partite pure state.
    dims : sequence of int
        The sub dimensions of the state.
    sysa : sequence of int
        The index(es) of the subsystem(s) to consider part of 'A'.
    sysb : sequence of int
        The index(es) of the subsystem(s) to consider part of 'B'.
    approx_thresh : int, optional
        The size of subsystem at which to switch to the approximate lanczos
        method. Set to ``None`` to never use the approximation.
    approx_opts
        Supplied to :func:`entropy_subsys_approx`, if used.

    Returns
    -------
    float
        The mutual information.

    See Also
    --------
    mutinf, entropy_subsys, entropy_subsys_approx, logneg_subsys
    """
    sysa, sysb = int2tup(sysa), int2tup(sysb)

    check_dims_and_indices(dims, sysa, sysb)

    sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
    sz_b = prod(d for i, d in enumerate(dims) if i in sysb)
    sz_c = prod(dims) // (sz_a * sz_b)

    kws = {'approx_thresh': approx_thresh, **approx_opts}

    if sz_c == 1:
        hab = 0.0
        ha = hb = entropy_subsys(psi_abc, dims, sysa, **kws)
    else:
        hab = entropy_subsys(psi_abc, dims, sysa + sysb, **kws)
        ha = entropy_subsys(psi_abc, dims, sysa, **kws)
        hb = entropy_subsys(psi_abc, dims, sysb, **kws)

    return hb + ha - hab


def schmidt_gap(psi_ab, dims, sysa):
    """Find the schmidt gap of the bipartition of ``psi_ab``. That is, the
    difference between the two largest eigenvalues of the reduced density
    operator.

    Parameters
    ----------
    psi_ab : vector
        Bipartite state.
    dims : sequence of int
        The sub-dimensions of the state.
    sysa :  sequence of int
        The indices of which dimensions to calculate the entropy for.

    Returns
    -------
    float
    """
    sysa = int2tup(sysa)
    sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
    sz_b = prod(dims) // sz_a

    # pure state
    if sz_b == 1:
        return 1.0

    # also check if system b is smaller, since spectrum is same for both
    if sz_b < sz_a:
        # if so swap things around
        sysb = [i for i in range(len(dims)) if i not in sysa]
        sysa = sysb

    rho_a = ptr(psi_ab, dims, sysa)
    el = eigvalsh(rho_a, k=2, which='LM')
    return abs(el[0] - el[1])


def tr_sqrt(A, rank=None):
    """Return the trace of the sqrt of a positive semidefinite operator.
    """
    if rank is None:
        el = eigvalsh(A, sort=False)
    else:
        el = eigvalsh(A, k=rank, which='LM', backend='AUTO')
    return np.sum(np.sqrt(el[el > 0.0]))


tr_sqrt_subsys = gen_bipartite_spectral_fn(tr_sqrt, tr_sqrt_subsys_approx, 1.0)
"""Compute the trace sqrt of a subsystem, possibly using an approximate
lanczos method when the subsytem is big.

Parameters
----------
psi_ab : vector
    Bipartite state.
dims : sequence of int
    The sub-dimensions of the state.
sysa :  sequence of int
    The indices of which dimensions to calculate the trace sqrt for.
approx_thresh : int, optional
    The size of sysa at which to switch to the approx method. Set to
    ``None`` to never use the approximation.
**approx_opts
    Supplied to :func:`tr_sqrt_subsys_approx`, if used.

Returns
-------
float
    The subsytem entropy.

See Also
--------
tr_sqrt, tr_sqrt_subsys_approx, partial_transpose_norm
"""


@ensure_qarray
def partial_transpose(p, dims=(2, 2), sysa=0):
    """Partial transpose of a density operator.

    Parameters
    ----------
    p : operator or vector
        The state to partially transpose.
    dims : tuple(int), optional
        The internal dimensions of the state.
    sysa : sequence of int
        The indices of 'system A', everything else assumed to be 'system B'.

    Returns
    -------
    operator

    See Also
    --------
    logneg, negativity
    """
    sysa = int2tup(sysa)

    ndims = len(dims)
    perm_ket_inds = []
    perm_bra_inds = []

    for i in range(ndims):
        if i in sysa:
            perm_ket_inds.append(i + ndims)
            perm_bra_inds.append(i)
        else:
            perm_ket_inds.append(i)
            perm_bra_inds.append(i + ndims)

    return (np.asarray(qu(p, "dop"))
            .reshape((*dims, *dims))
            .transpose((*perm_ket_inds, *perm_bra_inds))
            .reshape((prod(dims), prod(dims))))


def partial_transpose_norm(p, dims, sysa):
    """Compute the norm of the partial transpose for (log)-negativity,
    taking a shortcut (trace sqrt of reduced subsytem), when system is a
    vector.
    """
    sysa = int2tup(sysa)

    # check for pure bipartition -> easier to calc
    if isvec(p):
        sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
        sz_b = prod(dims) // sz_a

        # check if system b is smaller, since entropy is same for both a & b.
        if sz_b < sz_a:
            # if so swap things around
            sysb = [i for i in range(len(dims)) if i not in sysa]
            sysa = sysb

        rhoa = ptr(p, dims, sysa)
        return tr_sqrt(rhoa)**2

    return norm(partial_transpose(p, dims, sysa), "tr")


@zeroify
def logneg(p, dims=(2, 2), sysa=0):
    """Compute logarithmic negativity between two subsytems.
    This is defined as  log_2( | rho_{AB}^{T_B} | ). This only handles
    bipartitions (including pure states efficiently), and will not trace
    anything out.

    Parameters
    ----------
    p : ket vector or density operator
        State to compute logarithmic negativity for.
    dims : tuple(int), optional
        The internal dimensions of ``p``.
    sysa : int, optional
        Index of the first subsystem, A, relative to ``dims``.

    Returns
    -------
    float

    See Also
    --------
    negativity, partial_transpose, logneg_subsys_approx
    """
    return max(0.0, log2(partial_transpose_norm(p, dims, sysa)))


logarithmic_negativity = logneg


def logneg_subsys(psi_abc, dims, sysa, sysb,
                  approx_thresh=2**13, **approx_opts):
    """Compute the logarithmic negativity between two subsystems of a pure
    state, possibly using an approximate lanczos for large subsystems. Uses
    a special method if the two subsystems form a bipartition of the state.

    Parameters
    ----------
    psi_abc : vector
        Tri-partite pure state.
    dims : sequence of int
        The sub dimensions of the state.
    sysa : sequence of int
        The index(es) of the subsystem(s) to consider part of 'A'.
    sysb : sequence of int
        The index(es) of the subsystem(s) to consider part of 'B'.
    approx_thresh : int, optional
        The size of subsystem at which to switch to the approximate lanczos
        method. Set to ``None`` to never use the approximation.
    approx_opts
        Supplied to :func:`~quimb.logneg_subsys_approx`, if used.

    Returns
    -------
    float
        The logarithmic negativity.

    See Also
    --------
    logneg, mutinf_subsys, logneg_subsys_approx
    """
    sysa, sysb = int2tup(sysa), int2tup(sysb)

    check_dims_and_indices(dims, sysa, sysb)

    sz_a = prod(d for i, d in enumerate(dims) if i in sysa)
    sz_b = prod(d for i, d in enumerate(dims) if i in sysb)
    sz_ab = sz_a * sz_b
    sz_c = prod(dims) // sz_ab

    # check for pure bipartition
    if sz_c == 1:
        psi_ab_ppt_norm = tr_sqrt_subsys(
            psi_abc, dims, sysa, approx_thresh=approx_thresh, **approx_opts)**2
        return max(log2(psi_ab_ppt_norm), 0.0)

    # check whether to use approx lanczos method
    if (approx_thresh is not None) and (sz_ab >= approx_thresh):
        return logneg_subsys_approx(psi_abc, dims, sysa, sysb, **approx_opts)

    rho_ab = ptr(psi_abc, dims, sysa + sysb)

    # need to adjust for new dimensions and indices
    new_dims, new_sysa = [], []
    new_inds = iter(range(len(dims)))

    for i, d in enumerate(dims):
        if i in sysa:
            new_dims.append(d)
            new_sysa.append(next(new_inds))
        elif i in sysb:
            new_dims.append(d)
            next(new_inds)  # don't need sysb

    return logneg(rho_ab, new_dims, new_sysa)


def negativity(p, dims=(2, 2), sysa=0):
    """Compute negativity between two subsytems.

    This is defined as  (| rho_{AB}^{T_B} | - 1) / 2. If ``len(dims) > 2``,
    then the non-target dimensions will be traced out first.

    Parameters
    ----------
    p : ket vector or density operator
        State to compute logarithmic negativity for.
    dims : tuple(int), optional
        The internal dimensions of ``p``.
    sysa : int, optional
        Index of the first subsystem, A, relative to ``dims``.

    Returns
    -------
    float

    See Also
    --------
    logneg, partial_transpose, negativity_subsys_approx
    """
    return max(0.0, (partial_transpose_norm(p, dims, sysa) - 1) / 2)


@zeroify
def concurrence(p, dims=(2, 2), sysa=0, sysb=1):
    """Concurrence of two-qubit state.

    If ``len(dims) > 2``, then the non-target dimensions will be traced out
    first.

    Parameters
    ----------
    p : ket vector or density operator
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

    Y = pauli('Y')

    if isop(p):
        pt = dot(kron(Y, Y), dot(p.conj(), kron(Y, Y)))
        evals = (nla.eigvals(dot(p, pt)).real**2)**0.25
        return max(0, 2 * np.max(evals) - np.sum(evals))
    else:
        pt = dot(kron(Y, Y), p.conj())
        c = np.real(abs(dot(dag(p), pt))).item(0)
        return max(0, c)


def one_way_classical_information(p_ab, prjs, precomp_func=False):
    """One way classical information for two qubit density operator.

    Parameters
    ----------
    p_ab : operator
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
    """Quantum Discord for two qubit density operator.

    If ``len(dims) > 2``, then the non-target dimensions will be traced out
    first.

    Parameters
    ----------
    p : ket vector or density operator
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
    p1 : ket or density operator
        The first state.
    p2 : ket or density operator
        The second state.

    Returns
    -------
    float
    """
    p1_is_op, p2_is_op = isop(p1), isop(p2)

    # If both are pure kets then special case
    if (not p1_is_op) and (not p2_is_op):
        return sqrt(1 - expec(p1, p2))

    # Otherwise do full calculation
    return 0.5 * norm((p1 if p1_is_op else dop(p1)) -
                      (p2 if p2_is_op else dop(p2)), "tr")


def decomp(a, fn, fn_args, fn_d, nmlz_func, mode="p", tol=1e-3):
    """Decomposes an operator via the Hilbert-schmidt inner product.

    Can both print the decomposition or return it.

    Parameters
    ----------
    a : ket or density operator
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
        Pauli operator name and expec with ``a``.

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
            cff = expec(a, op)
            yield "".join(str(x) for x in perm), cff

    names_cffs = list(calc_name_and_overlap())
    # sort by descending expec and turn into OrderedDict
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
                                 fn=pauli,
                                 fn_args='IXYZ',
                                 fn_d=2,
                                 nmlz_func=lambda n: 2**-n)
"""Decompose an operator into Paulis."""

bell_decomp = functools.partial(decomp,
                                fn=bell_state,
                                fn_args=(0, 1, 2, 3),
                                fn_d=4,
                                nmlz_func=lambda x: 1)
"""Decompose an operator into bell-states."""


def correlation(p, A, B, sysa, sysb, dims=None, sparse=None,
                precomp_func=False):
    """Calculate the correlation between two sites given two operators.

    Parameters
    ----------
    p : ket or density operator
        State to compute correlations for, ignored if ``precomp_func=True``.
    A : operator
        Operator to act on first subsystem.
    B : operator
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
        sparse = issparse(A) or issparse(B)

    opts = {'sparse': sparse,
            'coo_build': sparse,
            'stype': 'csr' if sparse else None}
    opab = ikron((A, B), dims, (sysa, sysb), **opts)
    A = ikron((A,), dims, sysa, **opts)
    B = ikron((B,), dims, sysb, **opts)

    @realify
    def corr(state):
        return expec(opab, state) - expec(A, state) * expec(B, state)

    return corr if precomp_func else corr(p)


def pauli_correlations(p, ss=("xx", "yy", "zz"), sysa=0, sysb=1,
                       sum_abs=False, precomp_func=False):
    """Calculate the correlation between sites for a list of operator pairs
    choisen from the pauli matrices.

    Parameters
    ----------
    p : ket or density operator
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
            yield correlation(p, pauli(s1), pauli(s2), sysa, sysb,
                              precomp_func=precomp_func)

    if sum_abs:

        if precomp_func:
            return lambda p: sum((abs(corr(p)) for corr in gen_corr_list()))

        return sum((abs(corr) for corr in gen_corr_list()))

    return tuple(gen_corr_list())


def ent_cross_matrix(p, sz_blc=1, ent_fn=logneg, calc_self_ent=True,
                     upscale=False):
    """Calculate the pair-wise function ent_fn  between all sites or blocks
    of a state.

    Parameters
    ----------
    p : ket or density operator
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
    upscale : bool, optional
        Whether, if sz_blc != 1, to upscale the results so that the output
        array is the same size as if it was.

    Returns
    -------
    2D-array
        array of pairwise ent_fn results.
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

    else:
        # Range over pairwise blocks
        for i in range(0, sz_p - sz_blc + 1, sz_blc):
            for j in range(i, sz_p - sz_blc + 1, sz_blc):
                if i == j:
                    if calc_self_ent:
                        rhoa = ptr(p, dims, [i + b for b in range(sz_blc)])
                        psiap = purify(rhoa)
                        ent = ent_fn(psiap,
                                     dims=(2**sz_blc, 2**sz_blc)) / sz_blc
                    else:
                        ent = np.nan
                else:
                    rhoab = ptr(p, dims, [i + b for b in range(sz_blc)] +
                                         [j + b for b in range(sz_blc)])
                    ent = ent_fn(rhoab, dims=(2**sz_blc, 2**sz_blc)) / sz_blc
                ents[i // sz_blc, j // sz_blc] = ent
                ents[j // sz_blc, i // sz_blc] = ent

    if upscale:
        up_ents = np.tile(np.nan, (sz_p, sz_p))

        for i in range(n):
            for j in range(i, n):
                up_ents[i * sz_blc:(i + 1) * sz_blc,
                        j * sz_blc:(j + 1) * sz_blc] = ents[i, j]
                up_ents[j * sz_blc:(j + 1) * sz_blc,
                        i * sz_blc:(i + 1) * sz_blc] = ents[j, i]

        ents = up_ents

    return ents


def qid(p, dims, inds, precomp_func=False, sparse_comp=True,
        norm_func=norm, power=2, coeff=1):
    # Check inputs
    inds = (inds,) if isinstance(inds, numbers.Number) else inds

    # Construct operators
    ops_i = tuple(tuple(ikron(pauli(s), dims, ind, sparse=sparse_comp)
                        for s in "xyz")
                  for ind in inds)

    # Define function closed over precomputed operators
    def qid_func(x):
        if isvec(x):
            x = dop(x)
        return tuple(sum(coeff * norm_func(dot(x, op) - dot(op, x))**power
                         for op in ops)
                     for ops in ops_i)

    return qid_func if precomp_func else qid_func(p)


def is_degenerate(op, tol=1e-12):
    """Check if operator has any degenerate eigenvalues, determined relative
    to mean spacing of all eigenvalues.

    Parameters
    ----------
    op : operator or 1d-array
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
        evals = eigvalsh(op)
    else:
        evals = op
    l_gaps = evals[1:] - evals[:-1]
    l_tol = tol * (evals[-1] - evals[0]) / op.shape[0]
    return np.count_nonzero(abs(l_gaps) < l_tol)


def is_eigenvector(x, A, tol=1e-14):
    """Determines whether a vector is an eigenvector of an operator.

    Parameters
    ----------
    x : vector
        Vector to check.
    A : operator
        Matrix to check.
    tol : float, optional
        The variance must be smaller than this value.

    Returns
    -------
    bool
        Whether ``A @ x = l * x`` for some scalar ``l``.
    """
    mat_vec = dot(A, x)
    E = expec(x, mat_vec)
    E2 = expec(x, dot(A, mat_vec))
    return abs(E**2 - E2) < tol


@njit
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
    if sz_subsys > sz_total**0.5:
        sz_subsys = sz_total // sz_subsys

    n = sz_total // sz_subsys

    s = 0
    for k in range(n + 1, sz_total + 1):
        s += 1 / k
    s -= (sz_subsys - 1) / (2 * n)

    # Normalize into bits of entropy
    return s / log(2)


def heisenberg_energy(L):
    """Get the analytic isotropic heisenberg chain ground energy for length L.
    Useful for testing. Assumes the heisenberg model is defined with spin
    operators not pauli matrices (overall factor of 2 smaller). Taken from [1].

    [1] Nickel, Bernie. "Scaling corrections to the ground state energy
    of the spin-½ isotropic anti-ferromagnetic Heisenberg chain." Journal of
    Physics Communications 1.5 (2017): 055021

    Parameters
    ----------
    L : int
        The length of the chain.

    Returns
    -------
    energy : float
        The ground state energy.
    """
    Einf = (0.5 - 2 * log(2)) * L
    Efinite = pi**2 / (6 * L)
    correction = 1 + 0.375 / log(L)**3
    return (Einf - Efinite * correction) / 2
