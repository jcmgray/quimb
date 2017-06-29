"""
Functions for more advanced calculations of quantities and properties of
quantum objects.
"""
# TODO: move matrix functions to solve, add slepc versions ****************** #
# TODO: all docs ************************************************************ #
# TODO: sparse sqrtm function *********************************************** #

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
    _dot_dense,
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
from .gen.operators import sig
from .gen.states import (
    basis_vec,
    bell_state,
    bloch_state,
)


def expm(a, herm=True):
    """Matrix exponential, can be accelerated if explicitly hermitian.
    """
    if issparse(a):
        # convert to and from csc to suppress scipy warning
        return spla.expm(a.tocsc()).tocsr()
    elif not herm:
        return np.asmatrix(spla.expm(a))
    else:
        evals, evecs = eigsys(a)
        return _dot_dense(evecs, ldmul(np.exp(evals), evecs.H))


def sqrtm(a, herm=True):
    """Matrix square root, can be accelerated if explicitly hermitian.
    """
    if issparse(a):
        raise NotImplementedError("No sparse sqrtm available.")
    elif not herm:
        return np.asmatrix(sla.sqrtm(a))
    else:
        evals, evecs = eigsys(a)
        return _dot_dense(evecs, ldmul(np.sqrt(evals.astype(complex)),
                                       evecs.H))


def fidelity(rho, sigma):
    """Fidelity between two quantum states
    """
    if isvec(rho) or isvec(sigma):
        return overlap(rho, sigma)
    else:
        sqrho = sqrtm(rho)
        return tr(sqrtm(dot(sqrho, dot(sigma, sqrho))))
        # return norm(sqrtm(rho) @ sqrtm(sigma), "tr")


def purify(rho, sparse=False):
    """Take state rho and purify it into a wavefunction of squared
    dimension.
    """
    # TODO: trim zeros?
    d = rho.shape[0]
    evals, vs = eigsys(rho)
    evals = np.sqrt(np.clip(evals, 0, 1))
    psi = np.zeros(shape=(d**2, 1), dtype=complex)
    for i, evals in enumerate(evals.flat):
        psi += evals * kron(vs[:, i], basis_vec(i, d, sparse=sparse))
    return qu(psi)


@zeroify
def entropy(a, rank=None):
    """Compute the (von Neumann) entropy

    Parameters
    ----------
        a :
            Operator or list of eigenvalues.
        rank : int (optional)
            If operator has known rank, then a partial decomposition can be
            used to acclerate the calculation.

    Returns
    -------
        e: von neumann entropy
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
    """Find the mutual information between two subystems of a state

    Parameters
    ----------
        p: state, can be vector or operator
        dims: internal dimensions of state
        sysa: index of first subsystem
        sysb: index of second subsystem
        rank: if known, the rank of rho_ab, to speed calculation up

    Returns
    -------
        Ixy: mutual information
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
    """Partial transpose of state `p` with bipartition as given by
    `dims`.
    """
    p = qu(p, "dop")
    p = np.array(p)  \
        .reshape((*dims, *dims))  \
        .transpose((2, 1, 0, 3))  \
        .reshape((prod(dims), prod(dims)))
    return qu(p)


@zeroify
def negativity(p, dims=(2, 2), sysa=0, sysb=1):
    """Negativity between `sysa` and `sysb` of state `p` with subsystem
    dimensions `dims`
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
    """Logarithmic negativity between `sysa` and `sysb` of `p`, with
    subsystem dimensions `dims`.
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
    """Concurrence of two-qubit state `p`.
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
        p_ab: state of two qubits
        prjs: iterable of POVMs
        precomp_func: whether to return a pre-computed function, closed over
            the actual state.

    Returns
    -------
        The one-way classical information or the function to compute such
        given a set of POVMs
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
    """Quantum Discord for two qubit density matrix `p`.
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
def trace_distance(p, w):
    """Trace distance between states `p` and `w`.
    """
    p_is_op, w_is_op = isop(p), isop(w)
    if not p_is_op and not w_is_op:
        return sqrt(1 - overlap(p, w))
    return 0.5 * norm((p if p_is_op else dop(p)) -
                      (w if w_is_op else dop(w)), "tr")


def decomp(a, fn, fn_args, fn_d, nmlz_func, mode="p", tol=1e-3):
    """Decomposes an operator via the Hilbert-schmidt inner product into the
    pauli group. Can both print the decomposition or return it.

    Parameters
    ----------
        a: operator to decompose
        fn: function to generate operator/state to decompose with
        fn_args: list of args whose permutations will be used
        mode: string, include 'p' to print the decomp and/or 'c' to return dict
        tol: print operators with contirbution above tol only.

    Returns
    -------
        (names_cffs): OrderedDict of Pauli operator name and overlap with a.
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
        p: state
        opa: operator to act on first subsystem
        opb: operator to act on second subsystem
        sysa: index of first subsystem
        sysb: index of second subsystem
        sparse: whether to compute with sparse operators
        precomp_func: whether to return result or single arg function closed
            over precomputed operators

    Returns
    -------
        cab: correlation, <ab> - <a><b>
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
    """Calculate the correlation between sites for a list of operator pairs.

    Parameters
    ----------
        p: state
        sysa: index of first site
        sysb: index of second site
        ss: list of pairs specifiying pauli matrices
        sum_abs: whether to sum over the absolute values of each correlation
        precomp_func: whether to return the values or a single argument
            function closed over precomputed operators etc.

    Returns
    -------
        corrs: list of values or functions specifiying each correlation.
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
        ents: matrix of pairwise ent_fn results.
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
    to equal spacing of all eigenvalues.

    Parameters
    ----------
        op : list or operator-like
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
            Entropy.
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
