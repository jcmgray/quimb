"""
Functions for more advanced calculations of quantities and properties of
quantum objects.
"""
# TODO: move matrix functions to solve, add slepc versions ****************** #
# TODO: all docs ************************************************************ #
# TODO: sparse sqrtm function *********************************************** #

from math import sin, cos, pi, log2, sqrt
import collections
import itertools

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.optimize import minimize

from .accel import dot_dense, ldmul, issparse, isop, zeroify, realify
from .core import (qu, kron, eye, eyepad, tr, ptr, infer_size, overlap)
from .solve import (eigvals, eigsys, norm)
from .gen import sig, basis_vec, bell_state, bloch_state


def expm(a, herm=True):
    """ Matrix exponential, can be accelerated if explicitly hermitian. """
    if issparse(a):
        return spla.expm(a)
    elif not herm:
        return np.asmatrix(spla.expm(a))
    else:
        l, v = eigsys(a)
        return dot_dense(v, ldmul(np.exp(l), v.H))


def sqrtm(a, herm=True):
    """ Matrix square root, can be accelerated if explicitly hermitian. """
    if issparse(a):
        raise NotImplementedError("No sparse sqrtm available.")
    elif not herm:
        return np.asmatrix(sla.sqrtm(a))
    else:
        l, v = eigsys(a)
        return dot_dense(v, ldmul(np.sqrt(l.astype(complex)), v.H))


def fidelity(rho, sigma):
    if not isop(rho) or not isop(sigma):
        return overlap(rho, sigma)
    else:
        sqrho = sqrtm(rho)
        return tr(sqrtm(sqrho @ sigma @ sqrho))
        # return norm(sqrtm(rho) @ sqrtm(sigma), "tr")


def purify(rho, sparse=False):
    """ Take state rho and purify it into a wavefunction of squared
    dimension. """
    # TODO: trim zeros?
    d = rho.shape[0]
    ls, vs = eigsys(rho)
    ls = np.sqrt(ls)
    psi = np.zeros(shape=(d**2, 1), dtype=complex)
    for i, l in enumerate(ls.flat):
        psi += l * kron(vs[:, i], basis_vec(i, d, sparse=sparse))
    return qu(psi)


@zeroify
def entropy(a):
    """ Computes the (von Neumann) entropy of positive matrix `a`. """
    a = np.asarray(a)
    if np.ndim(a) == 1:
        l = a
    else:
        l = eigvals(a)
    l = l[l > 0.0]
    return np.sum(-l * np.log2(l))


@zeroify
def mutual_information(p, dims=[2, 2], sysa=0, sysb=1):
    """ Partitions `p` into `dims`, and finds the mutual information between
    the subsystems at indices `sysa` and `sysb` """
    if isop(p) or np.size(dims) > 2:  # mixed combined system
        rhoab = ptr(p, dims, (sysa, sysb))
        rhoa = ptr(rhoab, (dims[sysa], dims[sysb]), 0)
        rhob = ptr(rhoab, (dims[sysa], dims[sysb]), 1)
        hab = entropy(rhoab)
        ha, hb = entropy(rhoa), entropy(rhob)
    else:  # pure combined system
        hab = 0.0
        rhoa = ptr(p, dims, sysa)
        ha = entropy(rhoa)
        hb = ha
    return ha + hb - hab


def partial_transpose(p, dims=[2, 2]):
    """ Partial transpose of state `p` with bipartition as given by
    `dims`. """
    p = qu(p, "dop")
    p = np.array(p)\
        .reshape((*dims, *dims))  \
        .transpose((2, 1, 0, 3))  \
        .reshape((np.prod(dims), np.prod(dims)))
    return qu(p)


@zeroify
def negativity(p, dims=[2, 2], sysa=0, sysb=1):
    """ Negativity between `sysa` and `sysb` of state `p` with subsystem
    dimensions `dims` """
    if not isop(p):
        p = qu(p, qtype='dop')
    if len(dims) > 2:
        p = ptr(p, dims, [sysa, sysb])
        dims = [dims[sysa], dims[sysb]]
    n = (norm(partial_transpose(p, dims=dims), "tr") - 1.0) / 2.0
    return max(0.0, n)


@zeroify
def logarithmic_negativity(p, dims=[2, 2], sysa=0, sysb=1):
    """ Logarithmic negativity between `sysa` and `sysb` of `p`, with
    subsystem dimensions `dims`. """
    if not isop(p):
        p = qu(p, qtype='dop')
    if len(dims) > 2:
        p = ptr(p, dims, [sysa, sysb])
        dims = [dims[sysa], dims[sysb]]
    e = log2(norm(partial_transpose(p, dims), "tr"))
    return max(0.0, e)

logneg = logarithmic_negativity


@zeroify
def concurrence(p):
    """ Concurrence of two-qubit state `p`. """
    if isop(p):
        p = qu(p, "dop")  # make sure density operator
        pt = kron(sig(2), sig(2)) @ p.conj() @ kron(sig(2), sig(2))
        l = (nla.eigvals(p @ pt).real**2)**0.25
        return max(0, 2 * np.max(l) - np.sum(l))
    else:
        p = qu(p, "ket")
        pt = kron(sig(2), sig(2)) @ p.conj()
        c = np.real(abs(p.H @ pt)).item(0)
        return max(0, c)


def one_way_classical_information(p_ab, prjs, precomp_func=False):
    """ One way classical information for two qubit density matrix.

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
    p_a = ptr(p_ab, [2, 2], 0)
    s_a = entropy(p_a)

    def owci(prjs):
        def gen_paj():
            for prj in prjs:
                p_ab_j = (eye(2) & prj) @ p_ab
                prob = tr(p_ab_j)
                p_a_j = ptr(p_ab_j, [2, 2], 0) / prob
                yield prob, p_a_j
        return s_a - sum(p * entropy(rho) for p, rho in gen_paj())

    return owci if precomp_func else owci(prjs)


@zeroify
def quantum_discord(p):
    """ Quantum Discord for two qubit density matrix `p`. """
    p = qu(p, "dop")
    iab = mutual_information(p)
    owci = one_way_classical_information(p, None, precomp_func=True)

    def trial_qd(a):
        ax, ay, az = sin(a[0]) * cos(a[1]), sin(a[0]) * sin(a[1]), cos(a[0])
        prja = bloch_state(ax, ay, az)
        prjb = eye(2) - prja
        return iab - owci([prja, prjb])

    opt = minimize(trial_qd, (pi/2, pi),
                   method="SLSQP", bounds=((0, pi), (0, 2 * pi)))
    if opt.success:
        return opt.fun
    else:  # pragma: no cover
        raise ValueError(opt.message)


@zeroify
def trace_distance(p, w):
    """ Trace distance between states `p` and `w`. """
    if not isop(p) and not isop(w):
        return sqrt(1 - overlap(p, w))
    return 0.5 * norm(p - w, "tr")


def pauli_decomp(a, mode="p", tol=1e-3):
    """ Decomposes an operator via the Hilbert-schmidt inner product into the
    pauli group. Can both print the decomposition or return it.

    Parameters
    ----------
        a: operator to decompose
        mode: string, include 'p' to print the decomp and/or 'c' to return dict
        tol: print operators with contirbution above tol only.

    Returns
    -------
        nds: OrderedDict of Pauli operator name and overlap with a.

    Examples
    --------
    >>> pauli_decomp( singlet(), tol=1e-2)
    II  0.25
    XX -0.25
    YY -0.25
    ZZ -0.25
    """
    if not isop(a):
        a = qu(a, "dop")  # make sure operator
    n = infer_size(a)

    # define generator for inner product to iterate over efficiently
    def calc_name_and_overlap(fa):
        for perm in itertools.product("IXYZ", repeat=n):
            name = "".join(perm)
            op = kron(*[sig(s, sparse=True) for s in perm]) / 2**n
            d = np.trace(a @ op)
            yield name, d

    nds = [nd for nd in calc_name_and_overlap(a)]
    # sort by descending overlap and turn into OrderedDict
    nds.sort(key=lambda pair: -abs(pair[1]))
    nds = collections.OrderedDict(nds)
    # Print decomposition
    if "p" in mode:
        for x, d in nds.items():
            if abs(d) < 0.01:
                break
            dps = int(round(0.5 - np.log10(1.001 * tol)))  # decimal places
            print(x, "{: .{prec}f}".format(d, prec=dps))
    # Return full calculation
    if "c" in mode:
        return nds


def bell_fid(p):
    # TODO: bell_decomp, with arbitrary size, print and calc
    """ Outputs a tuple of state p's fidelities with the four bell states
    psi- (singlet) psi+, phi-, phi+ (triplets). """
    op = isop(p)

    def gen_bfs():
        for b in ["psi-", "psi+", "phi-", "phi+"]:
            psib = bell_state(b)
            if op:
                yield tr(psib.H @ p @ psib)
            else:
                yield abs(psib.H @ p)[0, 0]**2

    return [*gen_bfs()]


def correlation(p, opa, opb, sysa, sysb, dims=None, sparse=None,
                precomp_func=False):
    """ Calculate the correlation between two sites given two operators.

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
        dims = [2] * sz_p
    if sparse is None:
        sparse = issparse(opa) or issparse(opb)

    opts = {'sparse': sparse,
            'coo_build': sparse,
            'stype': 'csr' if sparse else None}
    opab = eyepad([opa, opb], dims, (sysa, sysb), **opts)
    opa = eyepad([opa], dims, sysa, **opts)
    opb = eyepad([opb], dims, sysb, **opts)

    @realify
    def corr(state):
        return overlap(opab, state) - overlap(opa, state) * overlap(opb, state)

    return corr if precomp_func else corr(p)


def pauli_correlations(p, ss=("xx", "yy", "zz"), sysa=0, sysb=1,
                       sum_abs=False, precomp_func=False):
    """ Calculate the correlation between sites for a list of operator pairs.

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


def ent_cross_matrix(p, ent_fun=concurrence, calc_self_ent=True):
    """ Calculate the pair-wise function ent_fun  between all sites
    of a state.

    Parameters
    ----------
        p: state
        ent_fun: function acting on space [2, 2], notionally entanglement
        calc_self_ent: whether to calculate the function for each site
            alone, purified. If the whole state is pure then this is the
            entanglement with the whole remaining system.

    Returns
    -------
        ents: matrix of pairwise ent_fun results.
    """
    sz_p = infer_size(p)
    ents = np.empty((sz_p, sz_p))
    for i in range(sz_p):
        for j in range(i, sz_p):
            if i == j:
                if calc_self_ent:
                    rhoa = ptr(p, [2]*sz_p, i)
                    psiap = purify(rhoa)
                    ent = ent_fun(psiap)
                else:
                    ent = np.nan
            else:
                rhoab = ptr(p, [2]*sz_p, [i, j])
                ent = ent_fun(rhoab)
            ents[i, j] = ent
            ents[j, i] = ent
    return ents


def qid(p, dims, inds, precomp_func=False, sparse_comp=True,
        norm_func=norm, pow=2, coeff=1/3):
    p = qu(p, "dop")
    inds = np.array(inds, ndmin=1)
    # Construct operators
    ops_i = [[eyepad(sig(s), dims, ind, sparse=sparse_comp)
              for s in "xyz"]
             for ind in inds]

    # Define function closed over precomputed operators
    def qid_func(x):
        qds = np.zeros(np.size(inds))
        for i, ops in enumerate(ops_i):
            for op in ops:
                qds[i] += coeff * norm_func(x @ op - op @ x)**pow
        return qds

    return qid_func if precomp_func else qid_func(p)
