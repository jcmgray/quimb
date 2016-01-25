"""
Functions for more advanced calculations of quantities and properties of
quantum objects.
"""

from itertools import product
from collections import OrderedDict
import numpy as np
from quijy.core import (isket, isbra, isop, qonvert, kron, ldmul, comm,
                        eyepad, tr)
from quijy.gen import sig
from quijy.solve import eigvals, eigsys, norm2


def trx(p, dims, keep):
    """ Perform partial trace.
    Input:
        p: state to perform partial trace on, vector or operator
        dims: list of subsystem dimensions
        keep: index of subsytems to keep
    Returns:
        Density matrix of subsytem dimensions dims[keep]
    """
    # Cast as ndarrays for 2D+ reshaping
    if np.size(keep) == np.size(dims):  # keep all subsystems
        if not isop(p):
            return p * p.H  # but return as density operator for consistency
        return p
    n = len(dims)
    dims = np.array(dims)
    keep = np.array(keep)
    keep = np.reshape(keep, [keep.size])  # make sure array
    lose = np.delete(np.arange(n), keep)
    dimkeep = np.prod(dims[keep])
    dimlose = np.prod(dims[lose])
    # Permute dimensions into block of keep and block of lose
    perm = np.r_[keep, lose]
    # Apply permutation to state and trace out block of lose
    if not isop(p):  # p = psi
        p = np.array(p)
        p = p.reshape(dims) \
            .transpose(perm) \
            .reshape([dimkeep, dimlose])
        p = np.matrix(p, copy=True)
        return qonvert(p * p.H)
    else:  # p = rho
        p = np.array(p)
        p = p.reshape(np.r_[dims, dims]) \
            .transpose(np.r_[perm, perm + n]) \
            .reshape([dimkeep, dimlose, dimkeep, dimlose]) \
            .trace(axis1=1, axis2=3)
        return qonvert(p)


def partial_transpose(p, dims=[2, 2]):
    """
    Partial transpose
    """
    if isket(p):  # state vector supplied
        p = p * p.H
    elif isbra(p):
        p = p.H * p
    p = np.array(p)\
        .reshape(np.concatenate((dims, dims)))  \
        .transpose([2, 1, 0, 3])  \
        .reshape([np.prod(dims), np.prod(dims)])
    return qonvert(p)


def entropy(rho):
    """
    Computes the (von Neumann) entropy of positive matrix rho
    """
    l = eigvals(rho)
    l = l[np.nonzero(l)]
    return np.sum(-l * np.log2(l))


def mutual_information(p, dims=[2, 2], sysa=0, sysb=1):
    """
    Partitions rho into dims, and finds the mutual information between the
    subsystems at indices sysa and sysb
    """
    if isop(p) or len(dims) > 2:  # mixed combined system
        rhoab = trx(p, dims, [sysa, sysb])
        rhoa = trx(rhoab, np.r_[dims[sysa], dims[sysb]], 0)
        rhob = trx(rhoab, np.r_[dims[sysa], dims[sysb]], 1)
        hab = entropy(rhoab)
        ha = entropy(rhoa)
        hb = entropy(rhob)
    else:  # pure combined system
        hab = 0.0
        rhoa = trx(p, dims, sysa)
        ha = entropy(rhoa)
        hb = ha
    return ha + hb - hab


def sqrtm(a):
    # returns sqrt of hermitan matrix, seems faster than scipy.linalg.sqrtm
    l, v = eigsys(a, sort=False)
    l = np.sqrt(l.astype(complex))
    return v * ldmul(l, v.H)


def trace_norm(a):
    """
    Returns the trace norm of operator a, that is, the sum of abs eigvals.
    """
    return np.absolute(eigvals(a, sort=False)).sum()


def trace_distance(p, w):
    return 0.5 * trace_norm(p - w)


def negativity(rho, dims=[2, 2], sysa=0, sysb=1):
    if np.size(dims) > 2:
        rho = trx(rho, dims, [sysa, sysb])
    n = (trace_norm(partial_transpose(rho)) - 1.0) / 2.0
    return max(0.0, n)


def logneg(rho, dims=[2, 2], sysa=0, sysb=1):
    if np.size(dims) > 2:
        rho = trx(rho, dims, [sysa, sysb])
        dims = [dims[sysa], dims[sysb]]
    e = np.log2(trace_norm(partial_transpose(rho, dims)))
    return max(0.0, e)


def concurrence(p):
    if isop(p):
        p = qonvert(p, 'dop')  # make sure density operator
        pt = kron(sig(2), sig(2)) * p.conj() * kron(sig(2), sig(2))
        l = np.sqrt(np.linalg.eigvals(p * pt)).real
        return max(0, 2 * np.max(l) - np.sum(l))
    else:
        p = qonvert(p, 'ket')
        pt = kron(sig(2), sig(2)) * p.conj()
        c = np.real(np.abs(p.H * p)).item(0)
        return max(0, c)


def qid(p, dims, inds, precomp_func=False, sparse_comp=True):
    inds = np.atleast_1d(inds)
    # Construct operators
    ops_i = list([list([eyepad(sig(s), dims, ind, sparse=sparse_comp)
                        for s in (1, 2, 3)])
                  for ind in inds])

    # Define function closed over precomputed operators
    def qid_func(x):
        qds = np.zeros(np.size(inds))
        for i, ops in enumerate(ops_i):
            for op in ops:
                qds[i] += 0.5 * norm2(comm(x, op))**2
        return np.sqrt(qds)

    return qid_func if precomp_func else qid_func(p)


def pauli_decomp(a, mode='p', tol=1e-3):
    """
    Decomposes an operator via the Hilbert-schmidt inner product into the
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
    a = qonvert(a, 'dop')  # make sure operator
    n = int(np.log2(a.shape[0]))  # infer number of qubits

    # define generator for inner product to iterate over efficiently
    def calc_name_and_overlap(fa):
        for perm in product('IXYZ', repeat=n):
            name = "".join(perm)
            op = kron(*[sig(s, sparse=True) for s in perm]) / 2**n
            d = tr(a * op)
            yield name, d
    nds = [nd for nd in calc_name_and_overlap(a)]

    # sort by descending overlap and turn into OrderedDict
    nds.sort(key=lambda pair: -abs(pair[1]))
    nds = OrderedDict(nds)

    # Print decomposition
    if 'p' in mode:
        for x, d in nds.items():
            if abs(d) < 0.01:
                break
            dps = int(round(0.5 - np.log10(1.001 * tol)))  # dec places to show
            print(x, '{: .{prec}f}'.format(d, prec=dps))

    # Return full calculation
    if 'c' in mode:
        return nds
