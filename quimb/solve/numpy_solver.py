import numpy as np
import numpy.linalg as nla

from .. import issparse


def sort_inds(a, method, sigma=None):
    """
    Return the sorting inds of a list

    Parameters
    ----------
        a: list to base sort on
        method: method of sorting list
        sigma: target

    Returns
    -------
        inds: indices that would sort `a` based on `method`
    """
    sfunc = {"LM": lambda a: -abs(a),
             "SM": lambda a: -abs(1/a),
             "SA": lambda a: a,
             "SR": lambda a: a.real,
             "SI": lambda a: a.imag,
             "LA": lambda a: -a,
             "LR": lambda a: -a.real,
             "LI": lambda a: -a.imag,
             "TM": lambda a: -1/abs(abs(a) - sigma),
             "TR": lambda a: -1/abs(a.real - sigma),
             "TI": lambda a: -1/abs(a.imag - sigma)}[method.upper()]
    return np.argsort(sfunc(a))


def numpy_seigsys(a, k=6, which=None, return_vecs=True, sigma=None,
                  isherm=True, ncv=None, sort=True, **kwargs):
    """
    Partial eigen-decomposition using numpy's dense linear algebra.

    Parameters
    ----------
        a: operator to partially eigen-decompose
        k: number of eigenpairs to return
        which: which part of the spectrum to target
        return_vecs: whether to return eigenvectors
        sigma: target eigenvalue
        isherm: whether `a` is hermitian
        ncv: (for compatibility purposes only)
        sort: (for compatibility purposes only)
        **kwargs: settings to pass to numpy.eig... functions

    Returns
    -------
        lk, (vk): k eigenvalues (and eigenvectors) sorted according to which
    """
    which = ("SA" if which is None and sigma is None else
             "TM" if which is None and sigma is not None else
             which)

    efunc = {(True, True): nla.eigh,
             (True, False): nla.eigvalsh,
             (False, True): nla.eig,
             (False, False): nla.eigvals}[(isherm, return_vecs)]

    if return_vecs:
        l, v = efunc(a.A if issparse(a) else a, **kwargs)
        sk = sort_inds(l, method=which, sigma=sigma)[:k]
        return l[sk], v[:, sk]
    else:
        l = efunc(a.A if issparse(a) else a, **kwargs)
        sk = sort_inds(l, method=which, sigma=sigma)[:k]
        return l[sk]
