"""Backend agnostic array operations.
"""

import numpy
from autoray import do, reshape, transpose, dag

from ..linalg.base_linalg import norm_fro_dense


# ------------- miscelleneous other backend agnostic functions -------------- #

def iscomplex(x):
    if not hasattr(x, 'dtype'):
        return isinstance(x, complex)
    return 'complex' in str(x.dtype)


def norm_fro(x):
    if isinstance(x, numpy.ndarray):
        return norm_fro_dense(x.reshape(-1))
    try:
        return do('linalg.norm', reshape(x, [-1]), 2)
    except AttributeError:
        return do('sum', do('multiply', do('conj', x), x)) ** 0.5


def _unitize_qr(x):
    """Perform isometrization using the QR decomposition.
    """
    fat = x.shape[0] < x.shape[1]
    if fat:
        x = transpose(x)

    Q = do('linalg.qr', x)[0]
    if fat:
        Q = transpose(Q)

    return Q


def _unitize_exp(x):
    r"""Perform isometrization using the using anti-symmetric matrix
    exponentiation.

    .. math::

            U_A = \exp{A - A^\dagger}

    If ``x`` is rectangular it is completed with zeros first.
    """
    m, n = x.shape
    d = max(m, n)
    x = do('pad', x, [[0, d - m], [0, d - n]], 'constant', constant_values=0.0)
    expx = do('linalg.expm', x - dag(x))
    return expx[:m, :n]


def _unitize_modified_gram_schmidt(A):
    """Perform isometrization explicitly using the modified Gram Schmidt
    procedure.
    """
    m, n = A.shape

    thin = m > n
    if thin:
        A = do('transpose', A)

    Q = []
    for j in range(0, min(m, n)):

        q = A[j, :]
        for i in range(0, j):
            rij = do('tensordot', do('conj', Q[i]), q, 1)
            q = q - rij * Q[i]

        Q.append(q / do('linalg.norm', q, 2))

    Q = do('stack', Q, axis=0, like=A)

    if thin:
        Q = do('transpose', Q)

    return Q


_UNITIZE_METHODS = {
    'qr': _unitize_qr,
    'exp': _unitize_exp,
    'mgs': _unitize_modified_gram_schmidt,
}


def unitize(x, method='qr'):
    """Generate a isometric (or unitary if square) matrix from array ``x``.

    Parameters
    ----------
    x : array
        The matrix to generate the isometry from.
    method : {'qr', 'exp', 'mgs'}, optional
        The method used to generate the isometry. Note ``'qr'`` is the fastest
        and most robust but, for example, some libraries cannot back-propagate
        through it.
    """
    return _UNITIZE_METHODS[method](x)
