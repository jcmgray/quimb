import pytest
import numpy as np
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


def rand_rect(m, n, sparse=False, dtype=complex):
    X = qu.rand_matrix(max(m, n), dtype=dtype, sparse=sparse)
    return X[:m, :n]


def usv2dense(U, s, VH):
    return U @ np.diag(s) @ VH


def rand_rank(m, n, k, dtype=complex):
    s = np.sort(qu.randn(k)**2)[::-1]

    U = qu.gen.rand.rand_iso(m, k, dtype=dtype)
    VH = qu.gen.rand.rand_iso(n, k, dtype=dtype).conj().T

    if U.dtype in ('float32', 'complex64'):
        s = s.astype('float32')

    return usv2dense(U, s, VH)


def rand_tn1d_sect(n, bd, dtype=complex):
    mps = qtn.MPS_rand_state(n + 2, bd, dtype=dtype)
    mpo = qtn.MPO_rand_herm(n + 2, 5, dtype=dtype)

    norm = qtn.TensorNetwork(qtn.tensor_network_align(mps.H, mpo, mps))
    norm.view_as_(qtn.TensorNetwork1D, like=mps)

    lix = qtn.bonds(norm[0], norm[1])
    rix = qtn.bonds(norm[n], norm[n + 1])

    to = norm[1:n + 1]

    return qtn.TNLinearOperator1D(to, lix, rix, 1, n + 1)


dtypes = ['float32', 'float64', 'complex64', 'complex128']


class TestRSVD:

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('shape', [(41, 31), (31, 41)])
    @pytest.mark.parametrize('sparse', [False, True])
    @pytest.mark.parametrize('q', [2, 3])
    @pytest.mark.parametrize('p', [0, 5])
    def test_rsvd(self, dtype, shape, sparse, q, p):
        X = rand_rect(*shape, dtype=dtype, sparse=sparse)

        k = 15
        U, s, V = qu.rsvd(X, k, q=q, p=p)

        assert U.shape == (shape[0], k)
        assert s.shape == (k,)
        assert V.shape == (k, shape[1])

        assert U.dtype == dtype
        assert V.dtype == dtype

        assert_allclose(U.conj().T @ U, np.eye(k), rtol=1e-5, atol=1e-5)
        assert_allclose(V @ V.conj().T, np.eye(k), rtol=1e-5, atol=1e-5)

        Ue, se, Ve = qu.svds(X, k)
        opt_err = qu.norm(X.A - usv2dense(Ue, se, Ve), 'fro')
        act_err = qu.norm(X.A - usv2dense(U, s, V), 'fro')

        assert act_err < 1.2 * opt_err

        assert_allclose(s[:k // 2], se[:k // 2], rtol=0.05)

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('shape', [(41, 31), (31, 41)])
    @pytest.mark.parametrize('q', [2, 3])
    @pytest.mark.parametrize('p', [0, 5])
    def test_rsvd_adaptive(self, dtype, shape, q, p):
        X = rand_rank(*shape, 10, dtype=dtype)
        U, s, V = qu.rsvd(X, 1e-6, q=q, p=p, k_start=10)

        k = s.size
        assert 10 <= k <= 20

        assert U.dtype == dtype
        assert V.dtype == dtype

        assert_allclose(U.conj().T @ U, np.eye(k), rtol=1e-6, atol=1e-6)
        assert_allclose(V @ V.conj().T, np.eye(k), rtol=1e-6, atol=1e-6)

        Ue, se, Ve = qu.svds(X, k)
        act_err = qu.norm(X - usv2dense(U, s, V), 'fro')

        assert act_err < 1e-4

        assert_allclose(s[:k // 2], se[:k // 2], rtol=0.1)

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('shape', [(410, 310), (310, 410)])
    @pytest.mark.parametrize('k_start', [4, 10, 16])
    @pytest.mark.parametrize('use_qb', [False, 10, True])
    def test_estimate_rank(self, dtype, shape, k_start, use_qb):
        rnk = 100
        X = rand_rank(*shape, rnk, dtype=dtype)

        Ue, se, VHe = qu.svd(X)
        assert_allclose(se[rnk:], 0.0, atol=1e-5)

        k = qu.estimate_rank(X, 1e-3, k_start=k_start,
                             use_qb=use_qb, use_sli=False)
        assert_allclose(k, 100, rtol=0.3)

        assert qu.estimate_rank(X, 1e-3, k_start=k_start, k_max=50) == 50

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('k_start', [2, 4, 8])
    @pytest.mark.parametrize('use_qb', [False, 10, True])
    def test_estimate_rank_lo(self, dtype, k_start, use_qb):
        X = rand_tn1d_sect(30, 10, dtype=dtype)

        Ue, se, VHe = qu.svd(X.A)
        actual_rank = sum(se > se[0] * 1e-3)

        k = qu.estimate_rank(X, 1e-3, k_start=k_start,
                             use_qb=use_qb, use_sli=False)
        assert_allclose(k, actual_rank, rtol=0.3 if use_qb else 0.5)

        assert qu.estimate_rank(X, 1e-3, k_start=k_start, k_max=8) == 8
