from pytest import fixture
import numpy as np
from numpy.testing import assert_allclose
from ... import ldmul, rand_uni, issparse, qu, rand_product_state
from ...solve import (eigsys, eigvals, eigvecs, seigvals, seigvecs,
                      seigsys, groundstate, groundenergy, svds, norm,
                      choose_ncv, svd)


@fixture
def premat():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    return u, a


@fixture
def prematsparse():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qu(a, sparse=True)
    return u, a


@fixture
def svdpremat():
    u, v = rand_uni(5), rand_uni(5)
    a = u @ ldmul(np.array([1, 2, 4, 3, 0]), v.H)
    return u, v, a


@fixture
def svdprematsparse():
    u, v = rand_uni(5), rand_uni(5)
    a = u @ ldmul(np.array([1, 2, 4, 3, 0]), v.H)
    a = qu(a, sparse=True)
    return u, v, a


class TestEigh:
    def test_eigsys(self, premat):
        u, a = premat
        l, _ = eigsys(a, sort=False)
        assert(set(np.rint(l)) == set((-1, 2, 4, -3)))
        l, v = eigsys(a)
        assert_allclose(l, [-3, -1, 2, 4])
        for i, j in zip([3, 0, 1, 2], range(4)):
            o = u[:, i].H @ v[:, j]
            assert_allclose(abs(o), 1.)

    def test_eigvals(self, premat):
        _, a = premat
        l = eigvals(a)
        assert_allclose(l, [-3, -1, 2, 4])

    def test_eigvecs(self, premat):
        u, a = premat
        v = eigvecs(a)
        for i, j in zip([3, 0, 1, 2], range(4)):
            o = u[:, i].H @ v[:, j]
            assert_allclose(abs(o), 1.)


class TestChooseNCV:
    def test_choose_ncv(self):
        assert(choose_ncv(1, 100) == 20)
        assert(choose_ncv(15, 100) == 31)
        assert(choose_ncv(50, 100) == 100)


class TestSeigs:
    def test_seigsys_small_dense_wvecs(self, premat):
        u, a = premat
        assert not issparse(a)
        lk, vk = seigsys(a, k=2)
        assert_allclose(lk, (-3, -1))
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, i].H @ vk[:, j]
            assert_allclose(abs(o), 1.)
        vk = seigvecs(a, k=2)
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, i].H @ vk[:, j]
            assert_allclose(abs(o), 1.)

    def test_seigsys_small_dense_novecs(self, premat):
        _, a = premat
        assert not issparse(a)
        lk = seigvals(a, k=2)
        assert_allclose(lk, (-3, -1))

    def test_seigsys_sparse_wvecs(self, prematsparse):
        u, a = prematsparse
        assert issparse(a)
        lk, vk = seigsys(a, k=2)
        assert_allclose(lk, (-3, -1))
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, i].H @ vk[:, j]
            assert_allclose(abs(o), 1.)
        vk = seigvecs(a, k=2)
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, i].H @ vk[:, j]
            assert_allclose(abs(o), 1.)

    def test_seigsys_small_sparse_novecs(self, prematsparse):
        _, a = prematsparse
        assert issparse(a)
        lk = seigvals(a, k=2)
        assert_allclose(lk, (-3, -1))

    def test_groundstate(self, premat):
        u, a = premat
        gs = groundstate(a)
        assert_allclose(abs(u[:, 3].H @ gs), 1.)

    def test_groundenergy(self, premat):
        _, a = premat
        ge = groundenergy(a)
        assert_allclose(ge, -3)


class TestSVD:
    def test_svd_full(self, svdpremat):
        u, v, a = svdpremat
        un, sn, vn = svd(a)
        assert_allclose(sn, [4, 3, 2, 1, 0], atol=1e-14)
        for i, j, in zip((0, 1, 2, 3, 4),
                         (2, 3, 1, 0, 4)):
            o = abs(un[:, i].H @ u[:, j])
            assert_allclose(o, 1.)
            o = abs(vn[i, :] @ v[:, j])
            assert_allclose(o, 1.)


class TestSVDS:
    def test_svds_smalldense_wvecs(self, svdpremat):
        u, v, a = svdpremat
        uk, sk, vk = svds(a, k=3, return_vecs=True)
        assert_allclose(sk, [4, 3, 2])
        for i, j in zip((0, 1, 2), (2, 3, 1)):
            o = abs(uk[:, i].H @ u[:, j])
            assert_allclose(o, 1.)
            o = abs(vk[i, :] @ v[:, j])
            assert_allclose(o, 1.)

    def test_svds_smalldense_nvecs(self, svdpremat):
        _, _, a = svdpremat
        sk = svds(a, k=3, return_vecs=False)
        assert_allclose(sk, [4, 3, 2])

    def test_svds_sparse_wvecs(self, svdprematsparse):
        u, v, a = svdprematsparse
        uk, sk, vk = svds(a, k=3, return_vecs=True)
        assert_allclose(sk, [4, 3, 2])
        for i, j in zip((0, 1, 2), (2, 3, 1)):
            o = abs(uk[:, i].H @ u[:, j])
            assert_allclose(o, 1.)
            o = abs(vk[i, :] @ v[:, j])
            assert_allclose(o, 1.)

    def test_svds_sparse_nvecs(self, svdprematsparse):
        _, _, a = svdprematsparse
        sk = svds(a, k=3, return_vecs=False)
        assert_allclose(sk, [4, 3, 2])


class TestNorms:
    def test_norm_fro_dense(self):
        a = qu([[1, 2], [3j, 4j]])
        assert norm(a, "fro") == (1 + 4 + 9 + 16)**0.5

    def test_norm_fro_sparse(self):
        a = qu([[3, 0], [4j, 0]], sparse=True)
        assert norm(a, "fro") == (9 + 16)**0.5

    def test_norm_spectral_dense(self, svdpremat):
        _, _, a = svdpremat
        assert_allclose(norm(a, "spectral"), 4.)

    def test_norm_spectral_sparse(self, svdprematsparse):
        _, _, a = svdprematsparse
        assert_allclose(norm(a, "spectral"), 4.)

    def test_norm_trace_dense(self):
        a = np.asmatrix(np.diag([-3, 1, 7]))
        assert norm(a, "trace") == 11
        a = rand_product_state(1, qtype="dop")
        assert_allclose(norm(a, "nuc"), 1)
