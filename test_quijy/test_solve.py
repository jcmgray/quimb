from functools import lru_cache
import scipy.sparse as sp
from pytest import fixture
from numpy.testing import assert_allclose
from quijy import ldmul, rand_uni
from quijy.solve import *


@fixture
def premat():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    return u, a


@fixture
def prematsparse():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qjf(a, sparse=True)
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
    a = qjf(a, sparse=True)
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


class TestEigsh:
    def test_seigsys_small_dense_wvecs(self, premat):
        u, a = premat
        assert not sp.issparse(a)
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
        assert not sp.issparse(a)
        lk = seigvals(a, k=2)
        assert_allclose(lk, (-3, -1))

    def test_seigsys_sparse_wvecs(self, prematsparse):
        u, a = prematsparse
        assert sp.issparse(a)
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
        assert sp.issparse(a)
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

    def test_norm2(self, svdpremat):
        _, _, a = svdpremat
        assert_allclose(norm2(a), 4.)


class TestChooseNCV:
    def test_choose_ncv(self):
        assert(choose_ncv(1, 100) == 20)
        assert(choose_ncv(15, 100) == 31)
        assert(choose_ncv(50, 100) == 100)
