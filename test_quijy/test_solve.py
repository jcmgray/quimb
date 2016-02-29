from functools import lru_cache
import scipy.sparse as sp
from numpy.testing import assert_allclose
from quijy.solve import *
from quijy.core import ldmul
from quijy.rand import rand_uni


@lru_cache(maxsize=2)
def fix(sparse=False):
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    if sparse:
        a = qjf(a, sparse=True)
    return u, a


class TestEigh:
    def test_eigsys(self):
        u, a = fix()
        l, _ = eigsys(a, sort=False)
        assert(set(np.rint(l)) == set((-1, 2, 4, -3)))
        l, v = eigsys(a)
        assert_allclose(l, [-3, -1, 2, 4])
        for i, j in zip([3, 0, 1, 2], range(4)):
            o = u[:, i].H @ v[:, j]
            assert_allclose(abs(o), 1.)

    def test_eigvals(self):
        _, a = fix()
        l = eigvals(a)
        assert_allclose(l, [-3, -1, 2, 4])

    def test_eigvecs(self):
        u, a = fix()
        v = eigvecs(a)
        for i, j in zip([3, 0, 1, 2], range(4)):
            o = u[:, i].H @ v[:, j]
            assert_allclose(abs(o), 1.)


class TestEigsh:
    def test_seigsys_small_dense_wvecs(self):
        u, a = fix()
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

    def test_seigsys_small_dense_novecs(self):
        _, a = fix()
        assert not sp.issparse(a)
        lk = seigvals(a, k=2)
        assert_allclose(lk, (-3, -1))

    def test_seigsys_sparse_wvecs(self):
        u, a = fix(True)
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

    def test_seigsys_small_dense_novecs(self):
        _, a = fix(True)
        assert sp.issparse(a)
        lk = seigvals(a, k=2)
        assert_allclose(lk, (-3, -1))

    def test_groundstate(self):
        u, a = fix()
        gs = groundstate(a)
        assert_allclose(abs(u[:, 3].H @ gs), 1.)

    def test_groundenergy(self):
        _, a = fix()
        ge = groundenergy(a)
        assert_allclose(ge, -3)


class TestSVDS:
    def test_svds_smalldense_wvecs(self):
        # TODO
        pass

    def test_svds_smalldense_nvecs(self):
        # TODO
        pass

    def test_svds_sparse_nvecs(self):
        # TODO
        pass

    def test_svds_sparse_nvecs(self):
        # TODO
        pass


class TestChooseNCV:
    def test_choose_ncv(self):
        assert(choose_ncv(1, 100) == 20)
        assert(choose_ncv(15, 100) == 31)
        assert(choose_ncv(50, 100) == 100)
