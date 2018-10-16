from pytest import fixture, mark
import numpy as np
from numpy.testing import assert_equal, assert_allclose

import quimb as qu
from quimb.linalg.numpy_linalg import (
    sort_inds,
    eigs_numpy,
)


@fixture
def xs():
    return np.array([-2.4 - 1j, -1 + 2.2j, 1 - 2.1j, 2.3 + 1j])


@fixture
def ham1():
    evecs = qu.rand_uni(5)
    evals = np.array([-5, -3, 0.1, 2, 4])
    return qu.dot(evecs, qu.ldmul(evals, evecs.H))


class TestSortInds:
    @mark.parametrize("method, inds, sigma",
                      [("LM", [0, 3, 1, 2], None),
                       ("SM", [2, 1, 3, 0], None),
                       ("SA", [0, 1, 2, 3], None),
                       ("SR", [0, 1, 2, 3], None),
                       ("SI", [2, 0, 3, 1], None),
                       ("LA", [3, 2, 1, 0], None),
                       ("LR", [3, 2, 1, 0], None),
                       ("LI", [1, 3, 0, 2], None),
                       ("TM", [1, 2, 3, 0], 2.41),
                       ("tm", [1, 2, 3, 0], 2.41),
                       ("TR", [2, 3, 1, 0], 1.01),
                       ("TI", [3, 1, 0, 2], 1.01)])
    def test_simple(self, xs, method, inds, sigma):
        assert_equal(sort_inds(xs, method, sigma), inds)


class TestNumpyEigk:
    @mark.parametrize("which, k, ls, sigma",
                      [("lm", 3, [-5, 4, -3], None),
                       ("sm", 3, [0.1, 2, -3], None),
                       ("tm", 3, [-3, 2, 4], 2.9)])
    def test_evals(self, ham1, which, k, ls, sigma):
        lk = eigs_numpy(ham1, k=k, which=which, return_vecs=False,
                        sigma=sigma, sort=False)
        assert_allclose(lk, ls)

    @mark.parametrize("which, k, sigma",
                      [("sa", 5, None)])
    def test_evecs(self, ham1, which, k, sigma):
        lk, vk = eigs_numpy(ham1, k=k, which=which, return_vecs=True,
                            sigma=sigma, sort=False)
        assert isinstance(vk, qu.qarray)
        assert_allclose(qu.dot(vk, qu.ldmul(lk, vk.H)), ham1)


class TestAutoBlock:

    def test_eigh(self):
        H = qu.ham_mbl(6, dh=2.5)
        a_el, a_ev = qu.eigh(H, autoblock=False)
        el, ev = qu.eigh(H, autoblock=True)

        assert qu.norm(ev @ qu.ldmul(el, ev.H) - H, 'fro') < 1e-12
        assert_allclose(a_el, el)
        assert_allclose(ev.H @ ev, np.eye(H.shape[0]), atol=1e-12)

    def test_eigvals(self):
        H = qu.ham_hubbard_hardcore(4)
        a_el = qu.eigvalsh(H, autoblock=False)
        el = qu.eigvalsh(H, autoblock=True)
        assert_allclose(a_el, el, atol=1e-12)
