from pytest import fixture, mark
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from ... import (
    dot,
    ldmul,
    rand_uni,
    )

from ...solve.numpy_solver import(
    sort_inds,
    numpy_seigsys,
    )


@fixture
def xs():
    return np.array([-2.4 - 1j, -1 + 2.2j, 1 - 2.1j, 2.3 + 1j])


@fixture
def ham1():
    v = rand_uni(5)
    l = np.array([-5, -3, 0.1, 2, 4])
    return dot(v, ldmul(l, v.H))


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
                       ("TM", [1, 2, 3, 0], 2.4),
                       ("tm", [1, 2, 3, 0], 2.4),
                       ("TR", [2, 3, 1, 0], 1),
                       ("TI", [3, 1, 0, 2], 1)])
    def test_simple(self, xs, method, inds, sigma):
        assert_equal(sort_inds(xs, method, sigma), inds)


class TestNumpySeigsys:
    @mark.parametrize("which, k, ls, sigma",
                      [("lm", 3, [-5, 4, -3], None),
                       ("sm", 3, [0.1, 2, -3], None),
                       ("tm", 3, [-3, 2, 4], 2.9)])
    def test_evals(self, ham1, which, k, ls, sigma):
        lk = numpy_seigsys(ham1, k=k, which=which, return_vecs=False,
                           sigma=sigma, sort=False)
        assert_allclose(lk, ls)

    @mark.parametrize("which, k, sigma",
                      [("sa", 5, None)])
    def test_evecs(self, ham1, which, k, sigma):
        lk, vk = numpy_seigsys(ham1, k=k, which=which, return_vecs=True,
                               sigma=sigma, sort=False)
        assert isinstance(vk, np.matrix)
        assert_allclose(dot(vk, ldmul(lk, vk.H)), ham1)
