# TODO: TEST NON_HERMITIAN
from pytest import fixture, mark
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose
from ... import (
    ldmul,
    rand_uni,
    qjf,
    rand_matrix,
    svds,
    slepc4py_found,
    rand_herm,
    seigsys,
    overlap,
    scipy_to_petsc_csr,
    eye,
    )
from ...solve.advanced_solve import aeigsys, asvds


slepc4py_notfound = not slepc4py_found()
slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = mark.skipif(slepc4py_notfound, reason=slepc4py_notfound_msg)


@fixture
def prematsparse():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qjf(a, sparse=True)
    return u, a


@slepc4py_test
class TestScipyToPETScConversion:
    def test_csr(self):
        a = rand_matrix(2, sparse=True, density=0.5)
        b = scipy_to_petsc_csr(a)
        assert b.getType() == 'seqaij'

    def test_bsr(self):
        a = sp.kron(rand_matrix(2), eye(2, sparse=True), format='bsr')
        b = scipy_to_petsc_csr(a)
        assert b.getType() == 'seqbaij'
        assert b.getBlockSize() == 2


@slepc4py_test
class TestAEigsys:
    def test_internal_eigvals(self, prematsparse):
        u, a = prematsparse
        lk = aeigsys(a, k=2, sigma=0.5, return_vecs=False)
        assert_allclose(lk, [-1, 2])

    def test_aeigsys_groundenergy(self, prematsparse):
        u, a = prematsparse
        lk = aeigsys(a, k=1, which="SR", return_vecs=False)
        assert_allclose(lk, -3)
        lk = aeigsys(a, k=1, which="lm", return_vecs=False)
        assert_allclose(lk, 4)

    def test_aeigsys_eigvecs(self):
        h = rand_herm(100, sparse=True, density=0.2)
        lks, vks = aeigsys(h, k=5)
        lka, vka = seigsys(h, k=5)
        assert vks.shape == vka.shape
        for ls, vs, la, va in zip(lks, vks.T, lka, vka.T):
            assert_allclose(ls, la)
            assert_allclose(overlap(vs, va), 1.0)


@slepc4py_test
class TestASvds:
    def test_svds_simple(self, prematsparse):
        u, a = prematsparse
        lk = asvds(a, k=1)
        assert_allclose(lk, 4)

    def test_svds_random_compare_scipy(self, prematsparse):
        a = rand_matrix(100, sparse=True, density=0.1)
        lk = asvds(a, k=5)
        ls = svds(a, k=5, return_vecs=False)
        assert_allclose(lk, ls)
