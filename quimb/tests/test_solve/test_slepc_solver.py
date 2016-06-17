# TODO: TEST NON_HERMITIAN

from pytest import fixture, mark
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose
from ... import (
    ldmul,
    rand_uni,
    qu,
    rand_matrix,
    scipy_svds,
    slepc4py_found,
    rand_herm,
    seigsys,
    overlap,
    eye,
    ham_heis,
    )

from ...solve.slepc_solver import (
    slepc_seigsys,
    slepc_svds,
    convert_to_petsc,
    )


slepc4py_notfound = not slepc4py_found()
slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = mark.skipif(slepc4py_notfound, reason=slepc4py_notfound_msg)


@fixture
def prematsparse():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qu(a, sparse=True)
    return u, a


@fixture
def bigsparsemat():
    return rand_matrix(100, sparse=True, density=0.1)


@slepc4py_test
class TestConvertToPETScConversion:
    def test_csr(self):
        a = rand_matrix(2, sparse=True, density=0.5)
        b = convert_to_petsc(a)
        assert b.getType() == 'seqaij'

    def test_bsr(self):
        a = sp.kron(rand_matrix(2), eye(2, sparse=True), format='bsr')
        b = convert_to_petsc(a)
        assert b.getType() == 'seqbaij'
        assert b.getBlockSize() == 2

    def test_vec(self):
        a = np.array([1, 2, 3, 4])
        b = convert_to_petsc(a)
        assert_allclose(b.getArray(), a)

    def test_dense(self):
        a = rand_matrix(3)
        b = convert_to_petsc(a)
        assert b.getType() == 'seqdense'


@slepc4py_test
class TestSlepcSeigsys:
    def test_internal_eigvals(self, prematsparse):
        u, a = prematsparse
        lk = slepc_seigsys(a, k=2, sigma=0.5, return_vecs=False)
        assert_allclose(lk, [-1, 2])

    @mark.parametrize("which, output", [
        ('lm', 4),
        ("sa", -3),
    ])
    def test_slepc_seigsys_groundenergy(self, prematsparse, which, output):
        u, a = prematsparse
        lk = slepc_seigsys(a, k=1, which=which, return_vecs=False)
        assert_allclose(lk, output)

    def test_slepc_seigsys_eigvecs(self):
        h = rand_herm(100, sparse=True, density=0.2)
        lks, vks = slepc_seigsys(h, k=5)
        lka, vka = seigsys(h, k=5)
        assert vks.shape == vka.shape
        for ls, vs, la, va in zip(lks, vks.T, lka, vka.T):
            assert_allclose(ls, la)
            assert_allclose(overlap(vs, va), 1.0)

    def test_aeigvals_all_consecutive(self):
        h = ham_heis(n=10, sparse=True)


@slepc4py_test
class TestSlepcSvds:
    def test_simple(self, prematsparse):
        u, a = prematsparse
        lk = slepc_svds(a, k=1, return_vecs=False)
        assert_allclose(lk, 4)

    def test_random_compare_scipy(self, bigsparsemat):
        a = bigsparsemat
        lk = slepc_svds(a, k=5, return_vecs=False)
        ls = scipy_svds(a, k=5, return_vecs=False)
        assert_allclose(lk, ls)

    def test_unitary_vectors(self, bigsparsemat):
        a = bigsparsemat
        uk, sk, vk = slepc_svds(a, k=10, return_vecs=True)
        assert_allclose(uk.H @ uk, eye(10), atol=1e-7)
        assert_allclose(vk @ vk.H, eye(10), atol=1e-7)
        pk, lk, qk = scipy_svds(a, k=10, return_vecs=True)
        assert_allclose(sk, lk)
        assert pk.shape == uk.shape
        assert vk.shape == qk.shape
        assert_allclose(abs(uk.H @ pk), eye(10), atol=1e-7)
        assert_allclose(abs(qk @ vk.H), eye(10), atol=1e-7)
