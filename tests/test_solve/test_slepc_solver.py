# TODO: TEST NON_HERMITIAN
# TODO: TEST multiprocessing throws no error with petsc

from pytest import fixture, mark
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from quimb import (
    qu,
    rand_uni,
    ldmul,
    rand_matrix,
    rand_herm,
    # rand_pos,
    rand_ket,
    eigsys,
    seigsys,
    overlap,
    eye,
)
from quimb.solve import SLEPC4PY_FOUND
from quimb.solve.scipy_solver import scipy_svds
if SLEPC4PY_FOUND:
    from quimb.solve.slepc_solver import (
        slepc_seigsys,
        slepc_svds,
        convert_mat_to_petsc,
        new_petsc_vec,
        slepc_mfn_multiply,
    )


slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = mark.skipif(not SLEPC4PY_FOUND, reason=slepc4py_notfound_msg)


@fixture
def prematsparse():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qu(a, sparse=True)
    return u, a


@fixture
def bigsparsemat():
    return rand_matrix(100, sparse=True, density=0.1)


# --------------------------------------------------------------------------- #
# TESTS                                                                       #
# --------------------------------------------------------------------------- #

@slepc4py_test
class TestConvertToPETScConversion:
    def test_csr(self):
        a = rand_matrix(2, sparse=True, density=0.5)
        b = convert_mat_to_petsc(a)
        assert b.getType() == 'seqaij'

    def test_bsr(self):
        a = sp.kron(rand_matrix(2), eye(2, sparse=True), format='bsr')
        b = convert_mat_to_petsc(a)
        assert b.getType() == 'seqbaij'
        assert b.getBlockSize() == 2

    # def test_vec(self):
    #     a = np.array([1, 2, 3, 4])
    #     b = convert_mat_to_petsc(a)
    #     assert_allclose(b.getArray(), a)

    def test_dense(self):
        a = rand_matrix(3)
        b = convert_mat_to_petsc(a)
        assert b.getType() == 'seqdense'

    def test_new_petsc_vector(self):
        a = new_petsc_vec(4)
        assert a.getArray() is not None


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
        # TODO ************************************************************** #
        # h = ham_heis(n=10, sparse=True)
        pass


@slepc4py_test
class TestSlepcSvds:
    def test_simple(self, prematsparse):
        u, a = prematsparse
        lk = slepc_svds(a, k=1, return_vecs=False)
        assert_allclose(lk, 4)

    @mark.parametrize("SVDType", ['cross', 'lanczos'])
    def test_random_compare_scipy(self, bigsparsemat, SVDType):
        a = bigsparsemat
        lk = slepc_svds(a, k=5, return_vecs=False, SVDType=SVDType)
        ls = scipy_svds(a, k=5, return_vecs=False)
        assert_allclose(lk, ls)

    @mark.parametrize("SVDType", ['cross', 'lanczos'])
    def test_unitary_vectors(self, bigsparsemat, SVDType):
        a = bigsparsemat
        uk, sk, vk = slepc_svds(a, k=10, return_vecs=True, SVDType=SVDType)
        assert_allclose(uk.H @ uk, eye(10), atol=1e-6)
        assert_allclose(vk @ vk.H, eye(10), atol=1e-6)
        pk, lk, qk = scipy_svds(a, k=10, return_vecs=True)
        assert_allclose(sk, lk)
        assert pk.shape == uk.shape
        assert vk.shape == qk.shape
        assert_allclose(abs(uk.H @ pk), eye(10), atol=1e-6)
        assert_allclose(abs(qk @ vk.H), eye(10), atol=1e-6)


@slepc4py_test
class TestSlepcMfnMultiply:

    def test_exp_sparse(self):

        a = rand_herm(100, sparse=True, density=0.1)
        k = rand_ket(100)

        out = slepc_mfn_multiply(a, k)

        al, av = eigsys(a.A)
        expected = av @ np.diag(np.exp(al)) @ av.conj().T @ k

        assert_allclose(out, expected)

    # def test_sqrt_sparse(self):

    #     a = rand_pos(100, sparse=True, density=0.1)
    #     k = rand_ket(100)

    #     out = slepc_mfn_multiply(a, k, fn='sqrt', isherm=True)

    #     al, av = eigsys(a.A)
    #     al[al < 0] = 0.0  # very small neg values spoil sqrt
    #     expected = av @ np.diag(np.sqrt(al)) @ av.conj().T @ k

    #     assert_allclose(out, expected)
