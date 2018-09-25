from pytest import fixture, mark
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

from quimb import (qu, rand_uni, ldmul, rand_matrix, rand_herm, rand_pos,
                   rand_ket, eigh, eye, norm)
from quimb.linalg import SLEPC4PY_FOUND
from quimb.linalg.scipy_linalg import svds_scipy
if SLEPC4PY_FOUND:
    from quimb.linalg.slepc_linalg import (
        eigs_slepc, svds_slepc, convert_mat_to_petsc, new_petsc_vec,
        mfn_multiply_slepc, ssolve_slepc,
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
class TestSlepceigs:
    def test_internal_eigvals(self, prematsparse):
        u, a = prematsparse
        lk = eigs_slepc(a, k=2, sigma=0.5, return_vecs=False)
        assert_allclose(lk, [-1, 2])

    @mark.parametrize("which, output", [
        ('lm', 4),
        ("sa", -3),
    ])
    def test_eigs_slepc_groundenergy(self, prematsparse, which, output):
        u, a = prematsparse
        lk = eigs_slepc(a, k=1, which=which, return_vecs=False)
        assert_allclose(lk, output)

    @mark.parametrize("dtype", ['real', 'complex'])
    def test_eigs_slepc_eigvecs(self, dtype):
        h = rand_herm(32, sparse=True, density=0.5)
        if dtype == 'real':
            h = h.real
        lks, vks = eigs_slepc(h, k=5)
        lka, vka = eigh(h, k=5, backend='scipy')
        assert vks.shape == vka.shape
        assert h.dtype == vks.dtype

        assert_allclose(lks, lka)
        assert_allclose(abs(vka.H @ vks), np.eye(5), atol=1e-8)


@slepc4py_test
class TestSlepcSvds:
    def test_simple(self, prematsparse):
        u, a = prematsparse
        lk = svds_slepc(a, k=1, return_vecs=False)
        assert_allclose(lk, 4)

    @mark.parametrize("SVDType", ['cross', 'lanczos'])
    def test_random_compare_scipy(self, bigsparsemat, SVDType):
        a = bigsparsemat
        lk = svds_slepc(a, k=5, return_vecs=False, SVDType=SVDType)
        ls = svds_scipy(a, k=5, return_vecs=False)
        assert_allclose(lk, ls)

    @mark.parametrize("SVDType", ['cross', 'lanczos'])
    def test_unitary_vectors(self, bigsparsemat, SVDType):
        a = bigsparsemat
        uk, sk, vk = svds_slepc(a, k=10, return_vecs=True, SVDType=SVDType)
        assert_allclose(uk.H @ uk, eye(10), atol=1e-6)
        assert_allclose(vk @ vk.H, eye(10), atol=1e-6)
        pk, lk, qk = svds_scipy(a, k=10, return_vecs=True)
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

        out = mfn_multiply_slepc(a, k)

        al, av = eigh(a.A)
        expected = av @ np.diag(np.exp(al)) @ av.conj().T @ k

        assert_allclose(out, expected)

    def test_sqrt_sparse(self):
        import scipy.sparse as sp

        a = rand_pos(32, sparse=True, density=0.1)
        a = a + 0.001 * sp.eye(32)
        k = rand_ket(32)

        out = mfn_multiply_slepc(a, k, fntype='sqrt', isherm=True)

        al, av = eigh(a.A)
        al[al < 0] = 0.0  # very small neg values spoil sqrt
        expected = av @ np.diag(np.sqrt(al)) @ av.conj().T @ k

        assert_allclose(out, expected, rtol=1e-6)


@slepc4py_test
class TestShellMatrix:

    def test_extermal_eigen(self):
        a = rand_herm(100, sparse=True)
        alo = sp.linalg.aslinearoperator(a)

        el_us, ev_us = sp.linalg.eigsh(alo, k=1, which='LA')
        el_u, ev_u = eigs_slepc(alo, k=1, which='LA')

        assert_allclose(el_us, el_u)
        assert_allclose(np.abs(ev_us.conj().T @ ev_u), 1.0)

        el_ls, ev_ls = sp.linalg.eigsh(alo, k=1, which='SA')
        el_l, ev_l = eigs_slepc(alo, k=1, which='SA')

        assert_allclose(el_ls, el_l)
        assert_allclose(np.abs(ev_ls.conj().T @ ev_l), 1.0)

    def test_internal_interior_default(self):
        a = rand_herm(100, sparse=True)
        alo = sp.linalg.aslinearoperator(a)
        el, ev = eigs_slepc(alo, k=1, which='TR', sigma=0.0)
        el_s, ev_s = sp.linalg.eigsh(a.tocsc(), k=1, which='LM', sigma=0.0)

        assert_allclose(el_s, el, rtol=1e-5)
        assert_allclose(np.abs(ev_s.conj().T @ ev), 1.0)

    def test_internal_shift_invert_linear_solver(self):
        a = rand_herm(100, sparse=True, seed=42)
        alo = sp.linalg.aslinearoperator(a)

        st_opts = {
            'STType': 'sinvert',
            'KSPType': 'bcgs',  # / 'gmres'
            'PCType': 'none',
        }

        el, ev = eigs_slepc(alo, k=1, which='TR', sigma=0.0, tol=1e-6,
                            st_opts=st_opts, EPSType='krylovschur')
        el_s, ev_s = sp.linalg.eigsh(a.tocsc(), k=1, which='LM', sigma=0.0)

        assert_allclose(el_s, el, rtol=1e-5)
        assert_allclose(np.abs(ev_s.conj().T @ ev), 1.0)

    def test_internal_shift_invert_precond(self):
        a = rand_herm(20, sparse=True, seed=42)
        alo = sp.linalg.aslinearoperator(a)

        st_opts = {
            'STType': 'precond',
            'KSPType': 'preonly',
            'PCType': 'none',
        }

        el, ev = eigs_slepc(alo, k=1, which='TR', sigma=0.0,
                            st_opts=st_opts, EPSType='gd')
        el_s, ev_s = sp.linalg.eigsh(a.tocsc(), k=1, which='LM', sigma=0.0)

        assert_allclose(el_s, el, rtol=1e-6)
        assert_allclose(np.abs(ev_s.conj().T @ ev), 1.0)

    def test_internal_shift_invert_precond_jd(self):
        a = rand_herm(20, sparse=True, seed=42)
        alo = sp.linalg.aslinearoperator(a)

        st_opts = {
            'STType': 'precond',
            'KSPType': 'bcgs',  # / 'gmres'
            'PCType': 'none',
        }

        el, ev = eigs_slepc(alo, k=1, which='TR', sigma=0.0,
                            st_opts=st_opts, EPSType='jd')
        el_s, ev_s = sp.linalg.eigsh(a.tocsc(), k=1, which='LM', sigma=0.0)

        assert_allclose(el_s, el, rtol=1e-6)
        assert_allclose(np.abs(ev_s.conj().T @ ev), 1.0)


@slepc4py_test
class TestCISS:

    def test_1(self):
        a = rand_herm(100, sparse=True)
        el, ev = eigh(a.A)
        which = abs(el) < 0.2
        el, ev = el[which], ev[:, which]

        offset = norm(a, 'fro')
        a = a + offset * sp.eye(a.shape[0])

        sl, sv = eigs_slepc(a, k=6, l_win=(-0.2 + offset, 0.2 + offset))
        sl -= offset
        assert_allclose(el, sl)
        assert_allclose(np.abs(sv.H @ ev), np.eye(el.size), atol=1e-11)


@slepc4py_test
class TestSSolve:

    def test_simple_dense(self):
        a = rand_herm(2**4, sparse=True)
        y = rand_ket(2**4)
        x = ssolve_slepc(a, y)
        assert_allclose(a @ x, y, atol=1e-12, rtol=1e-6)
