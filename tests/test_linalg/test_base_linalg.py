import pytest
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose

import quimb as qu
from quimb.linalg import SLEPC4PY_FOUND
from quimb.linalg.base_linalg import _rel_window_to_abs_window

eigs_backends = ["auto", "numpy", "scipy"]
svds_backends = ["numpy", "scipy"]

if SLEPC4PY_FOUND:
    eigs_backends += ["slepc-nompi", "slepc"]
    svds_backends += ["slepc-nompi", "slepc"]


# --------------------------------------------------------------------------- #
#                              Fixtures                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mat_herm_dense():
    np.random.seed(1)
    u = qu.rand_uni(4)
    a = u @ qu.ldmul(np.array([-1, 2, 4, -3]), u.H)
    #  |--|--|--|--|--|--|--|
    # -3    -1        2     4
    return u, a


@pytest.fixture
def mat_herm_sparse():
    np.random.seed(1)
    u = qu.rand_uni(4)
    a = u @ qu.ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qu.sparse(a)
    return u, a


@pytest.fixture
def mat_nherm_dense():
    np.random.seed(1)
    u, v = qu.rand_uni(5), qu.rand_uni(5)
    a = u @ qu.ldmul(np.array([1, 2, 4, 3, 0.1]), v.H)
    return u, v, a


@pytest.fixture
def mat_nherm_sparse():
    np.random.seed(1)
    u, v = qu.rand_uni(5), qu.rand_uni(5)
    a = u @ qu.ldmul(np.array([1, 2, 4, 3, 0.1]), v.H)
    a = qu.sparse(a)
    return u, v, a


@pytest.fixture
def ham1():
    u = qu.rand_uni(7)
    el = np.array([-3, 0, 1, 2, 3, 4, 7])
    return u @ qu.ldmul(el, u.H)


@pytest.fixture
def ham2():
    u = qu.rand_uni(7)
    el = np.array([-3.72, 0, 1, 1.1, 2.1, 2.2, 6.28])
    return u @ qu.ldmul(el, u.H)


# --------------------------------------------------------------------------- #
#                              Tests                                          #
# --------------------------------------------------------------------------- #

class TestEigh:
    def test_eigsys(self, mat_herm_dense):
        u, a = mat_herm_dense
        evals, v = qu.eigh(a)
        assert(set(np.rint(evals)) == set((-1, 2, 4, -3)))
        assert_allclose(evals, [-3, -1, 2, 4])
        for i, j in zip([3, 0, 1, 2], range(4)):
            o = u[:, [i]].H @ v[:, [j]]
            assert_allclose(abs(o), 1.)

    def test_eigvals(self, mat_herm_dense):
        _, a = mat_herm_dense
        evals = qu.eigvalsh(a)
        assert_allclose(evals, [-3, -1, 2, 4])

    def test_eigvecs(self, mat_herm_dense):
        u, a = mat_herm_dense
        v = qu.eigvecsh(a)
        for i, j in zip([3, 0, 1, 2], range(4)):
            o = u[:, [i]].H @ v[:, [j]]
            assert_allclose(abs(o), 1.)


class TestSeigs:
    @pytest.mark.parametrize("backend", eigs_backends)
    def test_eigs_small_dense_wvecs(self, mat_herm_dense, backend):
        u, a = mat_herm_dense
        assert not qu.issparse(a)
        lk, vk = qu.eigh(a, k=2, backend=backend)
        assert_allclose(lk, (-3, -1))
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, [i]].H @ vk[:, [j]]
            assert_allclose(abs(o), 1.)
        vk = qu.eigvecsh(a, k=2, backend=backend)
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, [i]].H @ vk[:, [j]]
            assert_allclose(abs(o), 1.)

    @pytest.mark.parametrize("backend", eigs_backends)
    def test_eigs_small_dense_novecs(self, mat_herm_dense, backend):
        _, a = mat_herm_dense
        assert not qu.issparse(a)
        lk = qu.eigvalsh(a, k=2, backend=backend)
        assert_allclose(lk, (-3, -1))

    @pytest.mark.parametrize("backend", eigs_backends)
    def test_eigs_sparse_wvecs(self, mat_herm_sparse, backend):
        u, a = mat_herm_sparse
        assert qu.issparse(a)
        lk, vk = qu.eigh(a, k=2, backend=backend)
        assert_allclose(lk, (-3, -1))
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, [i]].H @ vk[:, [j]]
            assert_allclose(abs(o), 1.)
        vk = qu.eigvecsh(a, k=2, backend=backend)
        for i, j in zip([3, 0], [0, 1]):
            o = u[:, [i]].H @ vk[:, [j]]
            assert_allclose(abs(o), 1.)

    @pytest.mark.parametrize("backend", eigs_backends)
    def test_eigs_small_sparse_novecs(self, mat_herm_sparse, backend):
        _, a = mat_herm_sparse
        assert qu.issparse(a)
        lk = qu.eigvalsh(a, k=2, backend=backend)
        assert_allclose(lk, (-3, -1))

    @pytest.mark.parametrize("backend", eigs_backends)
    def test_groundstate(self, mat_herm_dense, backend):
        u, a = mat_herm_dense
        gs = qu.groundstate(a, backend=backend)
        assert_allclose(abs(u[:, [3]].H @ gs), 1.)

    @pytest.mark.parametrize("backend", eigs_backends)
    def test_groundenergy(self, mat_herm_dense, backend):
        _, a = mat_herm_dense
        ge = qu.groundenergy(a, backend=backend)
        assert_allclose(ge, -3)

    @pytest.mark.parametrize("which", [None, "SA", "LA", "LM", "SM", "TR"])
    @pytest.mark.parametrize("k", [1, 2])
    def test_cross_equality(self, mat_herm_sparse, k, which):
        _, a = mat_herm_sparse
        sigma = 1 if which in {None, "TR"} else None
        lks, vks = zip(*(qu.eigh(a, k=k, which=which, sigma=sigma, backend=b)
                         for b in eigs_backends))
        lks, vks = tuple(lks), tuple(vks)
        for i in range(len(lks) - 1):
            assert_allclose(lks[i], lks[i + 1])
            assert_allclose(abs(vks[i].H @ vks[i + 1]), qu.eye(k), atol=1e-14)


class TestLOBPCG:
    def test_against_arpack(self):
        A = qu.rand_herm(32, dtype=float)
        lk, vk = qu.eigh(A, k=6, backend='lobpcg')
        slk, svk = qu.eigh(A, k=6, backend='scipy')
        assert_allclose(lk, slk)
        assert_allclose(np.eye(6), abs(vk.H @ svk), atol=1e-9, rtol=1e-9)


class TestEvalsWindowed:
    @pytest.mark.parametrize("backend", eigs_backends)
    def test_bound_spectrum(self, ham1, backend):
        h = ham1
        lmin, lmax = qu.bound_spectrum(h, backend=backend)
        assert_allclose((lmin, lmax), (-3, 7), atol=1e-13)

    def test_rel_window_to_abs_window(self):
        el0 = _rel_window_to_abs_window(5, 10, 0.5)
        assert_allclose(el0, 7.5)
        el0, eli, elf = _rel_window_to_abs_window(-20, -10, 0.5, 0.2)
        assert_allclose([el0, eli, elf], [-15, -16, -14])

    def test_dense(self, ham2):
        h = ham2
        el = qu.eigvalsh_window(h, 0.5, 2, w_sz=0.1)
        assert_allclose(el, [1, 1.1])

    def test_dense_cut(self, ham1):
        h = ham1
        el = qu.eigvalsh_window(h, 0.5, 5, w_sz=0.3)
        assert_allclose(el, [1, 2, 3])

    @pytest.mark.parametrize("backend", eigs_backends)
    def test_sparse(self, ham2, backend):
        h = qu.sparse(ham2)
        el = qu.eigvalsh_window(h, 0.5, 2, w_sz=0.1, backend=backend)
        assert_allclose(el, [1, 1.1])

    def test_sparse_cut(self, ham1):
        h = qu.sparse(ham1)
        el = qu.eigvalsh_window(h, 0.5, 5, w_sz=0.3)
        assert_allclose(el, [1, 2, 3])

    def test_dense_return_vecs(self, mat_herm_dense):
        u, a = mat_herm_dense
        ev = qu.eigvecsh_window(a, w_0=0.5, k=2, w_sz=0.8)
        assert ev.shape == (4, 2)
        assert_allclose(abs(u[:, :2].H @ ev[:, ]), [[1, 0], [0, 1]],
                        atol=1e-14)

    def test_sparse_return_vecs(self, mat_herm_sparse):
        u, a = mat_herm_sparse
        ev = qu.eigvecsh_window(a, w_0=0.5, k=2, w_sz=0.8)
        assert ev.shape == (4, 2)
        assert_allclose(abs(u[:, :2].H @ ev[:, ]), [[1, 0], [0, 1]],
                        atol=1e-14)


class TestSVD:
    def test_svd_full(self, mat_nherm_dense):
        u, v, a = mat_nherm_dense
        un, sn, vn = qu.svd(a)
        assert_allclose(sn, [4, 3, 2, 1, 0.1], atol=1e-14)
        for i, j, in zip((0, 1, 2, 3, 4),
                         (2, 3, 1, 0, 4)):
            o = abs(un[:, [i]].H @ u[:, [j]])
            assert_allclose(o, 1.)
            o = abs(vn[[i], :] @ v[:, [j]])
            assert_allclose(o, 1.)


class TestSVDS:
    @pytest.mark.parametrize("backend", svds_backends)
    def test_svds_smalldense_wvecs(self, mat_nherm_dense, backend):
        u, v, a = mat_nherm_dense
        uk, sk, vk = qu.svds(a, k=3, return_vecs=True, backend=backend)
        assert_allclose(sk, [4, 3, 2])
        for i, j in zip((0, 1, 2), (2, 3, 1)):
            o = abs(uk[:, [i]].H @ u[:, [j]])
            assert_allclose(o, 1.)
            o = abs(vk[[i], :] @ v[:, [j]])
            assert_allclose(o, 1.)

    @pytest.mark.parametrize("backend", svds_backends)
    def test_svds_smalldense_nvecs(self, mat_nherm_dense, backend):
        _, _, a = mat_nherm_dense
        sk = qu.svds(a, k=3, return_vecs=False, backend=backend)
        assert_allclose(sk, [4, 3, 2])

    @pytest.mark.parametrize("backend", svds_backends)
    def test_svds_sparse_wvecs(self, mat_nherm_sparse, backend):
        u, v, a = mat_nherm_sparse
        uk, sk, vk = qu.svds(a, k=3, return_vecs=True, backend=backend)
        assert_allclose(sk, [4, 3, 2])
        for i, j in zip((0, 1, 2), (2, 3, 1)):
            o = abs(uk[:, [i]].H @ u[:, [j]])
            assert_allclose(o, 1.)
            o = abs(vk[[i], :] @ v[:, [j]])
            assert_allclose(o, 1.)

    @pytest.mark.parametrize("backend", svds_backends)
    def test_svds_sparse_nvecs(self, mat_nherm_sparse, backend):
        _, _, a = mat_nherm_sparse
        sk = qu.svds(a, k=3, return_vecs=False, backend=backend)
        assert_allclose(sk, [4, 3, 2])


class TestNorms:
    def test_norm_fro_dense(self):
        a = qu.quimbify([[1, 2], [3j, 4j]])
        assert qu.norm(a, "fro") == (1 + 4 + 9 + 16)**0.5

    def test_norm_fro_sparse(self):
        a = qu.sparse([[3, 0], [4j, 0]])
        assert qu.norm(a, "fro") == (9 + 16)**0.5

    @pytest.mark.parametrize("backend", svds_backends)
    def test_norm_spectral_dense(self, mat_nherm_dense, backend):
        _, _, a = mat_nherm_dense
        assert_allclose(qu.norm(a, "spectral", backend=backend), 4.)

    @pytest.mark.parametrize("backend", svds_backends)
    def test_norm_spectral_sparse(self, mat_nherm_sparse, backend):
        _, _, a = mat_nherm_sparse
        assert_allclose(qu.norm(a, "spectral", backend=backend), 4.)

    def test_norm_trace_dense(self):
        a = qu.qarray(np.diag([-3, 1, 7]))
        assert qu.norm(a, "trace") == 11
        a = qu.rand_product_state(1, qtype="dop")
        assert_allclose(qu.norm(a, "nuc"), 1)


class TestExpm:
    @pytest.mark.parametrize("herm", [True, False])
    def test_zeros_dense(self, herm):
        p = qu.expm(np.zeros((2, 2), dtype=complex), herm=herm)
        assert_allclose(p, qu.eye(2))

    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("herm", [True, False])
    def test_eye(self, sparse, herm):
        p = qu.expm(qu.eye(2, sparse=sparse), herm=herm)
        assert_allclose((p.A if sparse else p) / np.e, qu.eye(2))
        if sparse:
            assert isinstance(p, sp.csr_matrix)


class TestSqrtm:
    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("herm", [True, False])
    def test_eye(self, herm, sparse):
        if sparse:
            with pytest.raises(NotImplementedError):
                p = qu.sqrtm(qu.eye(2, sparse=sparse), herm=herm)
        else:
            p = qu.sqrtm(qu.eye(2), herm=herm)
            assert_allclose(p, qu.eye(2))


class TestLazy:

    @pytest.mark.parametrize("sparse", [False, True])
    def test_basic(self, sparse):
        ownership = (0, 7)
        hl = qu.Lazy(qu.ham_heis, n=4, sparse=sparse, shape=(16, 16))
        print(hl)
        h = 1 * hl(ownership=ownership)
        h_ex = qu.ham_heis(n=4, sparse=sparse)[slice(*ownership), :]
        assert_allclose(h.A, h_ex.A)

    @pytest.mark.parametrize("backend", ['scipy', 'lobpcg'])
    def test_project_eig(self, backend):
        Hl = qu.Lazy(qu.ham_heis, 4, sparse=True, shape=(16, 16), cyclic=True)
        Pl = qu.Lazy(qu.zspin_projector, 4, shape=(16, 6))

        ge, gs = qu.eigh(Hl, P=Pl, k=1, backend=backend)

        assert ge == pytest.approx(-2)
        assert qu.expec(gs, gs) == pytest.approx(1.0)
