import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    rand_herm,
    rand_ket,
    eigsys,
)

from quimb.solve import SLEPC4PY_FOUND
from quimb.solve.scipy_solver import seigsys_scipy

if SLEPC4PY_FOUND:
    from quimb.solve.mpi_spawner import (
        seigsys_slepc_spawn,
        svds_slepc_spawn,
        mfn_multiply_slepc_spawn,
    )


slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = pytest.mark.skipif(not SLEPC4PY_FOUND,
                                   reason=slepc4py_notfound_msg)


@pytest.fixture
def bigsparsemat():
    return rand_herm(100, sparse=True, density=0.1)


@slepc4py_test
class TestSLEPcMPI:
    @pytest.mark.parametrize("num_workers", [None, 1, 2, 3])
    def test_seigsys(self, num_workers, bigsparsemat):
        el, ev = seigsys_slepc_spawn(bigsparsemat, num_workers=num_workers)
        elex, evex = seigsys_scipy(bigsparsemat)
        assert_allclose(el, elex)
        assert_allclose(np.abs(ev.H @ evex), np.eye(6), atol=1e-7)

    @pytest.mark.parametrize("num_workers", [None, 1, 2, 3])
    def test_expm_multiply(self, num_workers, bigsparsemat):
        a = bigsparsemat
        k = rand_ket(100)
        out = mfn_multiply_slepc_spawn(a, k)
        al, av = eigsys(a.A)
        expected = av @ np.diag(np.exp(al)) @ av.conj().T @ k
        assert_allclose(out, expected)

    @pytest.mark.parametrize("num_workers", [None, 1, 2, 3])
    def test_svds(self, num_workers):
        a = np.random.randn(13, 7) + 1.0j * np.random.randn(13, 7)
        u, s, v = svds_slepc_spawn(a, return_vecs=True)


@slepc4py_test
class TestMPIPool:
    def test_spawning_pool_in_pool(self, bigsparsemat):
        from quimb.solve.mpi_spawner import get_mpi_pool
        l1 = seigsys_slepc_spawn(bigsparsemat, return_vecs=False)
        pool = get_mpi_pool()
        f = pool.submit(seigsys_slepc_spawn,
                        bigsparsemat, return_vecs=False)
        l2 = f.result()
        assert_allclose(l1, l2)
