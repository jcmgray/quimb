import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    rand_herm,
    rand_ket,
    eigsys,
)

from quimb.linalg import SLEPC4PY_FOUND
from quimb.linalg.scipy_linalg import seigsys_scipy

if SLEPC4PY_FOUND:
    from quimb.linalg.mpi_launcher import (
        seigsys_slepc_spawn,
        svds_slepc_spawn,
        mfn_multiply_slepc_spawn,
        ALREADY_RUNNING_AS_MPI,
        NUM_MPI_WORKERS,
    )


slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = pytest.mark.skipif(not SLEPC4PY_FOUND,
                                   reason=slepc4py_notfound_msg)


num_workers_to_try = [
    None,
    1,
    2,
    3,
]


@pytest.fixture
def bigsparsemat():
    import numpy as np
    np.random.seed(42)
    return rand_herm(100, sparse=True, density=0.1)


@pytest.fixture
def big_vec():
    import numpy as np
    np.random.seed(2442)
    return rand_ket(100)


@slepc4py_test
class TestSLEPcMPI:
    @pytest.mark.parametrize("num_workers", num_workers_to_try)
    def test_seigsys(self, num_workers, bigsparsemat):

        if ((num_workers is not None) and
                ALREADY_RUNNING_AS_MPI and
                num_workers > 1 and
                num_workers != NUM_MPI_WORKERS):
            with pytest.raises(ValueError):
                seigsys_slepc_spawn(bigsparsemat, k=6, num_workers=num_workers)

        else:
            el, ev = seigsys_slepc_spawn(bigsparsemat, k=6,
                                         num_workers=num_workers)
            elex, evex = seigsys_scipy(bigsparsemat, k=6)
            assert_allclose(el, elex)
            assert_allclose(np.abs(ev.H @ evex), np.eye(6), atol=1e-7)

    @pytest.mark.parametrize("num_workers", num_workers_to_try)
    def test_expm_multiply(self, num_workers, bigsparsemat, big_vec):
        a = bigsparsemat
        k = big_vec

        if ((num_workers is not None) and
                ALREADY_RUNNING_AS_MPI and
                num_workers > 1 and
                num_workers != NUM_MPI_WORKERS):
            with pytest.raises(ValueError):
                mfn_multiply_slepc_spawn(a, k, num_workers=num_workers)

        else:
            out = mfn_multiply_slepc_spawn(a, k, num_workers=num_workers)
            al, av = eigsys(a.A)
            expected = av @ np.diag(np.exp(al)) @ av.conj().T @ k
            assert_allclose(out, expected)

    @pytest.mark.parametrize("num_workers", num_workers_to_try)
    def test_svds(self, num_workers):
        a = np.random.randn(13, 7) + 1.0j * np.random.randn(13, 7)

        if ((num_workers is not None) and
                ALREADY_RUNNING_AS_MPI and
                num_workers > 1 and
                num_workers != NUM_MPI_WORKERS):
            with pytest.raises(ValueError):
                svds_slepc_spawn(a, return_vecs=True, num_workers=num_workers)

        else:
            u, s, v = svds_slepc_spawn(a, return_vecs=True,
                                       num_workers=num_workers)


@slepc4py_test
class TestMPIPool:
    def test_spawning_pool_in_pool(self, bigsparsemat):
        from quimb.linalg.mpi_launcher import get_mpi_pool
        l1 = seigsys_slepc_spawn(bigsparsemat, k=6, return_vecs=False)
        pool = get_mpi_pool()
        f = pool.submit(seigsys_slepc_spawn, bigsparsemat,
                        k=6, return_vecs=False, num_workers=1)
        l2 = f.result()
        assert_allclose(l1, l2)
