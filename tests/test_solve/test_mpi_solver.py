import pytest
from numpy.testing import assert_allclose

from quimb import (
    rand_herm,
)

from quimb.solve import SLEPC4PY_FOUND

if SLEPC4PY_FOUND:
    from quimb.solve.mpi_spawner import slepc_mpi_seigsys


slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = pytest.mark.skipif(not SLEPC4PY_FOUND,
                                   reason=slepc4py_notfound_msg)


@pytest.fixture
def bigsparsemat():
    return rand_herm(100, sparse=True, density=0.1)


@slepc4py_test
class TestSLEPcMPI:
    @pytest.mark.parametrize("num_workers", [None, 1, 2, 3])
    def test_1(self, num_workers, bigsparsemat):
        slepc_mpi_seigsys(bigsparsemat, num_workers=num_workers)


@slepc4py_test
class TestMPIPool:
    def test_spawning_pool_in_pool(self, bigsparsemat):
        from quimb.solve.mpi_spawner import get_mpi_pool
        l1 = slepc_mpi_seigsys(bigsparsemat, return_vecs=False)
        pool = get_mpi_pool()
        f = pool.submit(slepc_mpi_seigsys, bigsparsemat, return_vecs=False)
        l2 = f.result()
        assert_allclose(l1, l2)
