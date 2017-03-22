import pytest

from quimb import (
    rand_matrix,

)

from quimb.solve import SLEPC4PY_FOUND

if SLEPC4PY_FOUND:
    from quimb.solve.mpi_spawner import slepc_mpi_seigsys


slepc4py_notfound_msg = "No SLEPc4py installation"
slepc4py_test = pytest.mark.skipif(not SLEPC4PY_FOUND,
                                   reason=slepc4py_notfound_msg)


@pytest.fixture
def bigsparsemat():
    return rand_matrix(100, sparse=True, density=0.1)


@slepc4py_test
class TestSLEPcMPI:
    @pytest.mark.parametrize("num_workers", [None, 1, 2, 3])
    def test_1(self, num_workers, bigsparsemat):
        slepc_mpi_seigsys(bigsparsemat, num_workers=num_workers)
