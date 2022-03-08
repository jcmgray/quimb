import pytest

import quimb as qu
import quimb.tensor as qtn


class Test3DManualContract:

    @pytest.mark.parametrize('canonize', [False, True])
    def test_ising_model(self, canonize):
        L = 5
        beta = 0.3
        fex = -2.7654417752878
        tn = qtn.TN3D_classical_ising_partition_function(L, L, L, beta=beta)
        Z = tn.contract_boundary(max_bond=8, canonize=canonize)
        f = -qu.log(Z) / (L**3 * beta)
        assert f == pytest.approx(fex, rel=1e-3)
