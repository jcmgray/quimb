import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


class TestSpinHam:

    @pytest.mark.parametrize("cyclic", [False, True])
    def test_var_terms(self, cyclic):
        n = 8
        Hd = qu.ham_mbl(n, dh=0.77, run=42, cyclic=cyclic)
        Ht = qtn.MPO_ham_mbl(n, dh=0.77, run=42, cyclic=cyclic).to_dense()
        assert_allclose(Hd, Ht)
