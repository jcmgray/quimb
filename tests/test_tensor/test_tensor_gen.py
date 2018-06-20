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

    @pytest.mark.parametrize("var_two", ['none', 'some', 'only'])
    @pytest.mark.parametrize("var_one", ['some', 'only', 'onnly-some',
                                         'def-only', 'none'])
    def test_specials(self, var_one, var_two):
        K1 = qu.rand_herm(2**1)

        n = 10
        HB = qtn.SpinHam(S=1 / 2)

        if var_two == 'some':
            HB += 1, K1, K1
            HB[4, 5] += 1, K1, K1
            HB[7, 8] += 1, K1, K1
        elif var_two == 'only':
            for i in range(n - 1):
                HB[i, i + 1] += 1, K1, K1
        else:
            HB += 1, K1, K1

        if var_one == 'some':
            HB += 1, K1
            HB[2] += 1, K1
            HB[3] += 1, K1
        elif var_one == 'only':
            for i in range(n - 1):
                HB[i] += 1, K1
        elif var_one == 'only-some':
            HB[1] += 1, K1
        elif var_one == 'def-only':
            HB += 1, K1

        HB.build_mpo(n)
        HB.build_nni(n)
