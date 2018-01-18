import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import approx_spectral_function, eigvals
from quimb.tensor import MPO_rand_herm
from quimb.tensor.tensor_approx_spectral import construct_lanczos_tridiag_MPO


class TestTensorSpectralApprox:

    def test_simple(self):
        A = MPO_rand_herm(10, 7)
        for x in construct_lanczos_tridiag_MPO(A, 5):
            pass

    @pytest.mark.parametrize("fn", [abs, np.cos])
    def test_approx_fn(self, fn):
        A = MPO_rand_herm(10, 7, normalize=True)
        xe = sum(fn(eigvals(A.to_dense())))
        xf = approx_spectral_function(A, fn, R=10)
        assert_allclose(xe, xf, rtol=0.2)
