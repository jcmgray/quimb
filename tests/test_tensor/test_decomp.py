import pytest
import numpy as np
import quimb as qu


def test_sgn_convention():
    from quimb.tensor.decomp import sgn

    assert sgn(1) == 1
    assert sgn(2.0) == 1
    assert sgn(-1) == -1
    assert sgn(-2.0) == -1
    assert sgn(0) == 1
    assert sgn(0.0) == 1
    assert sgn(0.0 + 0.0j) == 1
    assert sgn(1.0 + 2.0j) != 1
    assert sgn(1.0 + 2.0j) != -1
    assert abs(sgn(1.0 + 2.0j)) == pytest.approx(1)


@pytest.mark.parametrize('dtype', [
    'float64', 'float32', 'complex128', 'complex64'
])
def test_qr_stabilized_sign_bug(dtype):
    from quimb.tensor.decomp import qr_stabilized

    for _ in range(10):
        Q = qu.rand_uni(4, dtype=dtype)
        R = qu.rand_matrix(4, dtype=dtype)

        # make R strictly upper triangular
        ii, jj = np.indices(R.shape)
        R[ii >= jj] = 0.0

        X = Q @ R
        Q2, _, R2 = qr_stabilized(X)

        assert (
            abs(np.linalg.norm((Q2 @ R2) - X))
            < (1e-12 if dtype in ('float64', 'complex128') else 1e-6)
        )

