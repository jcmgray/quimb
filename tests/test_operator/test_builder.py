import pytest
from numpy.testing import assert_allclose

import quimb.experimental.operatorbuilder as qop


def assert_all_matrices_match(sob):
    Ad = sob.build_dense()
    As = sob.build_sparse_matrix()
    assert_allclose(Ad, As.toarray())
    Am = sob.build_mpo().to_dense()
    assert_allclose(Ad, Am)
    Ae = sob.build_matrix_ikron()
    assert_allclose(Ad, Ae)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
@pytest.mark.parametrize("m", [1, 3, 10])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("kmin", [0, None])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rand_operator_matrices(n, m, k, kmin, seed):
    if k > n:
        pytest.skip("k > n, skipping test")
    sob = qop.rand_operator(n, m, k, kmin=kmin, seed=seed, ops="xyz+-n")
    assert sob.nsites == n
    assert sob.nterms <= m  # can be less due to repeated terms
    assert_all_matrices_match(sob)
