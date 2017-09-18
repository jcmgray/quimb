import pytest
from numpy.testing import assert_allclose

from quimb import (
    prod,
    rand_ket,
    partial_transpose,
)

from quimb.linalg.approx_linalg import (
    get_cntrct_inds_ptr_dot,
    lazy_ptr_dot,
    get_cntrct_inds_ptr_ppt_dot,
    lazy_ptr_ppt_dot,
)

SZS = (5, 4, 3)
DIMS = [2**sz for sz in SZS]


@pytest.fixture
def tri_psi():
    return rand_ket(prod(DIMS))


@pytest.fixture
def bi_psi():
    return rand_ket(prod(DIMS[:-1]))


class TestLazyTensorEval:

    def test_get_cntrct_inds_ptr_dot_simple(self):
        dims = (5, 7)
        keep = (1,)

        ndim_ab = len(dims)
        dims_a = [d for i, d in enumerate(dims) if i in keep]

        ci_a_k, ci_ab_b, ci_ab_k = get_cntrct_inds_ptr_dot(
            ndim_ab, keep)
        assert dims_a == [7]
        assert ci_a_k == [1]
        assert ci_ab_b == [0, 1]
        assert ci_ab_k == [0, 2]

    def test_get_cntrct_inds_ptr_dot_many_body(self):
        dims = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        keep = (1, 2, 3, 4, 8, 9)

        ndim_ab = len(dims)
        dims_a = [d for i, d in enumerate(dims) if i in keep]

        ci_a_k, ci_ab_b, ci_ab_k = get_cntrct_inds_ptr_dot(
            ndim_ab, keep)
        assert dims_a == [11, 12, 13, 14, 18, 19]
        assert ci_a_k == [1, 2, 3, 4, 8, 9]
        assert ci_ab_b == list(range(11))
        assert ci_ab_k == [0, 11, 12, 13, 14, 5, 6, 7, 15, 16, 10]

    def test_lazy_ptr_dot_simple(self, tri_psi, bi_psi):
        rho_ab = tri_psi.ptr(DIMS, [0, 1])
        psi_out_expected = rho_ab @ bi_psi
        psi_out_got = lazy_ptr_dot(tri_psi, bi_psi, DIMS, keep=[0, 1])
        assert_allclose(psi_out_expected, psi_out_got)

    def test_get_cntrct_inds_ptr_ppt_dot_simple(self):
        dims = (4, 9, 7)
        keep_a = (0,)
        keep_b = (1,)

        ndim_abc = len(dims)
        dims_ab = [d for i, d in enumerate(dims)
                   if (i in keep_a) or (i in keep_b)]

        inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
            get_cntrct_inds_ptr_ppt_dot(ndim_abc, keep_a, keep_b)

        assert dims_ab == [4, 9]
        assert inds_ab_ket == [0, 4]
        assert inds_abc_bra == [3, 4, 2]
        assert inds_abc_ket == [0, 1, 2]
        assert inds_out == [3, 1]

    def test_get_cntrct_inds_ptr_ppt_dot_many_body(self):
        dims = (10, 20, 30, 40, 50, 60, 70, 80, 90)
        keep_a = (0, 1, 5)
        keep_b = (2, 3, 6)

        ndim_abc = len(dims)
        dims_ab = [d for i, d in enumerate(dims)
                   if (i in keep_a) or (i in keep_b)]

        inds_ab_ket, inds_abc_bra, inds_abc_ket, inds_out = \
            get_cntrct_inds_ptr_ppt_dot(ndim_abc, keep_a, keep_b)

        assert dims_ab == [10, 20, 30, 40, 60, 70]
        assert inds_ab_ket == [0, 1, 11, 12, 5, 14]
        assert inds_abc_bra == [9, 10, 11, 12, 4, 13, 14, 7, 8]
        assert inds_abc_ket == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert inds_out == [9, 10, 2, 3, 13, 6]

    def test_lazy_ptr_ppt_dot(self, tri_psi, bi_psi):
        rho_ab = tri_psi.ptr(DIMS, [0, 1])
        rho_ab_pt = partial_transpose(rho_ab, DIMS[:-1])
        psi_out_expected = rho_ab_pt @ bi_psi
        psi_out_got = lazy_ptr_ppt_dot(tri_psi, bi_psi, DIMS, 0, 1)
        assert_allclose(psi_out_expected, psi_out_got)
