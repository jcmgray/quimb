import pytest
from numpy.testing import assert_allclose

import quimb.tensor as qtn


@pytest.mark.parametrize("which_A", ["upper", "lower"])
@pytest.mark.parametrize("contract", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_tensor_network_apply_op_vec(which_A, contract, inplace):
    A = qtn.TN_from_edges_rand(
        qtn.edges_2d_square(3, 2),
        D=2,
        phys_dim=2,
        site_ind_id=("k{}", "b{}"),
        dtype=complex,
    )
    x = qtn.TN_from_edges_rand(
        qtn.edges_2d_square(3, 2),
        D=3,
        phys_dim=2,
        site_ind_id="x{}",
        dtype=complex,
    )

    Ad = A.to_dense()
    if which_A == "upper":
        Ad = Ad.T
    xd = x.to_dense()
    C = Ad @ xd

    Ax = qtn.tensor_network_apply_op_vec(
        A,
        x,
        which_A,
        inplace=inplace,
        contract=contract,
    )

    if contract:
        # checks fusing
        assert Ax.num_indices == x.num_indices

    if inplace:
        assert Ax is x
    else:
        assert isinstance(Ax, x.__class__)
        assert Ax.site_ind_id == x.site_ind_id

    assert_allclose(Ax.to_dense(), C)


@pytest.mark.parametrize("which_A", ["upper", "lower"])
@pytest.mark.parametrize("which_B", ["upper", "lower"])
@pytest.mark.parametrize("contract", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_tensor_network_apply_op_op(which_A, which_B, contract, inplace):
    A = qtn.TN_from_edges_rand(
        qtn.edges_2d_square(3, 2),
        D=2,
        phys_dim=2,
        site_ind_id=("k{}", "b{}"),
        dtype=complex,
    )
    B = qtn.TN_from_edges_rand(
        qtn.edges_2d_square(3, 2),
        D=3,
        phys_dim=2,
        site_ind_id=("x{}", "y{}"),
        dtype=complex,
    )
    Ad = A.to_dense()
    if which_A == "upper":
        Ad = Ad.T
    Bd = B.to_dense()
    if which_B == "lower":
        Bd = Bd.T
    C = Ad @ Bd
    if which_B == "lower":
        C = C.T

    AB = qtn.tensor_network_apply_op_op(
        A,
        B,
        which_A,
        which_B,
        inplace=inplace,
        contract=contract,
    )

    if contract:
        # checks fusing
        assert AB.num_indices == B.num_indices

    if inplace:
        assert AB is B
    else:
        assert isinstance(AB, B.__class__)
        assert AB.upper_ind_id == B.upper_ind_id
        assert AB.lower_ind_id == B.lower_ind_id

    assert_allclose(AB.to_dense(), C)
