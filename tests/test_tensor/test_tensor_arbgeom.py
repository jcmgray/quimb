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


def test_gate_with_op():
    A = qtn.MPO_rand(5, 3, dtype=complex)
    x = qtn.MPS_rand_state(5, 3, dtype=complex)
    y = A.to_dense() @ x.to_dense()
    x.gate_with_op_lazy_(A)
    assert_allclose(x.to_dense(), y)


def test_gate_sandwich_with_op():
    B = qtn.MPO_rand(5, 3, dtype=complex)
    A = qtn.MPO_rand(5, 3, dtype=complex)
    y = A.to_dense() @ B.to_dense() @ A.to_dense().conj().T
    B.gate_sandwich_with_op_lazy_(A)
    assert_allclose(B.to_dense(), y)


def test_normalize_simple():
    psi = qtn.PEPS.rand(3, 3, 2, dtype=complex)
    gauges = {}
    psi.gauge_all_simple_(100, 5e-6, gauges=gauges)
    psi.normalize_simple(gauges)

    for where in [
        [(0, 0)],
        [(1, 1), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
    ]:
        tags = [psi.site_tag(w) for w in where]
        k = psi.select_any(tags, virtual=False)
        k.gauge_simple_insert(gauges)

        assert k.H @ k == pytest.approx(1.0)
