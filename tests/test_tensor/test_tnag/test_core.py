import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


@pytest.mark.parametrize(
    "contract",
    [
        True,
        False,
        "split",
        "reduce-split",
        "split-gate",
        "swap-split-gate",
        "auto-split-gate",
    ],
)
@pytest.mark.parametrize("where", [[1], [2, 3]])
def test_gate_sandwich_basic_mpo(contract, where):
    # apply arbitrary complex 2-site gate to random complex MPO
    mpo = qtn.MPO_rand(5, 3, dtype=complex, seed=42)
    G = qu.rand_matrix(2 ** len(where), seed=42)
    # construct reference by densifying
    A = mpo.to_dense()
    IGI = qu.ikron(G, [mpo.phys_dim()] * mpo.nsites, where)
    GAG = IGI @ A @ IGI.H
    # apply gate via tensor network method
    gmpo = mpo.gate_sandwich(
        G,
        where=where,
        contract=contract,
        tags="GATE",
        tags_upper="KET",
        tags_lower="BRA",
    )
    assert "GATE" in gmpo.tag_map
    d_gmpo = gmpo.to_dense()
    assert_allclose(d_gmpo, GAG)


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


def test_gate_with_op_lazy():
    A = qtn.MPO_rand(5, 3, dtype=complex)
    x = qtn.MPS_rand_state(5, 3, dtype=complex)
    y = A.to_dense() @ x.to_dense()
    x.gate_with_op_lazy_(A)
    assert_allclose(x.to_dense(), y)


def test_gate_sandwich_with_op_lazy():
    B = qtn.MPO_rand(5, 3, dtype=complex)
    A = qtn.MPO_rand(5, 3, dtype=complex)
    y = A.to_dense() @ B.to_dense() @ A.to_dense().conj().T
    B.gate_sandwich_with_op_lazy_(A)
    assert_allclose(B.to_dense(), y)


def test_normalize_simple():
    psi = qtn.PEPS.rand(3, 3, 2, dtype=complex)
    gauges = {}
    info = {}
    psi.gauge_all_simple_(100, 5e-6, gauges=gauges, info=info)
    psi.normalize_simple(gauges)

    assert info["iterations"] <= 100
    if info["iterations"] < 100:
        assert info["max_sdiff"] < 5e-6

    for where in [
        [(0, 0)],
        [(1, 1), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
    ]:
        tags = [psi.site_tag(w) for w in where]
        k = psi.select_any(tags, virtual=False)
        k.gauge_simple_insert(gauges)

        assert k.H @ k == pytest.approx(1.0)


@pytest.mark.parametrize("grow_from", ["all", "any"])
def test_local_expectation_sloop_expand(grow_from):
    import quimb as qu

    edges = [(0, 1), (0, 2), (2, 3), (1, 3), (2, 4), (3, 5), (4, 5)]
    psi = qtn.TN_from_edges_rand(
        edges,
        D=3,
        phys_dim=2,
        seed=42,
        dist="uniform",
        loc=-0.1,
    )
    G = qu.rand_herm(4)
    where = (0, 2)
    o_ex = psi.local_expectation_exact(G, where)

    gauges = {}
    psi.gauge_all_simple_(100, 5e-6, gauges=gauges)
    psi.normalize_simple(gauges)

    # test loop generation per term
    o_c0 = psi.local_expectation_sloop_expand(
        G, where, sloops=0, gauges=gauges
    )
    assert o_c0 == pytest.approx(
        psi.local_expectation_cluster(G, where, gauges=gauges)
    )
    assert o_ex == pytest.approx(o_c0, rel=0.5, abs=0.01)
    o_c1 = psi.local_expectation_sloop_expand(
        G, where, sloops=4, gauges=gauges, grow_from=grow_from
    )
    assert o_ex == pytest.approx(o_c1, rel=0.5, abs=0.01)
    o_c2 = psi.local_expectation_sloop_expand(
        G, where, sloops=6, gauges=gauges, grow_from=grow_from
    )
    assert o_ex == pytest.approx(o_c2, rel=0.4, abs=0.01)

    # test manual loops supply
    sloops = tuple(psi.gen_paths_loops(6))
    o_cl = psi.local_expectation_sloop_expand(
        G, where, sloops=sloops, gauges=gauges, grow_from=grow_from
    )
    assert o_ex == pytest.approx(o_cl, rel=0.4, abs=0.01)


@pytest.mark.parametrize("grow_from", ["all", "any"])
def test_local_expectation_gloop_expand(grow_from):
    import quimb as qu

    edges = [(0, 1), (0, 2), (2, 3), (1, 3), (2, 4), (3, 5), (4, 5)]
    psi = qtn.TN_from_edges_rand(
        edges,
        D=3,
        phys_dim=2,
        seed=42,
        dist="uniform",
        loc=-0.1,
    )
    G = qu.rand_herm(4)
    where = (0, 2)
    o_ex = psi.local_expectation_exact(G, where)

    gauges = {}
    psi.gauge_all_simple_(100, 5e-6, gauges=gauges)
    psi.normalize_simple(gauges)

    # test cluster generation per term
    o_c0 = psi.local_expectation_gloop_expand(
        G, where, gloops=0, gauges=gauges
    )
    assert o_c0 == pytest.approx(
        psi.local_expectation_cluster(G, where, gauges=gauges)
    )
    assert o_ex == pytest.approx(o_c0, rel=0.5, abs=0.01)
    o_c1 = psi.local_expectation_gloop_expand(
        G, where, gloops=4, gauges=gauges, grow_from=grow_from
    )
    assert o_ex == pytest.approx(o_c1, rel=0.5, abs=0.01)
    o_c2 = psi.local_expectation_gloop_expand(
        G, where, gloops=6, gauges=gauges, grow_from=grow_from
    )
    assert o_ex == pytest.approx(o_c2, rel=0.4, abs=0.01)

    # test manual gloops supply
    gloops = tuple(psi.gen_gloops(4))
    o_cl = psi.local_expectation_gloop_expand(
        G, where, gloops=gloops, gauges=gauges, grow_from=grow_from
    )
    assert o_ex == pytest.approx(o_cl, rel=0.4, abs=0.01)
