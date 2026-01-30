import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


@pytest.mark.parametrize(
    "contract",
    (
        False,
        True,
        "split",
        "reduce-split",
        "split-gate",
        "swap-split-gate",
    ),
)
@pytest.mark.parametrize("where", [("A", "C"), ("B",)])
def test_gate_inds(contract, where):
    tn = qtn.TN_from_edges_rand(
        [("A", "B"), ("B", "C"), ("C", "A")],
        D=3,
        phys_dim=2,
        dtype=complex,
    )
    oix = tn._outer_inds.copy()
    inds = tuple(tn.site_ind(w) for w in where)
    p = tn.to_dense()
    G = qu.rand_matrix(2 ** len(where), dtype=complex)
    tn.gate_inds_(
        G,
        inds=inds,
        contract=contract,
        tags="GATE",
    )
    assert "GATE" in tn.tag_map

    if contract is True:
        assert tn.num_tensors == 4 - len(where)
    elif contract is False:
        assert tn.num_tensors == 4
    elif contract in ("split", "reduce-split"):
        assert tn.num_tensors == 3
        assert len(tn.tag_map["GATE"]) == len(where)
    elif contract in ("split-gate", "swap-split-gate"):
        assert tn.num_tensors == 3 + len(where)
        assert len(tn.tag_map["GATE"]) == len(where)
        assert tn.max_bond() == 4 if len(where) > 1 else 3

    assert tn._outer_inds == oix

    pG = tn.to_dense()
    GIG = qu.pkron(G, [2, 2, 2], list("ABC".index(i) for i in where))
    pGx = GIG @ p
    assert_allclose(pG, pGx)


@pytest.mark.parametrize(
    "contract",
    (
        False,
        True,
        "split",
        "reduce-split",
        "split-gate",
        "swap-split-gate",
    ),
)
@pytest.mark.parametrize("where", [("A", "C"), ("B",)])
def test_gate_sandwich_inds(contract, where):
    tn = qtn.TN_from_edges_rand(
        [("A", "B"), ("B", "C"), ("C", "A")],
        D=3,
        phys_dim=2,
        dtype=complex,
        site_ind_id=("k{}", "b{}"),
    )
    oix = tn._outer_inds.copy()

    # construct reference by densifying
    A = tn.to_dense()
    G = qu.rand_matrix(2 ** len(where), dtype=complex)
    IG = qu.pkron(G, [2, 2, 2], list("ABC".index(i) for i in where))
    GAG = IG @ A @ IG.H

    # apply gate via tensor network method
    inds_upper = tuple(tn.upper_ind(w) for w in where)
    inds_lower = tuple(tn.lower_ind(w) for w in where)
    tn.gate_sandwich_inds_(
        G,
        inds_upper=inds_upper,
        inds_lower=inds_lower,
        contract=contract,
        tags="GATE",
        tags_upper="KET",
        tags_lower="BRA",
    )
    assert "GATE" in tn.tag_map

    if contract is True:
        assert tn.num_tensors == 4 - len(where)
    elif contract is False:
        assert tn.num_tensors == 5
    elif contract in ("split", "reduce-split"):
        assert tn.num_tensors == 3
        assert len(tn.tag_map["GATE"]) == len(where)
        assert len(tn.tag_map["KET"]) == len(where)
        assert len(tn.tag_map["BRA"]) == len(where)
    elif contract in ("split-gate", "swap-split-gate"):
        assert tn.num_tensors == 3 + 2 * len(where)
        assert len(tn.tag_map["GATE"]) == 2 * len(where)
        assert tn.max_bond() == 4 if len(where) > 1 else 3

    assert tn._outer_inds == oix

    x = tn.to_dense()
    assert_allclose(x, GAG)
