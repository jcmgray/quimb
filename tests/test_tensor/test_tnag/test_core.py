import autoray as ar
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


def test_select_sites():
    edges = [(0, 1), (0, 2), (2, 3), (1, 3), (2, 4), (3, 5), (4, 5)]
    psi = qtn.TN_from_edges_rand(edges, D=2, phys_dim=2, seed=42)
    psi.exponent = 1.5

    sub = psi.select_sites([0, 2, 4])
    assert isinstance(sub, psi.__class__)
    assert sub.num_tensors == 3
    assert set(sub.gen_sites_present()) == {0, 2, 4}
    for site in (0, 2, 4):
        assert psi.site_tag(site) in sub.tag_map
    for site in (1, 3, 5):
        assert psi.site_tag(site) not in sub.tag_map
    # exponent not propagated by default
    assert sub.exponent == 0.0
    # virtual=True (default) shares tensor data with parent
    assert sub[0] is psi[0]

    # virtual=False takes copies
    sub_copy = psi.select_sites([0, 2, 4], virtual=False)
    assert sub_copy[0] is not psi[0]
    assert_allclose(sub_copy[0].data, psi[0].data)

    # with_exponent propagates the exponent
    sub_exp = psi.select_sites([0, 2, 4], with_exponent=True)
    assert sub_exp.exponent == 1.5


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


@pytest.mark.parametrize("strip_exponent", [False, True])
def test_normalize_simple_tree_with_exponents(strip_exponent):
    tn = qtn.TN_rand_tree(n=10, D=3, phys_dim=2, max_degree=3, seed=42)
    # generate non trivial exponent attr
    tn.equalize_norms_(1.0)
    nex = tn.norm(strip_exponent=strip_exponent)
    gauges = {}
    tn.gauge_all_simple_(1000, 1e-12, gauges=gauges)
    nge = tn.norm_gloop_expand(0, gauges=gauges, strip_exponent=strip_exponent)
    assert nge == pytest.approx(nex)


@pytest.mark.parametrize(
    "damping,power,smudge,fuse_multibonds",
    [
        (0.0, 1.0, 1e-12, True),  # default path, multibonds fused
        (0.5, 1.0, 1e-12, True),  # damped update
        (0.0, 0.5, 1e-12, True),  # power != 1
        (0.0, 1.0, 0.0, True),  # smudge == 0
        (0.0, 1.0, 1e-12, False),  # multibonds kept separate
    ],
)
def test_gauge_all_simple_options(damping, power, smudge, fuse_multibonds):
    # cyclic 2x2 PEPS has all multibonds
    psi = qtn.PEPS.rand(2, 2, bond_dim=3, cyclic=True, seed=10)
    norm0 = psi.norm()

    opts = dict(
        max_iterations=1000,
        tol=1e-11,
        damping=damping,
        power=power,
        smudge=smudge,
        fuse_multibonds=fuse_multibonds,
    )

    psig = psi.copy()
    psig.gauge_all_simple_(**opts)
    assert psig.norm() == pytest.approx(norm0)
    assert psig.num_indices == 8 if fuse_multibonds else 12

    gauges = {}
    info = {}
    psi.gauge_all_simple_(gauges=gauges, info=info, **opts)
    assert len(gauges) == (4 if fuse_multibonds else 8)

    assert info["iterations"] <= 1000
    if info["iterations"] < 1000:
        assert info["max_sdiff"] < 1e-11
    # the accrued scale is popped back out, not left as a pending exponent
    assert "exponent" not in info


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


# ------------------------- long range gating tests ------------------------- #


NSITES = 6
CHAIN_EDGES = qtn.edges_1d_chain(NSITES)
WHERE = (1, NSITES - 2)


def make_state(edges=CHAIN_EDGES, D=2, phys_dim=2, seed=42):
    """A random arbitrary-geometry state with a converged set of simple-update
    bond gauges.
    """
    psi = qtn.TN_from_edges_rand(
        edges, D=D, phys_dim=phys_dim, seed=seed, dtype=complex
    )
    gauges = {}
    psi.gauge_all_simple_(1000, 1e-13, gauges=gauges)
    return psi, gauges


def physical_state(psi, gauges):
    """The wavefunction TN represented by ``psi`` with its bond ``gauges``
    inserted.
    """
    k = psi.copy()
    k.gauge_simple_insert(gauges)
    return k


def exact_gated(psi, gauges, G, where):
    """Reference: the gauged state with ``G`` applied exactly (lazily, no
    truncation).
    """
    return physical_state(psi, gauges).gate(G, where)


def infidelity(A, B):
    """Normalized infidelity ``1 - |<A|B>|^2 / (|A| |B|)`` between two state
    TNs - invariant to global norm and phase.
    """
    return A.distance(B, normalized="infidelity")


@pytest.mark.parametrize("where", [(1, 4), (1, 3), (2, 4)])
@pytest.mark.parametrize("D", [2, 3])
def test_against_dense_no_truncation(where, D):
    # D is varied because the ``reduced`` mode of the compression sweep relies
    # on the canonical form, which a single bond dimension can mask if wrong
    psi, gauges = make_state(D=D)
    G = qu.rand_uni(4, seed=10)

    ref = exact_gated(psi, gauges, G, where)

    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G,
        where,
        gauges,
        max_bond=None,
        cutoff=0.0,
        renorm=False,
        inplace=True,
    )
    out = physical_state(psi, gauges)

    # no truncation and renorm off -> the represented state is exactly G|psi>,
    # including norm and phase, so the (normalized) Frobenius distance vanishes
    assert out.distance_normalized(ref) == pytest.approx(0.0, abs=1e-6)


def test_peps_long_range_no_truncation():
    # 5x3 PEPS, gate on a vertical pair two sites apart: the path runs
    # (1, 1)-(2, 1)-(3, 1), with the rest of the lattice as environment
    psi = qtn.PEPS.rand(5, 3, bond_dim=2, dtype=complex, seed=42)
    gauges = {}
    psi.gauge_all_simple_(1000, 1e-13, gauges=gauges)

    where = ((1, 1), (3, 1))
    G = qu.rand_uni(4, seed=10)

    ref = exact_gated(psi, gauges, G, where)

    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G,
        where,
        gauges,
        max_bond=None,
        cutoff=0.0,
        renorm=False,
        inplace=True,
    )
    out = physical_state(psi, gauges)

    # with no truncation the gate is applied losslessly to the represented
    # state, regardless of the (approximate) 2D simple-update environment
    assert out.distance_normalized(ref) == pytest.approx(0.0, abs=1e-6)


def test_monotonic_fidelity_increasing_max_bond():
    # larger bond dim so small max_bond actually truncates
    psi0, gauges0 = make_state(D=3)
    where = WHERE
    G = qu.rand_uni(4, seed=11)

    ref = exact_gated(psi0, gauges0, G, where)

    infids = []
    for max_bond in [1, 2, 3, 6, None]:
        psi = psi0.copy()
        gauges = gauges0.copy()
        qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
            psi,
            G,
            where,
            gauges,
            max_bond=max_bond,
            cutoff=0.0,
            inplace=True,
        )
        infids.append(infidelity(ref, physical_state(psi, gauges)))

    # infidelity should (weakly) decrease as more bond dimension is retained
    for ihi, ilo in zip(infids, infids[1:]):
        assert ilo <= ihi + 1e-10

    # and with no cap we recover the exact gated state
    assert infids[-1] == pytest.approx(0.0, abs=1e-9)


def test_identity_gate_no_bond_increase():
    psi, gauges = make_state()
    psi0 = psi.copy()
    gauges0 = gauges.copy()
    where = WHERE
    G = qu.eye(4)

    # bond sizes along the chain before
    before = [
        psi[psi.site_tag(i)].bonds_size(psi[psi.site_tag(j)])
        for i, j in CHAIN_EDGES
    ]

    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G=G,
        where=where,
        gauges=gauges,
        cutoff=1e-12,
        inplace=True,
    )

    after = [
        psi[psi.site_tag(i)].bonds_size(psi[psi.site_tag(j)])
        for i, j in CHAIN_EDGES
    ]
    # identity gate factors to bond 1 -> no bond should grow
    assert after == before

    # and the state is unchanged (up to norm from the gauge renorm)
    assert infidelity(
        physical_state(psi0, gauges0), physical_state(psi, gauges)
    ) == pytest.approx(0.0, abs=1e-10)


def test_no_temporary_tags_injected():
    psi, gauges = make_state()
    tags_before = {t: set(psi[t].tags) for t in psi.tags}

    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G=qu.rand_uni(4, seed=1),
        where=WHERE,
        gauges=gauges,
        inplace=True,
    )

    # none of the internal __TMP{i}__ tags should leak into the network
    for t in psi:
        assert not any(str(tag).startswith("__TMP") for tag in t.tags)
    # the original tag structure is untouched
    assert {t: set(psi[t].tags) for t in psi.tags} == tags_before


def test_index_order_preserved():
    psi, gauges = make_state()
    inds_before = {
        psi.site_tag(i): psi[psi.site_tag(i)].inds for i in range(NSITES)
    }

    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G=qu.rand_uni(4, seed=2),
        where=WHERE,
        gauges=gauges,
        inplace=True,
    )

    for i in range(NSITES):
        tag = psi.site_tag(i)
        # same index names, same order (bond dims may differ)
        assert psi[tag].inds == inds_before[tag]


def test_tensors_modified_inplace():
    psi, gauges = make_state()
    # keep references to the actual tensor objects and their data
    objs = {psi.site_tag(i): psi[psi.site_tag(i)] for i in range(NSITES)}
    data_before = {tag: t.data.copy() for tag, t in objs.items()}

    ret = qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G=qu.rand_uni(4, seed=3),
        where=WHERE,
        gauges=gauges,
        inplace=True,
    )

    assert ret is psi
    # only the tensors along the path between the two sites are touched
    path_sites = set(range(WHERE[0], WHERE[1] + 1))
    for i in range(NSITES):
        tag = psi.site_tag(i)
        t = objs[tag]
        # the Tensor object identity is always preserved
        assert psi[tag] is t
        old = data_before[tag]
        changed = (t.data.shape != old.shape) or not ar.do(
            "allclose", t.data, old
        )
        assert changed is (i in path_sites)


def test_gauges_updated_inplace():
    psi, gauges = make_state()
    before = gauges.copy()

    info = {}
    ret_id = id(gauges)
    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G=qu.rand_uni(4, seed=4),
        where=WHERE,
        gauges=gauges,
        info=info,
        inplace=True,
    )

    # same dict object, updated in place
    assert id(gauges) == ret_id
    # ``info`` reports the string bonds that were (re)gauged
    bond_ixs = [ix for _, ix in info]
    assert bond_ixs
    # they have new gauge values (length may also change)
    for ix in bond_ixs:
        g_new, g_old = gauges[ix], before[ix]
        changed = (g_new.shape != g_old.shape) or not ar.do(
            "allclose", g_new, g_old
        )
        assert changed


@pytest.mark.parametrize("renorm", [True, False])
def test_renorm_normalizes_string_gauges(renorm):
    psi, gauges = make_state()

    info = {}
    qtn.tnag.core.tensor_network_ag_gate_simple_long_range(
        psi,
        G=qu.rand_uni(4, seed=5),
        where=WHERE,
        gauges=gauges,
        renorm=renorm,
        info=info,
        inplace=True,
    )

    bond_ixs = [ix for _, ix in info]
    norms = [ar.do("linalg.norm", gauges[ix]) for ix in bond_ixs]
    ones = [1.0] * len(norms)
    if renorm:
        assert norms == pytest.approx(ones, abs=1e-12)
    else:
        # raw singular values are generally not unit-norm
        assert norms != pytest.approx(ones)
