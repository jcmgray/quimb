import importlib.util

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp
from quimb.tensor.belief_propagation.d2bp import _get_message_conditioner


@pytest.mark.parametrize("inplace", [False, True])
def test_converge_d2bp(inplace):
    tn = qtn.TN_rand_tree(6, 2, phys_dim=2, max_degree=3, seed=42)
    info = {}

    bp = qbp.converge_d2bp(
        tn,
        power=0.75,
        smudge=1e-6,
        max_iterations=100,
        tol=1e-10,
        inplace=inplace,
        info=info,
    )

    assert isinstance(bp, qbp.D2BP)
    assert bp.tn is tn if inplace else bp.tn is not tn
    assert bp.power == 0.75
    assert bp.smudge == 1e-6
    assert info["converged"]


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("diis", [True, False])
def test_contract(damping, dtype, diis):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # normalize exactly
    peps /= (peps.H @ peps) ** 0.5
    info = {}
    N_ap = qbp.contract_d2bp(
        peps, damping=damping, diis=diis, info=info, progbar=True
    )
    assert info["converged"]
    assert N_ap == pytest.approx(1.0, rel=0.3)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract_with_exponent(dtype):
    tn = qtn.TN_rand_tree(
        10, 3, phys_dim=2, max_degree=4, seed=42, dtype=dtype
    )
    Zex = tn.H @ tn
    tn.equalize_norms_(1.7)
    assert tn.exponent
    bp = qbp.D2BP(tn)
    bp.run()
    assert bp.contract() == pytest.approx(Zex, rel=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("local_convergence", [True, False])
def test_tree_exact(dtype, local_convergence):
    psi = qtn.TN_rand_tree(20, 3, 2, dtype=dtype, seed=42)
    norm2 = psi.H @ psi
    info = {}
    norm2_bp = qbp.contract_d2bp(
        psi, info=info, local_convergence=local_convergence, progbar=True
    )
    assert info["converged"]
    assert norm2_bp == pytest.approx(norm2, rel=1e-4)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("diis", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_compress(damping, dtype, diis):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # test that using the BP compression gives better fidelity than purely
    # local, naive compression scheme
    peps_c1 = peps.compress_all(max_bond=2)
    info = {}
    peps_c2 = peps.copy()
    qbp.compress_d2bp(
        peps_c2,
        max_bond=2,
        damping=damping,
        diis=diis,
        info=info,
        inplace=True,
        progbar=True,
    )
    assert peps_c2.max_bond() == 2
    assert info["converged"]
    fid1 = peps_c1.H @ peps_c2
    fid2 = peps_c2.H @ peps_c2
    assert abs(fid2) > abs(fid1)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_sample(dtype):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # normalize exactly
    peps /= (peps.H @ peps) ** 0.5
    config, peps_config, omega = qbp.sample_d2bp(peps, seed=42, progbar=True)
    assert all(ix in config for ix in peps.site_inds)
    assert 0.0 < omega < 1.0
    assert peps_config.outer_inds() == ()

    ptotal = 0.0
    nrepeat = 4
    for _ in range(nrepeat):
        _, peps_config, _ = qbp.sample_d2bp(peps, seed=42, progbar=True)
        ptotal += abs(peps_config.contract()) ** 2

    # check we are doing better than random guessing
    assert ptotal > nrepeat * 2**-peps.nsites


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_loop_series_expansion_order0_matches_partial_trace(dtype):
    # see gh-380
    peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype=dtype)
    bp = qbp.D2BP(peps)
    bp.run(max_iterations=1000, tol=1e-12)

    where = [(1, 1)]
    rho_pt = bp.partial_trace(where)
    rho_ge = bp.partial_trace_gloop_expand(where, gloops=0)
    rho_ls = bp.partial_trace_loop_series_expansion(
        where, gloops=0, multi_excitation_correct=False
    )

    assert_allclose(rho_ls, rho_pt, atol=1e-10)
    assert_allclose(rho_ls, rho_ge, atol=1e-10)
    assert_allclose(rho_ls, rho_ls.conj().T, atol=1e-10)

    rho_ls4 = bp.partial_trace_loop_series_expansion(
        where, gloops=4, multi_excitation_correct=False
    )
    assert_allclose(rho_ls4, rho_ls4.conj().T, atol=1e-10)


@pytest.mark.parametrize(
    "method",
    ["partial_trace_loop_series_expansion", "partial_trace_gloop_expand"],
)
def test_partial_trace_gloop_tree_returns_base(method):
    tn = qtn.TN_from_edges_rand(
        [(0, 1), (1, 2), (2, 3)], D=2, phys_dim=2, seed=42
    )
    bp = qbp.D2BP(tn)
    bp.run(tol=1e-12)
    with pytest.warns(UserWarning, match="only the target region"):
        rho = getattr(bp, method)([1])
    assert rho.shape == (2, 2)
    assert_allclose(np.trace(rho), 1.0)
    # on a tree bp is exact, so the fallback must give the base region rdm
    assert_allclose(rho, bp.partial_trace([1], get="matrix"))


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_loop_series_expansion_repeatable(dtype):
    # see gh-381
    peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype=dtype)
    bp = qbp.D2BP(peps)
    bp.run(max_iterations=1000, tol=1e-12)

    where = [(1, 1)]

    def loop_rdm():
        return bp.partial_trace_loop_series_expansion(
            where, gloops=4, multi_excitation_correct=False
        )

    r1 = loop_rdm()
    r2 = loop_rdm()
    r3 = loop_rdm()
    assert_allclose(r1, r2, atol=1e-12)
    assert_allclose(r1, r3, atol=1e-12)


def test_loop_series_expansion_converges_to_exact():
    # every pair of loops shares a site, so the connected series is exact
    peps = qtn.PEPS.rand(3, 3, bond_dim=2, seed=42, dist="uniform", loc=0.5)
    zex = (peps.H | peps).contract(output_inds=())

    bp = qbp.D2BP(peps)
    bp.run(tol=1e-13)
    z = bp.contract_loop_series_expansion(
        gloops=9,
        multi_excitation_correct=False,
    )
    assert z == pytest.approx(zex, rel=1e-12)


@pytest.mark.parametrize("seed", range(2))
def test_gate(seed):
    peps = qtn.PEPS.rand(3, 4, 3, seed=seed)
    peps.normalize_()
    G = qu.rand_uni(4, seed=seed)
    where = [(1, 1), (1, 2)]
    peps_g_ex = peps.gate(G, where, contract=False)
    # compute with no gauging
    peps_g_basic = peps.gate(G, where, contract="reduce-split", max_bond=3)
    d1 = peps_g_basic.distance_normalized(peps_g_ex)
    # run BP
    bp = qbp.D2BP(peps)
    bp.run()
    # gate with BP gauging
    bp.gate_(G, where, max_bond=3)
    bp.run()
    d2 = bp.tn.distance_normalized(peps_g_ex)
    assert d2 < d1
    assert abs(bp.contract()) ** 0.5 > 0.5


class TestMessageConditioner:
    def test_matches_simple_gauge_spectrum(self):
        message = np.diag([0.25, 4.0])

        # trivial conditioning is signalled by None
        assert _get_message_conditioner() is None

        condition = _get_message_conditioner(power=0.5)
        assert_allclose(condition(message), np.diag([0.5, 2.0]))

        condition = _get_message_conditioner(power=1.0, smudge=0.1)

        # smudge is relative to the largest square-root eigenvalue
        assert_allclose(condition(message), np.diag([0.7**2, 2.2**2]))
        condition = _get_message_conditioner(power=0.5, smudge=0.1)
        assert_allclose(condition(message), np.diag([0.7, 2.2]))

        # relative conditioning commutes with overall message scaling
        assert_allclose(condition(9 * message), 3 * condition(message))

    def test_properties_update_conditioner(self):
        tn = qtn.TN_rand_tree(4, 2, 2, seed=1)
        bp = qbp.D2BP(tn)
        bp.touched.clear()

        bp.power = 0.5
        assert bp.power == 0.5
        assert bp._message_conditioner is _get_message_conditioner(
            0.5,
            0.0,
            bp.backend,
        )
        assert set(bp.touched) == set(bp.exprs)

        bp.touched.clear()
        bp.smudge = 0.1
        assert bp.smudge == 0.1
        assert bp._message_conditioner is _get_message_conditioner(
            0.5,
            0.1,
            bp.backend,
        )
        assert set(bp.touched) == set(bp.exprs)


class TestConditionedMessageStore:
    def test_power_one_uses_raw_messages(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        bp = qbp.D2BP(peps)
        bp.run(max_iterations=100, tol=1e-10)
        # trivial conditioning: no separate copies stored at all
        assert bp._message_conditioner is None
        assert not bp._messages_conditioned
        key = next(iter(bp.messages))
        assert bp._get_message_conditioned(key) is bp.messages[key]

    def test_messages_stored_raw_and_cache_consistent(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        bp = qbp.D2BP(peps, power=0.5)
        bp.run(max_iterations=1000, tol=1e-12)
        # insertion environments have been computed and cached
        assert bp._messages_conditioned
        cond = bp._message_conditioner
        for key, mc in bp._messages_conditioned.items():
            m = bp.messages[key]
            # cache matches conditioning the raw message
            assert_allclose(mc, cond(m), atol=1e-12)
            # and the stored message itself is unconditioned
            assert not np.allclose(mc, m)

    def test_get_message_with_conditioning_override(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        bp = qbp.D2BP(peps)
        bp.run(max_iterations=100, tol=1e-10)
        key = next(iter(bp.messages))
        m = bp.messages[key]

        assert bp.get_message(key, power=1.0, smudge=0.0) is m

        condition = _get_message_conditioner(0.5, 0.0, bp.backend)
        assert_allclose(
            bp.get_message(key, power=0.5, smudge=0.0),
            condition(m),
        )

    def test_fixed_point_of_insertion_conditioning(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        bp = qbp.D2BP(peps, power=0.5)
        bp.run(max_iterations=1000, tol=1e-13)
        old = {k: m.copy() for k, m in bp.messages.items()}
        # a further full sweep should not change the raw messages
        bp.touched.update(bp.exprs)
        bp.iterate(tol=1e-13)
        for key, m in bp.messages.items():
            assert m == pytest.approx(old[key], abs=1e-10)

    def test_cache_invalidated_by_setters_and_reassignment(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        bp = qbp.D2BP(peps, power=0.5)
        bp.run(max_iterations=10, tol=1e-10)
        assert bp._messages_conditioned
        bp.power = 0.25
        assert not bp._messages_conditioned
        bp.run(max_iterations=10, tol=1e-10)
        assert bp._messages_conditioned
        # wholesale reassignment (e.g. by DIIS) also clears the cache
        bp.messages = dict(bp.messages)
        assert not bp._messages_conditioned

    def test_diis_with_power(self):
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        info = {}
        bp = qbp.D2BP(peps, power=0.5)
        bp.run(max_iterations=1000, tol=1e-10, diis=True, info=info)
        assert info["converged"]
        cond = bp._message_conditioner
        for key, mc in bp._messages_conditioned.items():
            assert_allclose(mc, cond(bp.messages[key]), atol=1e-12)

    @pytest.mark.parametrize("gauge_power", [1.0, 0.5])
    def test_gauge_symmetric_writes_back_raw_messages(self, gauge_power):
        # the gauging power conditions the projectors, not the stored
        # messages, which must remain a fixed point of the p=1 iteration
        peps = qtn.PEPS.rand(3, 3, 2, seed=42)
        bp = qbp.D2BP(peps)
        bp.run(max_iterations=1000, tol=1e-13)
        bp.gauge_symmetric(power=gauge_power, inplace=True)

        old = {k: m / np.linalg.norm(m) for k, m in bp.messages.items()}
        bp.touched.update(bp.exprs)
        bp.iterate(tol=1e-13)
        for key, m in bp.messages.items():
            m = m / np.linalg.norm(m)
            assert m == pytest.approx(old[key], abs=1e-10)

    @pytest.mark.parametrize("max_bond", [2, 64])
    @pytest.mark.parametrize("dtype", ["float64", "complex128"])
    def test_compress_writes_back_positive_messages(self, dtype, max_bond):
        # check the transpose and conjugation used by inplace writeback
        peps = qtn.PEPS.rand(3, 3, 2, seed=42, dtype=dtype)
        bp = qbp.D2BP(peps)
        bp.run(max_iterations=1000, tol=1e-13)
        bp.compress(max_bond=max_bond, cutoff=0.0, inplace=True)

        for m in bp.messages.values():
            m = m / np.linalg.norm(m)
            assert_allclose(m, m.conj().T, atol=1e-12)
            assert np.linalg.eigvalsh(m).min() > -1e-12


def test_gauge_insert_conditioning_and_inverse():
    tn = qtn.TN_rand_tree(2, 2, 2, seed=42)
    bp = qbp.D2BP(tn)
    ix = next(iter(bp.tn.inner_inds()))
    tid = next(iter(bp.tn.ind_map[ix]))
    bp.messages[ix, tid] = np.diag([0.25, 4.0])

    local = bp.tn._select_tids((tid,), virtual=False)
    original = local.copy()
    expected = local.copy()
    s = np.array([0.7, 2.2]) ** 0.5
    expected.tensor_map[tid].gate_(np.diag(s), ix)

    outer = bp.gauge_insert(local, power=0.5, smudge=0.1)
    assert local.distance_normalized(expected) == pytest.approx(
        0.0,
        abs=1e-7,
    )

    for t, jx, minv in outer:
        t.gate_(minv, jx)
    assert local.distance_normalized(original) == pytest.approx(
        0.0,
        abs=1e-7,
    )

    local = original.copy()
    outer = local.gauge_insert(
        bp,
        power=0.5,
        smudge=0.1,
        return_gauges="raw",
    )
    assert_allclose(outer[0][2], np.diag(s))
    assert local.distance_normalized(expected) == pytest.approx(
        0.0,
        abs=1e-7,
    )

    local = original.copy()
    outer = local.gauge_insert(
        bp,
        power=0.5,
        smudge=0.1,
        return_gauges=None,
    )
    assert outer is None
    assert local.distance_normalized(expected) == pytest.approx(
        0.0,
        abs=1e-7,
    )


@pytest.mark.parametrize("power", [1.0, 0.5])
@pytest.mark.parametrize("gauge_power", [1.0, 0.5])
def test_gauge_all_belief_propagation_powers(power, gauge_power):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42)
    tng = peps.gauge_all_belief_propagation(
        max_iterations=500,
        tol=1e-10,
        power=power,
        smudge=0.0,
        gauge_power=gauge_power,
    )
    # any combination of powers still only regauges the network
    assert tng.distance_normalized(peps) == pytest.approx(0.0, abs=1e-6)


def test_compress_d2bp_with_powers():
    peps = qtn.PEPS.rand(3, 4, 3, seed=42)
    tnc = qbp.compress_d2bp(
        peps,
        max_bond=2,
        power=0.5,
        gauge_power=0.5,
        tol=1e-10,
    )
    assert tnc.max_bond() == 2
    # random PEPS compress poorly, just check the result is sensible
    fid = abs(tnc.H @ peps) / (
        abs(tnc.H @ tnc) ** 0.5 * abs(peps.H @ peps) ** 0.5
    )
    assert fid > 0.2


@pytest.mark.parametrize("gauge_power", [1.0, 0.5])
def test_insert_compressor_with_d2bp_gauges(gauge_power):
    peps = qtn.PEPS.rand(3, 4, 2, dtype="complex128", seed=42)
    bp = qbp.D2BP(peps)
    bp.run(max_iterations=1000, tol=1e-12)

    ltags = (peps.site_tag(1, 0), peps.site_tag(1, 1))
    rtags = (peps.site_tag(1, 2), peps.site_tag(1, 3))
    pair = peps.select_any(ltags + rtags, virtual=False)
    geometry_hash = pair.geometry_hash()

    # construct the previous explicit message-square-root reference
    pair_gauged = pair.copy()
    for ix in pair_gauged.outer_inds():
        (tid,) = pair_gauged.ind_map[ix]
        try:
            m = bp.get_message(
                (ix, tid),
                power=gauge_power,
                smudge=0.0,
            )
        except KeyError:
            continue
        el, ev = np.linalg.eigh(m)
        el = np.clip(el, 0.0, None) ** 0.5
        msqrt = np.diag(el) @ ev.conj().T
        pair_gauged.tensor_map[tid].gate_(msqrt, ix)

    for mode in ("oblique", "nystrom"):
        qu.seed_rand(123)
        reference = pair.copy()
        pair_gauged.insert_compressor_between_regions_(
            ltags,
            rtags,
            max_bond=1,
            cutoff=0.0,
            mode=mode,
            insert_into=reference,
        )

        qu.seed_rand(123)
        direct = pair.insert_compressor_between_regions(
            ltags,
            rtags,
            max_bond=1,
            cutoff=0.0,
            mode=mode,
            gauges=bp,
            gauge_power=gauge_power,
        )

        assert pair.geometry_hash() == geometry_hash
        assert direct.distance_normalized(reference) == pytest.approx(
            0.0,
            abs=1e-6,
        )


@pytest.mark.parametrize("power", [0.75, 1.0])
@pytest.mark.parametrize("smudge", [0.0, 0.33])
def test_gauge_symmetric_with_conditioning(power, smudge):
    tn = qtn.TN_rand_tree(
        8,
        3,
        phys_dim=2,
        max_degree=3,
        seed=42,
    )
    bp = qbp.D2BP(tn, power=power, smudge=smudge)
    bp.run(max_iterations=1000, tol=1e-10)
    tn_before = bp.tn.copy()

    tn_gauged = bp.gauge_symmetric(inplace=True)

    assert tn_gauged is bp.tn
    assert tn_gauged.distance_normalized(tn_before) == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert tn_gauged.distance_normalized(tn) == pytest.approx(
        0.0,
        abs=1e-7,
    )

    # check the updated messages are now symmetric and diagonal
    for ix, tids in bp.tn.ind_map.items():
        if len(tids) != 2:
            continue
        tida, tidb = tids
        ma = bp.messages[ix, tida]
        mb = bp.messages[ix, tidb]
        assert ma == pytest.approx(mb, abs=1e-10)
        assert ma == pytest.approx(np.diag(np.diag(ma)), abs=1e-10)

    result = bp.iterate(tol=1e-10)
    assert result["ncheck"] == len(bp.exprs)


@pytest.mark.parametrize("inplace", [False, True])
def test_gauge_all_belief_propagation(inplace):
    tn = qtn.TN_rand_tree(
        8,
        3,
        phys_dim=2,
        max_degree=3,
        seed=42,
    )
    tn_before = tn.copy()
    messages = {}
    info = {}

    if inplace:
        gauge = tn.gauge_all_belief_propagation_
    else:
        gauge = tn.gauge_all_belief_propagation

    tn_gauged = gauge(
        messages=messages,
        max_iterations=1000,
        tol=1e-10,
        info=info,
    )

    assert (tn_gauged is tn) is inplace
    assert info["converged"]
    assert tn_gauged.distance_normalized(tn_before) == pytest.approx(
        0.0,
        abs=1e-7,
    )

    for ix, tids in tn_gauged.ind_map.items():
        if len(tids) != 2:
            continue
        tida, tidb = tids
        assert messages[ix, tida] == pytest.approx(
            messages[ix, tidb],
            abs=1e-10,
        )


requires_symmray = pytest.mark.skipif(
    importlib.util.find_spec("symmray") is None,
    reason="symmray not installed",
)


def _positive_message_dense(m):
    import autoray as ar

    if m.fermionic and not m.indices[1].dual:
        m = m.phase_flip(1)
    return ar.to_numpy(m.to_dense())


@requires_symmray
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("odd", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
class TestSymmetricMessages:
    """Check compression across parity and bond orientation."""

    @staticmethod
    def _scalar_tn(fermionic, odd, reverse, dtype="float64", seed=42):
        """Build a scalar network with optional reversed bond dualness."""
        import symmray as sr

        if odd:
            # odd sites on the diagonal, keeping the total charge even
            def site_charge(site):
                return int(site[0] == site[1])

        else:

            def site_charge(site):
                return 0

        tn = sr.TN_abelian_from_edges_rand(
            symmetry="Z2",
            edges=qtn.edges_2d_square(4, 4),
            bond_dim=2,
            phys_dim=None,
            fermionic=fermionic,
            site_charge=site_charge,
            dtype=dtype,
            seed=seed,
        )
        if reverse:
            tn = qtn.TensorNetwork(list(tn.tensors)[::-1])
        return tn

    def test_compress_full_rank_is_exact(self, fermionic, odd, reverse):
        if odd and not fermionic:
            pytest.skip("charge only matters for fermionic arrays")
        tn = self._scalar_tn(fermionic, odd, reverse)
        expected = tn.contract(all, optimize="auto-hq")
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=1000, tol=1e-12)

        # full rank projectors multiply to the identity
        tnc = bp.compress(max_bond=64, cutoff=0.0)
        value = tnc.contract(all, optimize="auto-hq")
        assert value == pytest.approx(expected, rel=1e-10)

    def test_compress_writes_back_positive_messages(
        self, fermionic, odd, reverse
    ):
        # complex data distinguishes a transpose from a dagger
        if odd and not fermionic:
            pytest.skip("charge only matters for fermionic arrays")
        tn = self._scalar_tn(fermionic, odd, reverse, dtype="complex128")
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=1000, tol=1e-13)
        bp.compress(max_bond=64, cutoff=0.0, inplace=True)

        for m in bp.messages.values():
            d = _positive_message_dense(m)
            d = d / np.linalg.norm(d)
            assert_allclose(d, d.conj().T, atol=1e-12)
            assert np.linalg.eigvalsh(d).min() > -1e-12

    def test_compress_twice_reuses_written_back_messages(
        self, fermionic, odd, reverse
    ):
        # a second compression checks that writeback preserves bond dualness
        if odd and not fermionic:
            pytest.skip("charge only matters for fermionic arrays")
        tn = self._scalar_tn(fermionic, odd, reverse, dtype="complex128")
        expected = tn.contract(all, optimize="auto-hq")
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=1000, tol=1e-13)

        # full rank both times, so only identities are ever inserted
        bp.compress(max_bond=64, cutoff=0.0, inplace=True)
        tnc = bp.compress(max_bond=64, cutoff=0.0, inplace=True)
        value = tnc.contract(all, optimize="auto-hq")
        assert value == pytest.approx(expected, rel=1e-8)

    def test_gauge_symmetric_writes_back_fixed_point(
        self, fermionic, odd, reverse
    ):
        if odd and not fermionic:
            pytest.skip("charge only matters for fermionic arrays")
        peps = self._scalar_tn(fermionic, odd, reverse)
        bp = qbp.D2BP(peps)
        bp.run(max_iterations=1000, tol=1e-13)
        bp.gauge_symmetric(inplace=True)

        old = {k: m / m.norm() for k, m in bp.messages.items()}
        bp.touched.update(bp.exprs)
        bp.iterate(tol=1e-13)
        for key, m in bp.messages.items():
            assert float((m / m.norm() - old[key]).norm()) < 1e-8


@requires_symmray
@pytest.mark.parametrize("duals", ["reversed", "canonical", "random"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("charge", [0, 1])
class TestSymmrayFermionicEnvironments:
    @staticmethod
    def make_tree(duals, flat, charge, phys_dim=2):
        import symmray as sr

        return sr.TN_abelian_from_edges_rand(
            "Z2",
            [(0, 1), (1, 2), (1, 3)],
            bond_dim=4,
            phys_dim=phys_dim,
            fermionic=True,
            flat=flat,
            subsizes="equal",
            duals=duals,
            site_charge=lambda site: charge,
            dtype="complex128",
            seed=42,
        )

    @staticmethod
    def check_messages(bp):
        for (ix, tid), m in bp.messages.items():
            assert not m.dummy_modes
            d = _positive_message_dense(m)
            assert_allclose(d, d.conj().T, atol=1e-10)
            assert np.linalg.eigvalsh(d).min() > -1e-10
            t = bp.tn.tensor_map[tid]
            assert m.duals[1] != t.data.duals[t.inds.index(ix)]

    def test_tree_norm_and_conditioning(self, duals, flat, charge):
        tn = self.make_tree(duals, flat, charge)
        bp = qbp.D2BP(tn)
        assert bp.normalize == "L2"
        assert bp.distance == "L2"
        self.check_messages(bp)
        bp.run(max_iterations=10, tol=1e-12)
        self.check_messages(bp)
        assert bp.contract() == pytest.approx(tn.norm() ** 2, rel=1e-10)

        for key, m in bp.messages.items():
            lazy = (-m).phase_global()
            assert_allclose(
                bp._normalize_fn(lazy).to_dense(),
                bp._normalize_fn(m).to_dense(),
                atol=1e-12,
            )
            d = _positive_message_dense(m)
            s2, w = np.linalg.eigh(d)
            s = np.sqrt(np.clip(s2, 0.0, None))
            expected = (w * (s + 0.01 * s.max()) ** 1.5) @ w.conj().T
            mc = bp.get_message(key, power=0.75, smudge=0.01)
            assert_allclose(_positive_message_dense(mc), expected, atol=1e-10)

        bp.power = 0.75
        bp.smudge = 0.01
        bp.iterate()
        self.check_messages(bp)

    @pytest.mark.parametrize(
        "where, tids_region",
        [
            ((1,), None),
            ((0, 1), None),
            ((0, 2), (0, 1, 2)),
            ((0, 2), (0, 1, 2, 3)),
        ],
    )
    def test_cluster_density_matrix(
        self, duals, flat, charge, where, tids_region
    ):
        tn = self.make_tree(duals, flat, charge)
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=10, tol=1e-12)
        # include the connecting path so the tree environment is exact
        rho = bp.partial_trace(
            where, tids_region=tids_region, normalized=False
        ).to_dense()
        exact = tn.partial_trace_exact(where, normalized=False).to_dense()
        assert_allclose(
            rho / np.trace(rho), exact / np.trace(exact), atol=1e-12
        )

    def test_direct_environments(self, duals, flat, charge, monkeypatch):
        import autoray as ar

        tn = self.make_tree(duals, flat, charge)
        bp = qbp.D2BP(tn, update="parallel")
        t = bp.tn.tensor_map[1]
        out = next(ix for ix in t.inds if ix not in bp.output_inds)
        (neighbor,) = (tid for tid in tn.ind_map[out] if tid != 1)

        # independently weight both copies with square-root factors
        weighted = t.copy()
        for ix in t.inds:
            if ix == out or ix in bp.output_inds:
                continue
            m = bp.messages[ix, 1]
            s2, w = m.eigh()
            s = ar.do("sqrt", ar.do("clip", s2, 0.0, None))
            weighted.gate_(qtn.decomp.ldmul(s, ar.dag(w)), ix)
        expected = ar.do(
            "gram", weighted.data, axes=weighted.inds.index(out)
        ).to_dense()
        expected /= np.linalg.norm(expected)

        def no_factorization(*args, **kwargs):
            raise AssertionError("environment message was factorized")

        monkeypatch.setattr(type(m), "eigh", no_factorization)
        with monkeypatch.context() as patch:

            def no_gate(*args, **kwargs):
                raise AssertionError("environment message was eagerly gated")

            patch.setattr(qtn.Tensor, "gate_", no_gate)
            bp.iterate()
            actual = bp.messages[out, neighbor].to_dense()
            assert_allclose(actual, expected, atol=1e-12)

            bp.run(max_iterations=10, tol=1e-12)
            assert bp.contract() == pytest.approx(tn.norm() ** 2, rel=1e-10)
            bp.get_cluster_norm((0, 1)).contract(all)
            local = tn._select_tids((1,), virtual=False)
            rest = tuple(ix for ix in t.inds if ix != out)
            for side in ("left", "right"):
                local.compute_reduced_factor(side, rest, (out,), gauges=bp)

        compressed = tn.insert_compressor_between_regions(
            [tn.site_tag(0)],
            [tn.site_tag(1)],
            gauges=bp,
            max_bond=4,
            cutoff=0.0,
        )
        assert compressed.distance_normalized(tn) < 1e-7

    def test_gate_and_reuse(self, duals, flat, charge, request):
        import symmray as sr

        if flat:
            # the tree has leaves, so reduce-split gating makes a size-1 bond
            request.applymarker(
                pytest.mark.xfail(
                    reason="flat fusing assumes every grouped axis carries "
                    "all charges, so the size-1 reduced bond breaks it",
                    strict=True,
                )
            )

        tn = self.make_tree(duals, flat, charge)
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=10, tol=1e-12)
        gate = sr.utils.get_rand(
            "Z2",
            (2, 2, 2, 2),
            duals=(False, False, True, True),
            fermionic=True,
            flat=flat,
            subsizes="equal",
            dtype="complex128",
            seed=43,
        )
        expected = tn.gate(gate, (0, 1), contract=False)
        bp.gate_(gate, (0, 1), max_bond=8, contract="reduce-split")
        assert bp.tn.distance_normalized(expected) < 1e-7
        self.check_messages(bp)

        # truncation changes the bond, then another gate reuses the messages
        bp.gate_(gate, (0, 1), max_bond=2)
        self.check_messages(bp)
        assert bp.tn.bond_size(0, 1) <= 2
        bp.run(max_iterations=10, tol=1e-12)
        self.check_messages(bp)
        assert bp.contract() == pytest.approx(bp.tn.norm() ** 2, rel=1e-10)

        one_site_gate = sr.utils.get_rand(
            "Z2",
            (2, 2),
            duals=(False, True),
            fermionic=True,
            flat=flat,
            subsizes="equal",
            dtype="complex128",
            seed=44,
        )
        bp.gate_(one_site_gate, (1,))
        bp.run(max_iterations=10, tol=1e-12)
        assert bp.contract() == pytest.approx(bp.tn.norm() ** 2, rel=1e-10)

    def test_projector_truncation(self, duals, flat, charge):
        tn = self.make_tree(duals, flat, charge, phys_dim=4)
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=10, tol=1e-12)
        compressed = tn.insert_compressor_between_regions(
            [tn.site_tag(0)],
            [tn.site_tag(1)],
            gauges=bp,
            max_bond=2,
            cutoff=0.0,
        )
        x = tn.to_dense(
            (tn.site_ind(0),),
            tuple(tn.site_ind(i) for i in (1, 2, 3)),
        )
        u, _, v = qtn.decomp.svd_truncated(x, max_bond=2, cutoff=0.0)
        expected_error = (x - u @ v).norm()
        assert expected_error > 0.0
        assert compressed.distance(tn) == pytest.approx(
            expected_error, rel=1e-9
        )

    def test_gauge_temp_restores_after_error(self, duals, flat, charge):
        bp = qbp.D2BP(self.make_tree(duals, flat, charge))
        bp.run(max_iterations=10, tol=1e-12)
        local = bp.tn.select(1)
        before = local.tensors[0].data.copy()
        with (
            pytest.raises(ValueError, match="test error"),
            bp.gauge_temp(local),
        ):
            raise ValueError("test error")
        after = local.tensors[0].data
        assert float((after - before).norm()) < 1e-10
        assert after.duals == before.duals


class TestMessageMaintenance:
    @pytest.mark.parametrize(
        "fermionic", [False, pytest.param(True, marks=requires_symmray)]
    )
    @pytest.mark.parametrize("strip_exponent", [False, True])
    def test_normalize_refreshes_expressions(self, fermionic, strip_exponent):
        if fermionic:
            tn = TestSymmrayFermionicEnvironments.make_tree("random", False, 1)
        else:
            tn = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=42)
        bp = qbp.D2BP(tn, local_convergence=False)
        bp.run(max_iterations=10, tol=1e-12)
        bp.normalize_tensors(strip_exponent=strip_exponent)
        for (ix, dest), (_, data) in bp.exprs.items():
            (src,) = (tid for tid in bp.tn.ind_map[ix] if tid != dest)
            assert data[0] is bp.tensor_dual_map[src].data
            assert data[1] is bp.tn.tensor_map[src].data
        reference = qbp.D2BP(
            bp.tn,
            messages={k: v.copy() for k, v in bp.messages.items()},
            local_convergence=False,
        )
        reference.sign = bp.sign
        reference.exponent = bp.exponent
        bp.iterate()
        reference.iterate()
        for key in bp.messages:
            assert (
                bp._distance_fn(bp.messages[key], reference.messages[key])
                < 1e-12
            )
        assert bp.contract() == pytest.approx(reference.contract(), rel=1e-10)
        if not strip_exponent:
            assert bp.contract() == pytest.approx(bp.tn.norm() ** 2, rel=1e-10)

    def test_complex_gate_message_order(self):
        tn = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=42)
        bp = qbp.D2BP(tn)
        bp.run(max_iterations=10, tol=1e-12)
        bp.gate_(qu.rand_uni(4, seed=43), (1, 2), max_bond=8)
        with bp.gauge_temp(bp.tn.select_any(["I1", "I2"])):
            ta, tb = bp.tn[1], bp.tn[2]
            left, (ix,), right = qtn.group_inds(ta, tb)
            a = ta.to_dense(left, (ix,))
            b = tb.to_dense((ix,), right)
            expected = (a.conj().T @ a, (b @ b.conj().T).T)
            for tensor, m in zip((tb, ta), expected):
                (tid,) = bp.tn._get_tids_from_tags(tensor.tags)
                actual = bp.messages[ix, tid]
                assert_allclose(actual, m, atol=1e-10)

    @pytest.mark.parametrize("where", [(1,), (1, 2)])
    def test_gate_rejects_external_tn(self, where):
        tn = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=42)
        bp = qbp.D2BP(tn)
        external = bp.tn.copy()
        gate = qu.rand_uni(2 ** len(where), seed=43)
        with pytest.raises(ValueError, match="managed tensor network"):
            bp.gate_(gate, where, tn=external)
        assert external.distance_normalized(bp.tn) < 1e-12


@requires_symmray
@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("flat", [False, True])
def test_symmray_fermionic_bp_backend(backend, flat, request):
    import autoray as ar
    import symmray as sr

    module = pytest.importorskip(backend)
    if backend == "jax":
        module.config.update("jax_enable_x64", True)
    tn = TestSymmrayFermionicEnvironments.make_tree("random", flat, 1)
    tn.apply_to_arrays(lambda x: ar.to(x, backend))
    bp = qbp.D2BP(tn, power=0.75, smudge=0.01)
    bp.run(max_iterations=10, tol=1e-10)
    compressed = tn.insert_compressor_between_regions(
        [tn.site_tag(0)],
        [tn.site_tag(1)],
        gauges=bp,
        max_bond=4,
        cutoff=0.0,
    )
    compressed_np = compressed.copy()
    tn_np = tn.copy()
    compressed_np.apply_to_arrays(lambda x: ar.to(x, "numpy"))
    tn_np.apply_to_arrays(lambda x: ar.to(x, "numpy"))
    assert_allclose(
        compressed_np.to_dense().to_dense(),
        tn_np.to_dense().to_dense(),
        atol=1e-7,
    )
    gate = sr.utils.get_rand(
        "Z2",
        (2, 2, 2, 2),
        duals=(False, False, True, True),
        fermionic=True,
        flat=flat,
        subsizes="equal",
        dtype="complex128",
        seed=43,
    )
    gate = qtn.Tensor(gate, inds="abcd").to(backend=backend).data
    if flat:
        # BP and compression work; only the following degree-1 gate is blocked
        request.applymarker(
            pytest.mark.xfail(
                reason="flat reduce-split gating of a degree-1 site is "
                "blocked by symmray fuse handling",
                strict=True,
            )
        )
    bp.gate_(gate, (0, 1), max_bond=2)
    bp.iterate()
    for m in bp.messages.values():
        assert m.backend == backend
