import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp
from quimb.tensor.belief_propagation.d2bp import _get_message_conditioner


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
        message = np.diag([0.25, 1.0])

        # no conditioning should return the original message directly
        assert _get_message_conditioner()(message) is message

        condition = _get_message_conditioner(power=0.5)
        assert_allclose(condition(message), np.diag([0.5, 1.0]))

        condition = _get_message_conditioner(power=1.0, smudge=0.1)

        # the message spectrum is squared, so smudge acts on its square root
        assert_allclose(condition(message), np.diag([0.6**2, 1.1**2]))
        condition = _get_message_conditioner(power=0.5, smudge=0.1)
        assert_allclose(condition(message), np.diag([0.6, 1.1]))

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
