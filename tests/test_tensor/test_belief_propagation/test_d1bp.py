from unittest import mock

import numpy as np
import pytest

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp


@pytest.mark.parametrize("local_convergence", [False, True])
def test_contract_tree_exact(local_convergence):
    tn = qtn.TN_rand_tree(20, 3)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_d1bp(
        tn, info=info, local_convergence=local_convergence, progbar=True
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-10)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract_with_exponent(dtype):
    tn = qtn.TN_rand_tree(10, 3, max_degree=4, seed=42, dtype=dtype)
    Zex = tn.contract()
    tn.equalize_norms_(1.7)
    assert tn.exponent
    bp = qbp.D1BP(tn)
    bp.run()
    assert bp.contract() == pytest.approx(Zex, rel=1e-5)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("diis", [False, True])
def test_contract_normal(damping, diis):
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    info = {}
    Z_bp = qbp.contract_d1bp(
        tn, damping=damping, diis=diis, info=info, progbar=True
    )
    assert info["converged"]
    assert Z == pytest.approx(Z_bp, rel=1e-1)


def test_get_gauged_tn():
    tn = qtn.TN2D_from_fill_fn(lambda s: qu.randn(s, dist="uniform"), 6, 6, 2)
    Z = tn.contract()
    bp = qbp.D1BP(tn)
    bp.run()
    Zbp = bp.contract()
    assert Z == pytest.approx(Zbp, rel=1e-1)
    tn_gauged = bp.get_gauged_tn()
    Zg = qu.prod(array.item(0) for array in tn_gauged.arrays)
    assert Z == pytest.approx(Zg, rel=1e-1)


def _get_gloop_bp():
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (2, 4),
        (3, 5),
    ]
    tn = qtn.TN_from_edges_rand(
        edges,
        D=2,
        seed=42,
        dist="uniform",
        loc=0.5,
    )
    tn.equalize_norms_(1.7)
    bp = qbp.D1BP(tn)
    bp.run(tol=1e-12)
    assert bp.converged
    return tn, bp


class TestGloopExpand:
    def test_get_normalized_tn_non_mutating(self):
        _, bp = _get_gloop_bp()
        bp.normalize_message_pairs()
        z_bp = bp.contract()
        arrays = {tid: t.data.copy() for tid, t in bp.tn.tensor_map.items()}
        messages = {key: m.copy() for key, m in bp.messages.items()}
        stored_scale = bp.sign, bp.exponent

        tn_norm, sign, exponent = bp.get_normalized_tn()

        assert tn_norm is not bp.tn
        assert tn_norm.exponent == 0.0
        assert (bp.sign, bp.exponent) == stored_scale
        for tid, data in arrays.items():
            assert np.array_equal(bp.tn.tensor_map[tid].data, data)
            zr = bp.get_cluster((tid,), tn=tn_norm).contract()
            assert zr == pytest.approx(1.0)
        for key, message in messages.items():
            assert np.array_equal(bp.messages[key], message)
        assert sign * 10**exponent == pytest.approx(z_bp)

    @pytest.mark.parametrize(
        "method", ["contract_gloop_expand", "contract_loop_series_expansion"]
    )
    def test_default_gloops_use_the_automatic_size(self, method):
        # the default `gloops=None` means generate them, not an empty supply
        _, bp = _get_gloop_bp()
        z_auto = getattr(bp, method)()
        _, bp = _get_gloop_bp()
        z_min = getattr(bp, method)(gloops="min")
        # the only loop here is the 4-cycle, so both sizes agree
        assert z_auto == pytest.approx(z_min)

    def test_singleton_completion_for_product_and_sum(self):
        _, bp = _get_gloop_bp()
        gloop = (0, 1, 2, 3)

        for combine in ("prod", "sum"):
            info = {}
            z = bp.contract_gloop_expand(
                gloops=(gloop,),
                autoreduce=False,
                combine=combine,
                info=info,
            )
            _, sign, exponent = bp.get_normalized_tn()
            contractions = info["contractions"]
            assert set(contractions) == {frozenset(gloop)}

            zloop = contractions[frozenset(gloop)]
            scale = sign * 10**exponent
            if combine == "prod":
                expected = scale * zloop
            else:
                expected = scale * (zloop + 2.0)
            assert z == pytest.approx(expected)

    def test_cached_dangling_regions_and_unit_regions(self):
        _, bp = _get_gloop_bp()
        gloops = (
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 5),
        )
        info = {}

        with mock.patch.object(
            bp, "get_cluster", wraps=bp.get_cluster
        ) as get_cluster:
            bp.contract_gloop_expand(gloops=gloops, info=info)
            assert get_cluster.call_count == 1
            bp.contract_gloop_expand(gloops=gloops, info=info)
            assert get_cluster.call_count == 1

        assert set(info["contractions"]) == {frozenset({0, 1, 2, 3})}
        assert "neighbors" in info

    def test_autoreduce_fixed_point_equivalence(self):
        _, bp = _get_gloop_bp()
        gloops = (
            (0, 1, 2, 3, 4),
            (0, 1, 2, 3, 5),
        )
        z_full = bp.contract_gloop_expand(gloops=gloops, autoreduce=False)
        z_reduced = bp.contract_gloop_expand(gloops=gloops, autoreduce=True)
        assert z_reduced == pytest.approx(z_full, rel=1e-11)

    @pytest.mark.parametrize("gloops", [(), 0])
    @pytest.mark.parametrize("autoreduce", [False, True])
    def test_no_gloops_matches_bp(self, gloops, autoreduce):
        _, bp = _get_gloop_bp()
        z_bp = bp.contract()
        info = {}
        z_gloop = bp.contract_gloop_expand(
            gloops=gloops,
            autoreduce=autoreduce,
            info=info,
        )
        assert z_gloop == pytest.approx(z_bp)
        assert not info["contractions"]

    @pytest.mark.parametrize("combine", ["prod", "sum"])
    def test_strip_exponent_and_progress(self, combine):
        _, bp = _get_gloop_bp()
        kwargs = {
            "gloops": ((0, 1, 2, 3),),
            "autoreduce": False,
            "combine": combine,
            "info": {},
        }

        with mock.patch("tqdm.tqdm", side_effect=lambda x: x) as tqdm:
            mantissa, exponent = bp.contract_gloop_expand(
                strip_exponent=True,
                progbar=True,
                **kwargs,
            )
        tqdm.assert_called_once()

        z = bp.contract_gloop_expand(strip_exponent=False, **kwargs)
        assert z == pytest.approx(mantissa * 10**exponent)
