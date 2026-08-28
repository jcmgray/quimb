from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sps

import quimb as qu
import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp
from quimb.tensor.belief_propagation.sparse_ops import (
    compute_all_tensor_messages_coo,
    contract_tensor_messages_coo,
    parse_coo,
)


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


def _zero_out_half(tn, seed=42):
    """Zero out half the entries of each tensor, always keeping the largest,
    so that no tensor vanishes entirely.
    """
    rng = np.random.default_rng(seed)
    for t in tn:
        data = t.data.copy()
        keep = np.argmax(np.abs(data))
        data[rng.random(data.shape) < 0.5] = 0.0
        data.flat[keep] = t.data.flat[keep]
        t.modify(data=data)


class TestLoopSeriesExpansion:
    # the six-site ring has one bond across the middle
    # every pair of loops shares a site, so the connected series is exact
    theta_edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3))

    def get_tn_and_bp(self):
        tn = qtn.TN_from_edges_rand(
            self.theta_edges, D=2, seed=42, dist="uniform", loc=0.5
        )
        bp = qbp.D1BP(tn)
        bp.run(tol=1e-13)
        assert bp.converged
        return tn, bp

    def test_full_series_is_exact(self):
        tn, bp = self.get_tn_and_bp()
        z = bp.contract_loop_series_expansion(
            gloops=6, multi_excitation_correct=False
        )
        assert z == pytest.approx(tn.contract(output_inds=()), rel=1e-12)

    def test_explicit_tids_are_expanded_into_every_loop(self):
        tn, _ = self.get_tn_and_bp()
        gloops = tuple(tn.gen_gloops(6))
        _, bp = self.get_tn_and_bp()
        z_tids = bp.contract_loop_series_expansion(gloops=gloops)
        _, bp = self.get_tn_and_bp()
        z_auto = bp.contract_loop_series_expansion(gloops=6)
        assert z_tids == pytest.approx(z_auto, rel=1e-12)

    def test_explicit_patches_are_used_directly(self):
        _, bp = self.get_tn_and_bp()
        # bond names are network-specific
        patches = tuple(bp.tn.gen_gloops_edge_induced(6))
        z_patches = bp.contract_loop_series_expansion(gloops=patches)
        _, bp = self.get_tn_and_bp()
        z_auto = bp.contract_loop_series_expansion(gloops=6)
        assert z_patches == pytest.approx(z_auto, rel=1e-12)


class TestSparse:
    @pytest.mark.parametrize("shape", [(5,), (4, 6), (3, 4, 5), (2, 3, 4, 3)])
    @pytest.mark.parametrize("dtype", ["float64", "complex128"])
    def test_kernels_match_einsum(self, shape, dtype):
        rng = np.random.default_rng(42)
        x = rng.normal(size=shape)
        ms = [rng.normal(size=d) for d in shape]
        if dtype == "complex128":
            x = x + 1j * rng.normal(size=shape)
            ms = [m + 1j * rng.normal(size=m.size) for m in ms]
        x[rng.random(shape) < 0.5] = 0.0
        coo = parse_coo(sps.coo_array(x))

        syms = "abcdefg"[: len(shape)]
        new_ms = compute_all_tensor_messages_coo(coo, ms)
        for k in range(len(shape)):
            js = [j for j in range(len(shape)) if j != k]
            eq = syms + "".join("," + syms[j] for j in js) + "->" + syms[k]
            expected = np.einsum(eq, x, *(ms[j] for j in js))
            assert new_ms[k] == pytest.approx(expected)

        eq = syms + "".join("," + s for s in syms) + "->"
        z = contract_tensor_messages_coo(coo, ms)
        assert z == pytest.approx(np.einsum(eq, x, *ms))

    def test_messages_promote_dtype(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(3, 4))
        x[rng.random(x.shape) < 0.5] = 0.0
        ms = [rng.normal(size=d) + 1j * rng.normal(size=d) for d in x.shape]

        new_ms = compute_all_tensor_messages_coo(
            parse_coo(sps.coo_array(x)), ms
        )
        assert new_ms[0] == pytest.approx(x @ ms[1])
        assert new_ms[1] == pytest.approx(ms[0] @ x)

    def test_duplicate_coordinates_are_summed(self):
        data = np.array([1.5, 2.5, -1.0])
        coords = (np.array([0, 0, 1]), np.array([1, 1, 0]))
        x = sps.coo_array((data, coords), shape=(2, 2))
        assert not x.has_canonical_format
        dense = np.zeros((2, 2))
        np.add.at(dense, coords, data)
        ms = [np.array([0.3, -0.7]), np.array([1.1, 0.5])]

        new_ms = compute_all_tensor_messages_coo(parse_coo(x), ms)
        assert new_ms[0] == pytest.approx(dense @ ms[1])
        assert new_ms[1] == pytest.approx(ms[0] @ dense)

    def test_parse_coo_only_accepts_coo(self):
        x = np.eye(3)
        assert parse_coo(x) is None
        assert parse_coo(sps.csr_array(x)) is None
        coords, data = parse_coo(sps.coo_array(x))
        assert len(coords) == 2
        assert data == pytest.approx(np.ones(3))

    @pytest.mark.parametrize("dtype", ["float64", "complex128"])
    def test_tree_contraction_is_exact(self, dtype):
        tn = qtn.TN_rand_tree(20, 3, seed=42, dtype=dtype)
        _zero_out_half(tn)
        Z = tn.contract()
        assert Z != 0.0

        for t in tn:
            t.modify(data=sps.coo_array(t.data))
        assert tn.backend == "scipy"

        info = {}
        Z_bp = qbp.contract_d1bp(tn, info=info)
        assert info["converged"]
        assert Z_bp == pytest.approx(Z, rel=1e-10)

    def test_matches_dense_messages(self):
        tn = qtn.TN2D_from_fill_fn(
            lambda s: qu.randn(s, dist="uniform"), 6, 6, 2
        )
        _zero_out_half(tn)

        tn_sparse = tn.copy()
        for t in tn_sparse:
            t.modify(data=sps.coo_array(t.data))

        bp = qbp.D1BP(tn)
        bp_sparse = qbp.D1BP(tn_sparse)
        # messages are dense vectors even though the tensors are not
        assert bp_sparse.backend == "numpy"
        for key, m in bp.messages.items():
            assert bp_sparse.messages[key] == pytest.approx(m)

        bp.run(tol=1e-12)
        bp_sparse.run(tol=1e-12)
        assert bp.converged
        for key, m in bp.messages.items():
            assert bp_sparse.messages[key] == pytest.approx(m, abs=1e-10)
        assert bp_sparse.contract() == pytest.approx(bp.contract(), rel=1e-8)

    def test_mixed_dense_and_sparse_tensors(self):
        tn = qtn.TN_rand_tree(10, 3, seed=42)
        Z = tn.contract()

        for i, t in enumerate(tn):
            if i % 2:
                t.modify(data=sps.coo_array(t.data))

        Z_bp = qbp.contract_d1bp(tn)
        assert Z_bp == pytest.approx(Z, rel=1e-10)
