import pytest

import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp
from quimb.tensor.belief_propagation.bp_common import (
    create_lazy_community_edge_map,
)


class TestLazyCommunityEdgeMap:
    @pytest.mark.parametrize("include_reverse", [False, True])
    def test_touch_map_chain(self, include_reverse):
        tn = qtn.TN_from_edges_rand(qtn.edges_1d_chain(3), 2, seed=42)
        *_, touch_map = create_lazy_community_edge_map(
            tn, include_reverse=include_reverse
        )
        expected = {("I1", "I2")}
        if include_reverse:
            expected.add(("I1", "I0"))
        assert set(touch_map["I0", "I1"]) == expected
        # the end site has no other neighbors
        expected = {("I0", "I1")} if include_reverse else set()
        assert set(touch_map["I1", "I0"]) == expected

    @pytest.mark.parametrize("include_reverse", [False, True])
    def test_touch_map_loop(self, include_reverse):
        edges = qtn.edges_1d_chain(4, cyclic=True)
        tn = qtn.TN_from_edges_rand(edges, 2, seed=42)
        *_, touch_map = create_lazy_community_edge_map(
            tn, include_reverse=include_reverse
        )
        assert len(touch_map) == 8
        for (i, j), touched in touch_map.items():
            assert all(a == j for a, _ in touched)
            assert ((j, i) in touched) == include_reverse
            assert len(touched) == (2 if include_reverse else 1)

    @pytest.mark.parametrize("include_reverse", [False, True])
    def test_parallel_bonds(self, include_reverse):
        # two columns as sites, joined by one bond per row
        tn = qtn.TN2D_rand(3, 2, 2, seed=42)
        edges, neighbors, _, touch_map = create_lazy_community_edge_map(
            tn, site_tags=["Y0", "Y1"], include_reverse=include_reverse
        )
        assert len(edges["Y0", "Y1"]) == 3
        assert neighbors == {"Y0": ["Y1"], "Y1": ["Y0"]}
        expected = (("Y1", "Y0"),) if include_reverse else ()
        assert touch_map["Y0", "Y1"] == expected


class TestSweepOrder:
    @pytest.mark.parametrize("cls", ["D1BP", "D2BP", "L1BP", "L2BP"])
    def test_tree_converges_in_two_sweeps(self, cls):
        # first site is in the middle of a chain with shuffled tensor order
        edges = [
            *((i, i + 1) for i in range(5)),
            (0, 6),
            *((i, i + 1) for i in range(6, 10)),
        ]
        phys_dim = 2 if cls in ("D2BP", "L2BP") else None
        tn = qtn.TN_from_edges_rand(
            edges, 3, phys_dim=phys_dim, dist="normal", loc=1.0, seed=42
        )
        order = [3, 8, 0, 5, 10, 1, 7, 4, 9, 2, 6]
        tn = qtn.TensorNetwork([tn[i] for i in order]).view_like_(tn)
        bp = getattr(qbp, cls)(tn)
        assert bp.sweeper.alternate
        info = {}
        bp.run(max_iterations=100, tol=1e-10, info=info)
        assert info["converged"]
        # two sweeps give exact messages with no pending updates
        assert info["iterations"] == 2

    @pytest.mark.parametrize("cls", ["D1BP", "D2BP", "L1BP", "L2BP"])
    def test_loops_use_fixed_order(self, cls):
        edges = qtn.edges_1d_chain(6, cyclic=True)
        phys_dim = 2 if cls in ("D2BP", "L2BP") else None
        tn = qtn.TN_from_edges_rand(
            edges, 2, phys_dim=phys_dim, dist="normal", loc=1.0, seed=42
        )
        bp = getattr(qbp, cls)(tn)
        assert not bp.sweeper.alternate
        with pytest.raises(ValueError):
            getattr(qbp, cls)(tn, sweep_order="bfs")

    def test_local_restart_crosses_tree_in_one_sweep(self):
        edges = qtn.edges_1d_chain(12)
        psi = qtn.TN_from_edges_rand(
            edges, 3, phys_dim=2, dist="normal", loc=1.0, seed=42
        )
        bp = qbp.D2BP(psi)
        bp.run(max_iterations=100, tol=1e-12)
        # project a middle site, then restart from its messages
        psi.isel_({psi.site_ind(6): 0})
        bp_local = qbp.D2BP(psi, messages=dict(bp.messages))
        bp_local.update_touched_from_tids(*psi._get_tids_from_tags("I6"))
        info = {}
        bp_local.run(max_iterations=100, tol=1e-10, info=info)
        assert info["converged"]
        assert info["iterations"] == 1
        bp_full = qbp.D2BP(psi)
        bp_full.run(max_iterations=100, tol=1e-12)
        assert bp_local.contract() == pytest.approx(bp_full.contract())
