import pytest

import quimb.tensor as qtn
import quimb.tensor.belief_propagation as qbp


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
