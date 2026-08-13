import pickle

import numpy as np
import pytest
from pytest import approx

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tnag.tebd import trotter_schedule


def dense_prod(gates, sites, d=2):
    """The dense operator the ordered sequence of ``gates`` applies, with
    ``sites`` giving the order of the subsystems.
    """
    dims = [d] * len(sites)
    where_to_inds = {site: i for i, site in enumerate(sites)}
    U = qu.eye(d ** len(sites))
    for g in gates:
        inds = [where_to_inds[site] for site in g.where]
        U = qu.pkron(g.U, dims, inds) @ U
    return U


def dense_ham(ham, sites, d=2):
    """The dense hamiltonian ``ham`` represents, with ``sites`` giving the
    order of the subsystems.
    """
    dims = [d] * len(sites)
    where_to_inds = {site: i for i, site in enumerate(sites)}
    H = np.zeros((d ** len(sites),) * 2)
    for where, term in ham.items():
        inds = [where_to_inds[site] for site in where]
        H = H + qu.pkron(term, dims, inds)
    return H


class TestTrotterSchedule:
    def test_order1(self):
        assert trotter_schedule(3, order=1) == [
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
        ]

    def test_order2_is_palindromic(self):
        assert trotter_schedule(2, order=2) == [(0, 0.5), (1, 1.0), (0, 0.5)]
        sched = trotter_schedule(4, order=2)
        assert sched == list(reversed(sched))

    def test_single_layer(self):
        for order in (1, 2, 4):
            sched = trotter_schedule(1, order=order)
            assert all(k == 0 for k, _ in sched)
            assert sum(frac for _, frac in sched) == approx(1.0)

    @pytest.mark.parametrize("order", [1, 2, 4])
    @pytest.mark.parametrize("nlayers", [1, 2, 3, 5])
    def test_every_layer_gets_unit_total_fraction(self, nlayers, order):
        sched = trotter_schedule(nlayers, order=order)
        for k in range(nlayers):
            total = sum(frac for kk, frac in sched if kk == k)
            assert total == approx(1.0)

    def test_bad_order(self):
        with pytest.raises(ValueError):
            trotter_schedule(2, order=3)


class TestGetTrotterGates:
    def test_structure_obc_order2(self):
        H = qtn.LocalHam1D(L=6, H2=qu.ham_heis(2))
        gates = H.get_trotter_gates(-0.1, order=2, steps=2)

        # even-half, odd, even (fused), odd, even-half
        assert [(g.where, g.frac, g.layer, g.step) for g in gates] == [
            ((0, 1), 0.5, 0, 0),
            ((2, 3), 0.5, 0, 0),
            ((4, 5), 0.5, 0, 0),
            ((3, 4), 1.0, 1, 0),
            ((1, 2), 1.0, 1, 0),
            ((0, 1), 1.0, 2, 0),
            ((2, 3), 1.0, 2, 0),
            ((4, 5), 1.0, 2, 0),
            ((3, 4), 1.0, 3, 1),
            ((1, 2), 1.0, 3, 1),
            ((0, 1), 0.5, 4, 1),
            ((2, 3), 0.5, 4, 1),
            ((4, 5), 0.5, 4, 1),
        ]

    def test_fused_only_has_half_gates_at_the_endpoints(self):
        H = qtn.LocalHam1D(L=8, H2=qu.ham_heis(2))
        gates = H.get_trotter_gates(-0.1, order=2, steps=5)
        halves = [g.layer for g in gates if g.frac != 1.0]
        assert set(halves) == {0, gates[-1].layer}

        # unfused instead has them at every step boundary
        gates = H.get_trotter_gates(
            -0.1, order=2, steps=5, fuse_adjacent=False
        )
        assert len({g.layer for g in gates if g.frac != 1.0}) == 10

    def test_alternate_reverses_every_other_layer(self):
        H = qtn.LocalHam1D(L=6, H2=qu.ham_heis(2))
        wheres = [
            g.where for g in H.get_trotter_gates(-0.1, order=1, alternate=True)
        ]
        assert wheres == [(0, 1), (2, 3), (4, 5), (3, 4), (1, 2)]

        wheres = [
            g.where
            for g in H.get_trotter_gates(-0.1, order=1, alternate=False)
        ]
        assert wheres == [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4)]

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("L", [4, 5, 6])
    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_fusing_and_alternating_dont_change_the_propagator(
        self, L, cyclic, order
    ):
        H = qtn.LocalHam1D(L=L, H2=qu.ham_heis(2), cyclic=cyclic)
        sites = range(L)
        opts = {"order": order, "steps": 3}
        Ufused = dense_prod(H.get_trotter_gates(-0.13, **opts), sites)
        Uplain = dense_prod(
            H.get_trotter_gates(-0.13, fuse_adjacent=False, **opts), sites
        )
        Useq = dense_prod(
            H.get_trotter_gates(-0.13, alternate=False, **opts), sites
        )
        np.testing.assert_allclose(Ufused, Uplain, atol=1e-12)
        np.testing.assert_allclose(Ufused, Useq, atol=1e-12)

    @pytest.mark.parametrize("cyclic", [False, True])
    @pytest.mark.parametrize("L", [5, 6])
    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_error_scales_with_order(self, L, cyclic, order):
        H = qtn.LocalHam1D(L=L, H2=qu.ham_heis(2), cyclic=cyclic)
        Hd = qu.ham_heis(L, cyclic=cyclic, sparse=False)

        errs = []
        for x in (0.08, 0.04):
            gates = H.get_trotter_gates(-1j * x, order=order)
            U = dense_prod(gates, range(L))
            errs.append(np.linalg.norm(U - qu.expm(-1j * x * Hd)))

        # halving x should reduce the error by 2**(order + 1)
        assert errs[0] / errs[1] == approx(2 ** (order + 1), rel=0.1)

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_real_time_gates_are_unitary(self, order):
        H = qtn.LocalHam1D(L=5, H2=qu.ham_heis(2), cyclic=True)
        gates = H.get_trotter_gates(-1j * 0.1, order=order, steps=2)
        U = dense_prod(gates, range(5))
        np.testing.assert_allclose(U.conj().T @ U, qu.eye(2**5), atol=1e-12)

    def test_explicit_ordering(self):
        H = qtn.LocalHam1D(L=4, H2=qu.ham_heis(2))
        layers = [((1, 2),), ((0, 1), (2, 3))]
        gates = H.get_trotter_gates(-0.1, order=2, ordering=layers)
        assert [g.where for g in gates] == [
            (1, 2),
            (2, 3),
            (0, 1),
            (1, 2),
        ]
        assert [g.frac for g in gates] == [0.5, 1.0, 1.0, 0.5]

    def test_arbitrary_geometry(self):
        edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
        H2 = qu.ham_heis(2)
        ham = qtn.LocalHamGen(H2={e: H2 for e in edges})
        sites = range(5)
        Hd = dense_ham(ham, sites)

        errs = []
        for x in (0.08, 0.04):
            U = dense_prod(ham.get_trotter_gates(-1j * x, order=2), sites)
            errs.append(np.linalg.norm(U - qu.expm(-1j * x * Hd)))

        assert errs[0] / errs[1] == approx(8, rel=0.1)

    def test_no_terms(self):
        ham = qtn.LocalHamGen(H2={})
        assert ham.get_trotter_gates(-0.1) == ()

    def test_unpacks_as_gate_and_where(self):
        H = qtn.LocalHam1D(L=4, H2=qu.ham_heis(2))
        (g,) = H.get_trotter_gates(-0.1, order=1, ordering=[((0, 1),)])
        U, where = g
        assert where == (0, 1)
        np.testing.assert_allclose(U, g.U)
        assert (g.frac, g.layer, g.step) == (1.0, 0, 0)

        # the attributes should survive a round trip
        g2 = pickle.loads(pickle.dumps(g))
        assert (g2.where, g2.frac, g2.layer, g2.step) == (
            (0, 1),
            1.0,
            0,
            0,
        )
