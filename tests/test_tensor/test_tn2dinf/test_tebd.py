import numpy as np
import pytest

import quimb as qu
from quimb.tensor.tn2dinf.core import (
    PEPSInfinite2D,
)
from quimb.tensor.tn2dinf.geometry import GeometryInfinite2D
from quimb.tensor.tn2dinf.tebd import (
    LocalHamInfinite2D,
    SimpleUpdateInfinite2D,
)

geom_square_2x2 = GeometryInfinite2D.square()
geom_square_2x2_nnn = GeometryInfinite2D.square(couplings=2)


class TestLocalHamInfinite2D:
    def test_uniform_two_site_terms(self):
        geom = geom_square_2x2
        H2 = qu.ham_heis(2)
        ham = LocalHamInfinite2D(geom, H2)
        assert set(ham.terms) == set(geom.bond_types)
        for bt in geom.bond_types:
            assert ham.terms[bt] == pytest.approx(H2)

    def test_get_gate_order_invariant(self):
        geom = geom_square_2x2
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        a, b = geom.bond_types[0]
        assert ham.get_gate((a, b)) is ham.get_gate((b, a))

    def test_get_gate_expm_cached(self):
        geom = geom_square_2x2
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        bt = geom.bond_types[0]
        assert ham.get_gate_expm(bt, -0.1) is ham.get_gate_expm(bt, -0.1)
        assert ham.get_gate_expm(bt, -0.1).shape == (4, 4)

    def test_one_site_term_absorbed_evenly(self):
        geom = geom_square_2x2
        H2 = qu.ham_heis(2)
        Z = qu.pauli("Z")
        ham = LocalHamInfinite2D(geom, H2, H1=Z)
        eye2 = np.eye(2)
        bt = geom.bond_types[0]
        (_, sta), (_, stb) = bt
        na = len(ham._site_type_to_covering[sta])
        nb = len(ham._site_type_to_covering[stb])
        expected = H2 + qu.kron(Z, eye2) / na + qu.kron(eye2, Z) / nb
        assert ham.terms[bt] == pytest.approx(expected)

    def test_missing_two_site_term_raises(self):
        geom = geom_square_2x2
        partial = {
            geom.bond_types[0]: qu.ham_heis(2)
        }  # no default, incomplete
        with pytest.raises(ValueError):
            LocalHamInfinite2D(geom, partial)

    def test_get_auto_ordering_commuting_layers(self):
        geom = geom_square_2x2
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        ordering = ham.get_auto_ordering(group=True)
        flat = [bt for layer in ordering for bt in layer]
        assert set(flat) == set(geom.bond_types)
        for layer in ordering:
            seen = set()
            for (_, sta), (_, stb) in layer:
                assert sta not in seen and stb not in seen
                seen.update((sta, stb))

    def test_different_geometry_same_site_types(self):
        H2 = qu.ham_heis(2)
        ham_nn = LocalHamInfinite2D(geom_square_2x2, H2)
        ham_nnn = LocalHamInfinite2D(geom_square_2x2_nnn, H2)
        assert ham_nn.site_types == ham_nnn.site_types
        assert len(ham_nnn.bond_types) > len(ham_nn.bond_types)

    def test_nsites_and_repr(self):
        geom = geom_square_2x2
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        assert ham.nsites == 4
        assert "LocalHamInfinite2D" in repr(ham)


class TestSimpleUpdateInfinite2D:
    def _setup(self, bond_dim=2, seed=0):
        geom = geom_square_2x2
        rng = np.random.default_rng(seed)
        psi = PEPSInfinite2D.from_fill_fn(
            lambda s: rng.standard_normal(s), geom, bond_dim=bond_dim
        )
        ham = LocalHamInfinite2D(geom, qu.ham_heis(2))
        return psi, ham

    def test_construct_initializes_gauges_and_energy(self):
        psi, ham = self._setup()
        su = SimpleUpdateInfinite2D(psi, ham, D=4, progbar=False)
        assert su.gauges  # gauges initialized + equilibrated at construction
        assert np.isfinite(su.energy)  # per-site energy is computable

    def test_evolve_lowers_energy(self):
        psi, ham = self._setup()
        su = SimpleUpdateInfinite2D(psi, ham, D=4, progbar=False)
        e0 = su.energy
        su.evolve(30, tau=0.3)
        assert su.energies[-1] < e0

    def test_get_state_return_splits_gauges(self):
        psi, ham = self._setup()
        su = SimpleUpdateInfinite2D(psi, ham, D=2, progbar=False)
        out, gauges = su.get_state(absorb_gauges="return")
        assert isinstance(out, PEPSInfinite2D)
        assert isinstance(gauges, dict) and gauges

    def test_reaches_heisenberg_ground_state(self):
        # end-to-end via the driver, QMC reference e/site ~ -0.6694
        psi, ham = self._setup()
        su = SimpleUpdateInfinite2D(psi, ham, D=4, progbar=False)
        for tau in (0.3, 0.1, 0.03):
            su.evolve(100, tau=tau)
        assert -0.68 < su.energies[-1] < -0.60

    def test_compute_local_expectation_defaults_to_ham_and_gauges(self):
        psi, ham = self._setup()
        su = SimpleUpdateInfinite2D(psi, ham, D=4, progbar=False)
        # bare call: terms -> ham, gauges -> current; per-cell energy
        e_cell = su.compute_local_expectation_cluster()
        assert np.isfinite(e_cell)
        # matches the driver's per-site energy * nsites
        assert e_cell == pytest.approx(su.energy * ham.nsites)
        # explicit terms/gauges give the same
        e_explicit = su.compute_local_expectation_cluster(
            ham, gauges=su.gauges
        )
        assert e_cell == pytest.approx(e_explicit)
        # return_all / kwargs forward through
        per = su.compute_local_expectation_cluster(return_all=True)
        assert set(per) == set(ham.bond_types)

    def test_compute_local_expectation_max_distance_and_gloops(self):
        psi, ham = self._setup()
        su = SimpleUpdateInfinite2D(psi, ham, D=4, progbar=False)
        su.evolve(20, tau=0.3)
        su.equilibrate()
        e0 = su.compute_local_expectation_cluster()
        e1 = su.compute_local_expectation_cluster(max_distance=1)
        eg = su.compute_local_expectation_gloop_expand(gloops=4)
        for e in (e0, e1, eg):
            assert np.isfinite(e) and abs(np.imag(e)) < 1e-10
