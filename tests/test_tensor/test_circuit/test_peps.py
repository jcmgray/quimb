import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


class TestCircuitPEPSSimpleUpdate:
    def test_requires_geometry(self):
        # none of edges, gates or psi0 given -> cannot define the geometry
        with pytest.raises(ValueError):
            qtn.CircuitPEPSSimpleUpdate()

    def test_geometry_inferred_from_gates(self):
        # passing the two-qubit gates instead of edges should reconstruct the
        # same geometry, and give the same state as the explicit-edges path
        edges = [(0, 1), (1, 2), (0, 2)]
        rng = np.random.default_rng(5)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(3)
        ]
        gates += [qtn.Gate("CZ", params=(), qubits=list(e)) for e in edges]

        circ_edges = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ_gates = qtn.CircuitPEPSSimpleUpdate(gates=gates, max_bond=16)
        # geometry matches (as undirected edges) and the gates are not applied
        assert {frozenset(e) for e in circ_gates.edges} == {
            frozenset(e) for e in circ_edges.edges
        }
        assert circ_gates.num_gates == 0

        circ_edges.apply_gates(gates)
        circ_gates.apply_gates(gates)
        Z = qu.pauli("Z").astype(complex)
        for i in range(3):
            xe = circ_edges.local_expectation(Z, i, max_distance=3)
            xg = circ_gates.local_expectation(Z, i, max_distance=3)
            assert float(np.real(xg)) == pytest.approx(
                float(np.real(xe)), abs=1e-10
            )

    def test_geometry_inferred_from_psi0(self):
        # build a state, hand it back as psi0, and check the geometry is read
        # from its bonds and expectations are preserved
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        rng = np.random.default_rng(11)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(4)
        ]
        gates += [qtn.Gate("CZ", params=(), qubits=list(e)) for e in edges]

        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ.apply_gates(gates)
        psi = circ.psi

        circ2 = qtn.CircuitPEPSSimpleUpdate(psi0=psi, max_bond=16)
        assert {frozenset(e) for e in circ2.edges} == {
            frozenset(e) for e in edges
        }
        Z = qu.pauli("Z").astype(complex)
        for i in range(4):
            x1 = circ.local_expectation(Z, i, max_distance=4)
            x2 = circ2.local_expectation(Z, i, max_distance=4)
            assert float(np.real(x2)) == pytest.approx(
                float(np.real(x1)), abs=1e-6
            )

    def test_construction_and_initial_state(self):
        # 2x3 grid of integer-labelled sites
        edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        # one tensor per site, bonds only on the edges, all bond dim 1
        assert circ.N == 6
        assert circ._psi.max_bond() == 1
        assert circ.psi.norm() == pytest.approx(1.0)

    def test_target_api(self):
        # the exact usage requested in the issue, on a 3x3 grid of integer sites
        ncol = 3
        edges = []
        for r in range(3):
            for c in range(3):
                s = r * ncol + c
                if c + 1 < 3:
                    edges.append((s, s + 1))
                if r + 1 < 3:
                    edges.append((s, s + ncol))
        rng = np.random.default_rng(42)
        gates = [
            qtn.Gate(
                "U3", params=rng.uniform(0, 2 * np.pi, size=3), qubits=[i]
            )
            for i in range(9)
        ]
        gates += [
            qtn.Gate("CZ", params=(), qubits=list(edge)) for edge in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(gates)
        peps = circ.psi
        assert peps.max_bond() <= 8
        assert peps.num_tensors == 9

    def test_matches_exact_on_a_chain(self):
        # on a chain, simple update at large bond is exact, so local
        # expectations should match a dense Circuit to high precision
        N = 5
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(0)

        gates = []
        for i in range(N):
            gates.append(
                qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            )
        for i, j in edges:
            gates.append(qtn.Gate("CNOT", params=(), qubits=[i, j]))
            gates.append(
                qtn.Gate("RZ", params=[rng.uniform(0, np.pi)], qubits=[j])
            )
            gates.append(qtn.Gate("CNOT", params=(), qubits=[i, j]))

        circ_su = qtn.CircuitPEPSSimpleUpdate(
            edges=edges, max_bond=32, cutoff=1e-12
        )
        circ_su.apply_gates(gates)
        circ_su.equilibrate()

        circ_exact = qtn.Circuit(N=N)
        circ_exact.apply_gates(gates)

        Z = qu.pauli("Z").astype(complex)
        for i in range(N):
            x_su = circ_su.local_expectation(Z, i, max_distance=N)
            x_ex = circ_exact.local_expectation(Z, i)
            assert float(np.real(x_su)) == pytest.approx(
                float(np.real(x_ex)), abs=1e-8
            )

    def test_equilibrate_preserves_expectations(self):
        N = 4
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(7)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(N)
        ]
        for i, j in edges:
            gates.append(qtn.Gate("CNOT", params=(), qubits=[i, j]))

        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=32)
        circ.apply_gates(gates)

        Z = qu.pauli("Z").astype(complex)
        before = [
            float(np.real(circ.local_expectation(Z, i, max_distance=N)))
            for i in range(N)
        ]
        circ.equilibrate()
        after = [
            float(np.real(circ.local_expectation(Z, i, max_distance=N)))
            for i in range(N)
        ]
        for b, a in zip(before, after):
            assert b == pytest.approx(a, abs=1e-8)

    def test_matches_exact_on_2x2_plaquette(self):
        # a 2x2 grid has a 4-cycle, so simple update is genuinely approximate,
        # but a shallow circuit still matches exact closely. checks both
        # local_expectation and the gauge-absorbed psi.
        edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
        rng = np.random.default_rng(3)
        gates = [
            qtn.Gate("RY", params=[rng.uniform(0, np.pi)], qubits=[i])
            for i in range(4)
        ]
        for i, j in edges:
            gates.append(qtn.Gate("CZ", params=(), qubits=[i, j]))

        circ = qtn.CircuitPEPSSimpleUpdate(
            edges=edges, max_bond=16, cutoff=1e-12
        )
        circ.apply_gates(gates)

        circ_exact = qtn.Circuit(N=4)
        circ_exact.apply_gates(gates)

        Z = qu.pauli("Z").astype(complex)
        # gauge-absorbed psi should be a normalized state
        psi = circ.psi
        assert ((psi.H & psi) ^ all) == pytest.approx(1.0, abs=1e-6)

        for i in range(4):
            x_su = circ.local_expectation(Z, i, max_distance=4)
            # measure the same way directly on the returned psi
            x_psi = psi.compute_local_expectation_cluster(
                {(i,): Z}, max_distance=4, normalized=True
            )
            x_ex = circ_exact.local_expectation(Z, i)
            assert float(np.real(x_su)) == pytest.approx(
                float(np.real(x_ex)), abs=1e-6
            )
            assert float(np.real(x_psi)) == pytest.approx(
                float(np.real(x_ex)), abs=1e-6
            )

    def test_psi_access_does_not_mutate_gauges(self):
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(
            [
                qtn.Gate("H", params=(), qubits=[0]),
                qtn.Gate("CNOT", params=(), qubits=[0, 1]),
                qtn.Gate("CNOT", params=(), qubits=[1, 2]),
            ]
        )
        before = {k: np.asarray(v).copy() for k, v in circ.gauges.items()}
        _ = circ.psi
        _ = circ.psi
        assert set(circ.gauges) == set(before)
        for k in before:
            assert_allclose(np.asarray(circ.gauges[k]), before[k])

    def test_controlled_and_exact_methods_raise(self):
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        # controlled gates can't be expressed by the simple update rule
        with pytest.raises(ValueError):
            circ.apply_gates(
                [qtn.Gate("X", params=(), qubits=[2], controls=[0])]
            )
        # exact-state methods are not meaningful for a gauged approximate state
        circ.apply_gates([qtn.Gate("H", params=(), qubits=[0])])
        with pytest.raises(NotImplementedError):
            circ.amplitude("000")
        with pytest.raises(NotImplementedError):
            list(circ.sample(2))
        with pytest.raises(NotImplementedError):
            circ.partial_trace([0])
        with pytest.raises(NotImplementedError):
            circ.sample_chaotic_rehearse()
        with pytest.raises(NotImplementedError):
            circ.uni
        # a two-qubit gate off any declared edge is rejected
        with pytest.raises(ValueError):
            circ.apply_gates([qtn.Gate("CZ", params=(), qubits=[0, 2])])

    def test_to_dense_matches_exact_on_a_chain(self):
        # on a chain at large bond simple update is exact, so the gauged dense
        # contraction should match a dense Circuit up to a global phase
        N = 5
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(2)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), i)
            for i in range(N)
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(
            edges=edges, max_bond=32, cutoff=1e-12
        )
        circ.apply_gates(gates)

        exact = qtn.Circuit(N=N)
        exact.apply_gates(gates)

        k_su = circ.to_dense()
        k_ex = exact.to_dense()
        # column vector of the full state, matching Circuit.to_dense
        assert k_su.shape == k_ex.shape == (2**N, 1)
        assert qu.fidelity(k_su, k_ex) == pytest.approx(1.0, abs=1e-8)

    def test_get_state_absorb_gauges(self):
        # the three absorb_gauges modes should all describe the same state
        edges = [(0, 1), (1, 2), (0, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(
            [
                qtn.Gate("H", params=(), qubits=[0]),
                qtn.Gate("CNOT", params=(), qubits=[0, 1]),
                qtn.Gate("CZ", params=(), qubits=[1, 2]),
            ]
        )
        # absorbed network is exactly the `psi` property
        psi_absorbed = circ.get_state(absorb_gauges=True)
        assert_allclose(psi_absorbed.to_dense(), circ.psi.to_dense())
        # gauges added but uncontracted should give the same dense state
        psi_uncontracted = circ.get_state(absorb_gauges=False)
        assert qu.fidelity(
            psi_uncontracted.to_dense(), psi_absorbed.to_dense()
        ) == pytest.approx(1.0, abs=1e-10)
        # "return" hands back the raw network plus a copy of the gauges
        raw, gauges = circ.get_state(absorb_gauges="return")
        assert set(gauges) == set(circ.gauges)
        assert gauges is not circ.gauges
        raw.gauge_simple_insert(gauges)
        assert qu.fidelity(
            raw.to_dense(), psi_absorbed.to_dense()
        ) == pytest.approx(1.0, abs=1e-10)

    def test_local_expectation_on_grid_coordinate_sites(self):
        # sites are 2D coordinate tuples; a single-site `where` must not be
        # misread as a two-site operator (regression for the site handling)
        edges = qtn.edges_2d_square(2, 2)
        sites = sorted({s for e in edges for s in e})
        rng = np.random.default_rng(0)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ.apply_gates(gates)
        circ.equilibrate(max_iterations=300, tol=1e-12)

        qmap = {s: i for i, s in enumerate(sites)}
        ref = qtn.Circuit(N=len(sites))
        for G, *where in gates:
            ref.apply_gate(G, *(qmap[s] for s in where))

        Z = qu.pauli("Z").astype(complex)
        for s in sites:
            x = circ.local_expectation(Z, s, max_distance=2)
            xe = complex(ref.local_expectation(Z, qmap[s]))
            assert complex(x) == pytest.approx(xe, abs=1e-6)

    def test_local_expectation_accepts_list_where(self):
        # a multi-site `where` given as a list must not crash on the set
        # membership check, and should match the tuple form
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(
            [
                qtn.Gate("H", params=(), qubits=[0]),
                qtn.Gate("CNOT", params=(), qubits=[0, 1]),
            ]
        )
        Z = qu.pauli("Z").astype(complex)
        ZZ = qu.kron(Z, Z)
        x_list = circ.local_expectation(ZZ, [0, 1], max_distance=2)
        x_tuple = circ.local_expectation(ZZ, (0, 1), max_distance=2)
        assert complex(x_list) == pytest.approx(complex(x_tuple))

    def test_copy_is_independent(self):
        edges = qtn.edges_2d_square(2, 2)
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=8)
        rng = np.random.default_rng(3)
        circ.apply_gates(
            [
                (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
                for a, b in edges
            ]
        )
        other = circ.copy()
        n_before = circ.num_gates
        # the copy carries the geometry and gauges and is independently usable
        assert set(other.edges) == set(circ.edges)
        assert other.gauges is not circ.gauges
        assert len(other.gauges) == len(circ.gauges)
        a, b = circ.edges[0]
        other.apply_gates([(qu.rand_uni(4), a, b)])
        # mutating the copy must not touch the original
        assert circ.num_gates == n_before
        assert other.num_gates == n_before + 1

    def test_renorm_and_equilibrate_every(self):
        # the renorm and equilibrate_every options should be accepted and keep
        # normalized expectations correct on a chain (where SU is exact)
        N = 5
        edges = qtn.edges_1d_chain(N, cyclic=False)
        rng = np.random.default_rng(1)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), i)
            for i in range(N)
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ = qtn.CircuitPEPSSimpleUpdate(
            edges=edges,
            max_bond=32,
            cutoff=1e-12,
            renorm=True,
            equilibrate_every=3,
        )
        circ.apply_gates(gates)
        circ.equilibrate()

        exact = qtn.Circuit(N=N)
        exact.apply_gates(gates)
        Z = qu.pauli("Z").astype(complex)
        for i in range(N):
            x = circ.local_expectation(Z, i, max_distance=N)
            xe = complex(exact.local_expectation(Z, i))
            assert complex(x) == pytest.approx(xe, abs=1e-8)

    def test_cluster_error_decreases_with_distance(self):
        # on a loopy lattice with an essentially exact (large max_bond) state,
        # a larger cluster must give a more accurate local expectation
        edges = qtn.edges_2d_square(3, 3)
        sites = sorted({s for e in edges for s in e})
        rng = np.random.default_rng(11)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        qmap = {s: i for i, s in enumerate(sites)}
        ref = qtn.Circuit(N=len(sites))
        for G, *where in gates:
            ref.apply_gate(G, *(qmap[s] for s in where))
        Z = qu.pauli("Z").astype(complex)
        exact = {s: complex(ref.local_expectation(Z, qmap[s])) for s in sites}

        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=16)
        circ.apply_gates(gates)
        circ.equilibrate(max_iterations=200, tol=1e-10)

        def err(md):
            return max(
                abs(circ.local_expectation(Z, s, max_distance=md) - exact[s])
                for s in sites
            )

        assert err(3) < err(0)

    def test_scales_to_a_larger_lattice(self):
        # the point of a PEPS simulator: run a circuit on a lattice far too
        # large to simulate exactly, with the bond dimension capped
        edges = qtn.edges_2d_square(6, 6)
        max_bond = 8
        circ = qtn.CircuitPEPSSimpleUpdate(edges=edges, max_bond=max_bond)
        sites = sorted({s for e in edges for s in e})
        rng = np.random.default_rng(2)
        gates = [
            (qu.rand_uni(2, seed=int(rng.integers(1 << 30))), s) for s in sites
        ]
        gates += [
            (qu.rand_uni(4, seed=int(rng.integers(1 << 30))), a, b)
            for a, b in edges
        ]
        circ.apply_gates(gates)
        psi = circ.psi
        assert psi.num_tensors == len(sites)
        assert psi.max_bond() <= max_bond
