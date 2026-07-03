import numpy as np
import pytest

import quimb as qu
import quimb.tensor as qtn


class TestCircuitPEPOSimpleUpdate:
    @staticmethod
    def _mat(a):
        # a gate/observable array may be stored as a (2, 2, ..., 2) tensor;
        # collapse it to a square matrix for the dense reference
        A = np.asarray(a, dtype=complex)
        d = int(round(A.size**0.5))
        return A.reshape(d, d)

    @classmethod
    def _exact(cls, N, obs_site, obs, gates):
        U = np.eye(2**N, dtype=complex)
        for g in gates:
            U = qu.pkron(cls._mat(g.array), [2] * N, list(g.qubits)) @ U
        Of = qu.pkron(cls._mat(obs), [2] * N, [obs_site])
        p = qu.basis_vec(0, 2**N)
        return complex((p.conj().T @ (U.conj().T @ Of @ U) @ p)[0, 0])

    @staticmethod
    def _chain_gates(N, depth, seed):
        rng = np.random.default_rng(seed)
        gates = []
        for layer in range(depth):
            for i in range(N):
                gates.append(
                    qtn.Gate(
                        "U3", params=rng.uniform(0, 2 * np.pi, 3), qubits=[i]
                    )
                )
            for i in range(layer % 2, N - 1, 2):
                # SU4 gate arrays are stored as (2, 2, 2, 2) tensors, which
                # exercises the matrix-reshape daggering during evolution
                gates.append(
                    qtn.Gate(
                        "SU4",
                        params=rng.uniform(0, 2 * np.pi, 15),
                        qubits=[i, i + 1],
                    )
                )
        return gates

    def test_is_circuit_subclass(self):
        circ = qtn.CircuitPEPOSimpleUpdate(edges=[(0, 1), (1, 2)])
        # non-exact simulators compose `CircuitBase`, not the exact `Circuit`
        assert isinstance(circ, qtn.circuit.CircuitBase)
        assert not isinstance(circ, qtn.Circuit)
        # the shared gate-specification API works
        circ.h(0)
        circ.cx(0, 1)
        circ.apply_gate("CX", 1, 2)
        assert circ.num_gates == 3
        assert [g.label for g in circ.gates] == ["H", "CX", "CX"]

    @pytest.mark.parametrize("obs", ["X", "Y", "Z"])
    def test_matches_dense_chain_su4(self, obs):
        N = 4
        edges = [(i, i + 1) for i in range(N - 1)]
        gates = self._chain_gates(N, depth=3, seed=42)

        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=2**N)
        circ.apply_gates(gates)

        v = circ.local_expectation(qu.pauli(obs), 1)
        r = self._exact(N, 1, qu.pauli(obs), gates)
        assert complex(v) == pytest.approx(r, abs=1e-7)

    def test_su4_gate_dagger(self):
        # SU4 gate arrays are (2, 2, 2, 2) tensors; a plain reverse-axes
        # transpose gives the wrong dagger. A single SU4 sandwiched around a
        # local observable must match the exact value.
        edges = [(0, 1)]
        rng = np.random.default_rng(0)
        gate = qtn.Gate(
            "SU4", params=rng.uniform(0, 2 * np.pi, 15), qubits=[0, 1]
        )
        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=4)
        circ.apply_gate(gate)

        v = circ.local_expectation(qu.pauli("Z"), 0)
        r = self._exact(2, 0, qu.pauli("Z"), [gate])
        assert complex(v) == pytest.approx(r, abs=1e-9)

    def test_geometry_inferred_from_gates(self):
        rng = np.random.default_rng(1)
        gates = [
            qtn.Gate(
                "SU4", params=rng.uniform(0, 2 * np.pi, 15), qubits=[0, 1]
            ),
            qtn.Gate(
                "SU4", params=rng.uniform(0, 2 * np.pi, 15), qubits=[1, 2]
            ),
        ]
        circ = qtn.CircuitPEPOSimpleUpdate(gates=gates, max_bond=8)
        assert set(circ.sites) == {0, 1, 2}
        assert frozenset((0, 1)) in {frozenset(e) for e in circ.edges}

    def test_reverse_lightcone_preserves_result(self):
        # a single qubit observable: gates outside its reverse lightcone are
        # skipped (U^dag U = 1), and the result must be unchanged
        N = 6
        edges = [(i, i + 1) for i in range(N - 1)]
        gates = self._chain_gates(N, depth=4, seed=5)

        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=2**N)
        circ.apply_gates(gates)

        v = circ.local_expectation(qu.pauli("Z"), 0)
        r = self._exact(N, 0, qu.pauli("Z"), gates)
        assert complex(v) == pytest.approx(r, abs=1e-6)

    def test_get_evolved_operator(self):
        edges = [(0, 1), (1, 2)]
        gates = self._chain_gates(3, depth=2, seed=7)
        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=8)
        circ.apply_gates(gates)

        op = circ.get_evolved_operator(qu.pauli("Z"), 1)
        assert isinstance(op, qtn.TensorNetworkGenOperator)

        # the sandwiched network contracts to the same scalar as
        # local_expectation
        tn = circ.get_evolved_operator_with_state(qu.pauli("Z"), 1)
        assert complex(tn.contract(all)) == pytest.approx(
            complex(circ.local_expectation(qu.pauli("Z"), 1)), abs=1e-9
        )

    def test_two_site_observable(self):
        N = 4
        edges = [(i, i + 1) for i in range(N - 1)]
        gates = self._chain_gates(N, depth=3, seed=11)
        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=2**N)
        circ.apply_gates(gates)

        zz = qu.pauli("Z") & qu.pauli("Z")
        v = circ.local_expectation(zz, (1, 2))
        U = np.eye(2**N, dtype=complex)
        for g in gates:
            U = qu.pkron(self._mat(g.array), [2] * N, list(g.qubits)) @ U
        Of = qu.pkron(self._mat(zz), [2] * N, [1, 2])
        p = qu.basis_vec(0, 2**N)
        r = complex((p.conj().T @ (U.conj().T @ Of @ U) @ p)[0, 0])
        assert complex(v) == pytest.approx(r, abs=1e-7)

    def test_2d_grid_dynamics(self):
        # 2x3 grid Ising Trotter dynamics; vertical bonds are long range in any
        # 1D ordering, exercising the arbitrary geometry PEPO
        Lx, Ly = 2, 3

        def site(x, y):
            return x * Ly + y

        edges = []
        for x in range(Lx):
            for y in range(Ly):
                if y + 1 < Ly:
                    edges.append((site(x, y), site(x, y + 1)))
                if x + 1 < Lx:
                    edges.append((site(x, y), site(x + 1, y)))
        N = Lx * Ly

        dt = 0.15
        rzz = qu.expm(-1j * dt * (qu.pauli("Z") & qu.pauli("Z")))
        rx = qu.expm(-1j * dt * qu.pauli("X"))
        gates = []
        for _ in range(2):
            for i in range(N):
                gates.append(qtn.Gate.from_raw(rx, qubits=[i]))
            for a, b in edges:
                gates.append(qtn.Gate.from_raw(rzz, qubits=[a, b]))

        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=2**N)
        circ.apply_gates(gates)

        v = circ.local_expectation(qu.pauli("Z"), site(0, 1))
        r = self._exact(N, site(0, 1), qu.pauli("Z"), gates)
        assert complex(v) == pytest.approx(r, abs=1e-7)

    def test_requires_geometry(self):
        with pytest.raises(ValueError):
            qtn.CircuitPEPOSimpleUpdate()

    def test_validation(self):
        edges = [(0, 1), (1, 2)]
        circ = qtn.CircuitPEPOSimpleUpdate(edges=edges, max_bond=8)

        with pytest.raises(ValueError):
            # two qubit gate not on an edge
            circ.apply_gate(qtn.Gate.from_raw(qu.rand_uni(4), qubits=[0, 2]))
        with pytest.raises(ValueError):
            # three qubit gate
            circ.apply_gate(
                qtn.Gate.from_raw(qu.rand_uni(8), qubits=[0, 1, 2])
            )
        with pytest.raises(ValueError):
            # controlled gate
            circ.apply_gate(
                qtn.Gate.from_raw(qu.rand_uni(2), qubits=[1], controls=[0])
            )
        with pytest.raises(ValueError):
            # observable off the geometry
            circ.local_expectation(qu.pauli("Z"), 5)
        with pytest.raises(ValueError):
            # two site observable not on an edge
            circ.local_expectation(qu.pauli("Z") & qu.pauli("Z"), (0, 2))

    def test_state_methods_unsupported(self):
        circ = qtn.CircuitPEPOSimpleUpdate(edges=[(0, 1)], max_bond=4)
        circ.apply_gate(qtn.Gate("H", params=(), qubits=[0]))
        with pytest.raises(NotImplementedError):
            circ.psi
        with pytest.raises(NotImplementedError):
            circ.sample(10)
        with pytest.raises(NotImplementedError):
            circ.amplitude("00")
