import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn


class TestCircuitGates:
    @pytest.mark.parametrize(
        "Circ", [qtn.Circuit, qtn.CircuitMPS, qtn.CircuitDense]
    )
    def test_all_gate_methods(self, Circ):
        import random

        g_nq_np = [
            # single qubit
            ("x", 1, 0),
            ("y", 1, 0),
            ("z", 1, 0),
            ("s", 1, 0),
            ("sdg", 1, 0),
            ("t", 1, 0),
            ("tdg", 1, 0),
            ("h", 1, 0),
            ("sx", 1, 0),
            ("sxdg", 1, 0),
            ("iden", 1, 0),
            ("x_1_2", 1, 0),
            ("y_1_2", 1, 0),
            ("z_1_2", 1, 0),
            ("w_1_2", 1, 0),
            ("hz_1_2", 1, 0),
            # single qubit parametrizable
            ("rx", 1, 1),
            ("ry", 1, 1),
            ("rz", 1, 1),
            ("u3", 1, 3),
            ("u2", 1, 2),
            ("u1", 1, 1),
            ("phase", 1, 1),
            # two qubit
            ("cx", 2, 0),
            ("cy", 2, 0),
            ("cz", 2, 0),
            ("cnot", 2, 0),
            ("swap", 2, 0),
            ("iswap", 2, 0),
            # two qubit parametrizable
            ("rxx", 2, 1),
            ("ryy", 2, 1),
            ("rzz", 2, 1),
            ("crx", 2, 1),
            ("cry", 2, 1),
            ("crz", 2, 1),
            ("cu3", 2, 3),
            ("cu2", 2, 2),
            ("cu1", 2, 1),
            ("cphase", 2, 1),
            ("fsim", 2, 2),
            ("fsimg", 2, 5),
            ("givens", 2, 1),
            ("givens2", 2, 2),
            ("xx_plus_yy", 2, 2),
            ("xx_minus_yy", 2, 2),
            ("su4", 2, 15),
        ]
        random.shuffle(g_nq_np)

        psi0 = qtn.MPS_rand_state(2, 2)
        circ = Circ(2, psi0=psi0, tags="PSI0")

        for g, n_q, n_p in g_nq_np:
            args = [
                *np.random.uniform(0, 2 * np.pi, size=n_p),
                *np.random.choice([0, 1], replace=False, size=n_q),
            ]
            getattr(circ, g)(*args)

        assert circ.psi.H @ circ.psi == pytest.approx(1.0)
        assert abs((circ.psi.H & psi0) ^ all) < 0.99999999

    def test_su4(self):
        psi0 = qtn.MPS_rand_state(2, 2)
        circ_a = qtn.Circuit(psi0=psi0)
        params = qu.randn(15)

        circ_a.su4(*params, 0, 1)
        psi_a = circ_a.to_dense()

        circ_b = qtn.Circuit(psi0=psi0)
        (
            theta1,
            phi1,
            lamda1,
            theta2,
            phi2,
            lamda2,
            theta3,
            phi3,
            lamda3,
            theta4,
            phi4,
            lamda4,
            t1,
            t2,
            t3,
        ) = params
        circ_b.u3(theta1, phi1, lamda1, 0)
        circ_b.u3(theta2, phi2, lamda2, 1)
        circ_b.cnot(1, 0)
        circ_b.rz(t1, 0)
        circ_b.ry(t2, 1)
        circ_b.cnot(0, 1)
        circ_b.ry(t3, 1)
        circ_b.cnot(1, 0)
        circ_b.u3(theta3, phi3, lamda3, 0)
        circ_b.u3(theta4, phi4, lamda4, 1)
        psi_b = circ_b.to_dense()

        assert qu.fidelity(psi_a, psi_b) == pytest.approx(1.0)

    def test_three_qubit_gates(self):
        psi0 = qtn.MPS_rand_state(3, 2)
        circ = qtn.Circuit(psi0=psi0)
        circ.ccx(0, 1, 2)
        circ.cswap(2, 1, 0)
        circ.toffoli(0, 1, 2)
        circ.ccy(1, 0, 2)
        circ.ccz(1, 2, 0)
        circ.fredkin(2, 1, 0)
        psi = circ.psi.to_dense()
        assert qu.expec(psi, psi) == pytest.approx(1.0)

    def test_auto_split_gate(self):
        ops = [
            ("u3", 1.0, 2.0, 3.0, 0),
            ("u3", 2.0, 3.0, 1.0, 1),
            ("u3", 3.0, 1.0, 2.0, 2),
            ("cz", 0, 1),
            ("iswap", 1, 2),
            ("cx", 2, 0),
            ("iswap", 2, 1),
            ("h", 0),
            ("h", 1),
            ("h", 2),
        ]
        cnorm = qtn.Circuit.from_gates(
            ops, gate_opts=dict(contract="split-gate")
        )
        assert cnorm.psi.max_bond() == 4

        cswap = qtn.Circuit.from_gates(
            ops, gate_opts=dict(contract="swap-split-gate")
        )
        assert cswap.psi.max_bond() == 4

        cauto = qtn.Circuit.from_gates(
            ops, gate_opts=dict(contract="auto-split-gate")
        )
        assert cauto.psi.max_bond() == 2

        assert qu.fidelity(
            cnorm.psi.to_dense(), cswap.psi.to_dense()
        ) == pytest.approx(1.0)
        assert qu.fidelity(
            cswap.psi.to_dense(), cauto.psi.to_dense()
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "name, densefn, nparam, nqubit",
        [
            ("rx", qu.Rx, 1, 1),
            ("ry", qu.Ry, 1, 1),
            ("rz", qu.Rz, 1, 1),
            ("u3", qu.U_gate, 3, 1),
            ("fsim", qu.fsim, 2, 2),
            ("fsimg", qu.fsimg, 5, 2),
        ],
    )
    def test_parametrized_gates_rx(self, name, densefn, nparam, nqubit):
        k0 = qu.rand_ket(2**nqubit)
        params = qu.randn(nparam)
        kf = densefn(*params) @ k0
        k0mps = qtn.MatrixProductState.from_dense(k0, [2] * nqubit)
        circ = qtn.Circuit(psi0=k0mps, gate_opts={"contract": False})
        getattr(circ, name)(*params, *range(nqubit), parametrize=True)
        tn = circ.psi
        assert isinstance(tn["GATE_0"], qtn.PTensor)
        assert_allclose(circ.to_dense(), kf)

    def test_apply_raw_gate(self):
        k0 = qu.rand_ket(4)
        psi0 = qtn.MatrixProductState.from_dense(k0, [2] * 2)
        circ = qtn.Circuit(psi0=psi0)
        U = qu.rand_uni(4)
        circ.apply_gate_raw(U, [0, 1], tags="UCUSTOM")
        assert len(circ.gates) == 1
        assert "UCUSTOM" in circ.psi.tags
        assert qu.fidelity(circ.to_dense(), U @ k0) == pytest.approx(1)

    def test_apply_controlled_gate_basic_equiv(self):
        circ = qtn.Circuit(3)
        circ.apply_gate("x", qubits=(2,), controls=(0, 1))
        U = circ.get_uni().to_dense()
        assert_allclose(U, qu.toffoli())

        circ = qtn.Circuit(3)
        circ.apply_gate("swap", qubits=(1, 2), controls=(0,))
        U = circ.get_uni().to_dense()
        assert_allclose(U, qu.fredkin())

    def test_multi_controlled_circuit(self):
        import random

        N = 10
        circ = qtn.Circuit(N)
        regs = list(range(N))
        random.shuffle(regs)
        circ.apply_gate("H", regs[0])
        for i in range(N - 1):
            circ.apply_gate("CNOT", regs[i], regs[i + 1])
        circ.apply_gate("X", N - 1, controls=range(N - 1))
        circ.apply_gate("SWAP", qubits=(N - 2, N - 1), controls=range(N - 2))
        (b,) = circ.sample(1, group_size=3, seed=42)
        assert b[N - 2] == "0"


class TestGate:
    def test_copy_with_is_nonmutating(self):
        g = qtn.Gate("RX", params=(0.5,), qubits=(2,))
        g2 = g.copy_with(qubits=(3,))
        assert g2.qubits == (3,)
        assert g.qubits == (2,)  # original unchanged
        assert g2.label == "RX"
        assert tuple(g2.params) == (0.5,)

    @pytest.mark.parametrize(
        "label, params, qubits, L",
        [
            ("CX", (), (0, 1), 2),
            ("CX", (), (0, 2), 4),
            ("RZZ", (0.7,), (1, 3), 4),
        ],
    )
    def test_build_mpo_is_unitary(self, label, params, qubits, L):
        gate = qtn.Gate(label, params=params, qubits=qubits)
        mpo = gate.build_mpo(L=L)
        assert isinstance(mpo, qtn.MatrixProductOperator)
        assert mpo.L == L
        U = mpo.to_dense()
        assert_allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=1e-10)
