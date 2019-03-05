import pytest
import numpy as np

import quimb as qu
import quimb.tensor as qtn


def rand_reg_graph(reg, n, seed=None):
    import networkx as nx
    G = nx.random_regular_graph(reg, n, seed=seed)
    return G


def graph_to_circ(G, gamma0=-0.743043, beta0=0.754082):
    n = G.number_of_nodes()

    # add all the gates
    circ = "{}\n".format(n)
    for i in range(n):
        circ += "H {}\n".format(i)
    for i, j in G.edges:
        circ += "CNOT {} {}\n".format(i, j)
        circ += "Rz {} {}\n".format(gamma0, j)
        circ += "CNOT {} {}\n".format(i, j)
    for i in range(n):
        circ += "Rx {} {}\n".format(beta0, i)

    return circ


class TestCircuit:

    def test_prepare_GHZ(self):
        qc = qtn.Circuit(3)
        gates = [
            ('H', 0),
            ('H', 1),
            ('CNOT', 1, 2),
            ('CNOT', 0, 2),
            ('H', 0),
            ('H', 1),
            ('H', 2),
        ]
        qc.apply_circuit(gates)
        assert qu.expec(qc.psi.to_dense(), qu.ghz_state(3)) == pytest.approx(1)

    def test_rand_reg_qaoa(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_circ(G)
        qc = qtn.Circuit.from_qasm(qasm)
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    def test_rand_reg_qaoa_mps_swapsplit(self):
        G = rand_reg_graph(reg=3, n=18, seed=42)
        qasm = graph_to_circ(G)
        qc = qtn.CircuitMPS.from_qasm(qasm)
        assert len(qc.psi.tensors) == 18
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)

    @pytest.mark.parametrize(
        'Circ', [qtn.Circuit, qtn.CircuitMPS, qtn.CircuitDense]
    )
    def test_all_gate_methods(self, Circ):
        rots = ['rx', 'ry', 'rz']
        g1s = ['x', 'y', 'z', 's', 't', 'h', 'iden']
        g2s = ['cx', 'cy', 'cz', 'cnot', 'swap']
        g_rand = np.random.permutation(rots + g1s + g2s + ['u3'])

        psi0 = qtn.MPS_rand_state(2, 2)
        circ = Circ(2, psi0)

        for g in g_rand:
            if g == 'u3':
                angles = np.random.uniform(0, 2 * np.pi, size=3)
                i = np.random.choice([0, 1])
                args = (*angles, i)
            elif g in rots:
                theta = np.random.uniform(0, 2 * np.pi)
                i = np.random.choice([0, 1])
                args = (theta, i)
            elif g in g1s:
                i = np.random.choice([0, 1])
                args = (i,)
            elif g in g2s:
                i, j = np.random.permutation([0, 1])
                args = (i, j)

            getattr(circ, g)(*args)

        assert circ.psi.H @ circ.psi == pytest.approx(1.0)
        assert abs((circ.psi.H & psi0) ^ all) < 0.99999999
