import pytest

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
        G = rand_reg_graph(3, 18, seed=42)
        qasm = graph_to_circ(G)
        qc = qtn.Circuit.from_qasm(qasm)
        assert (qc.psi.H & qc.psi) ^ all == pytest.approx(1.0)
