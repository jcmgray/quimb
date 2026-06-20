import pytest

import quimb as qu
import quimb.tensor as qtn

from ._helpers import rand_reg_graph


class TestCircuitGen:
    @pytest.mark.parametrize(
        "ansatz,cyclic",
        [
            ("zigzag", False),
            ("brickwork", False),
            ("brickwork", True),
            ("rand", False),
            ("rand", True),
        ],
    )
    @pytest.mark.parametrize("n", [4, 5])
    def test_1D_ansatzes(self, ansatz, cyclic, n):
        depth = 3
        num_pairs = n if cyclic else n - 1

        fn = {
            "zigzag": qtn.circ_ansatz_1D_zigzag,
            "brickwork": qtn.circ_ansatz_1D_brickwork,
            "rand": qtn.circ_ansatz_1D_rand,
        }[ansatz]

        opts = dict(
            n=n,
            depth=3,
            gate_opts=dict(contract=False),
        )
        if cyclic:
            opts["cyclic"] = True
        if ansatz == "rand":
            opts["seed"] = 42

        circ = fn(**opts)
        tn = circ.uni

        # total number of entangling gates
        assert len(tn["CZ"]) == num_pairs * depth

        # number of entangling gates per pair
        for i in range(num_pairs):
            assert len(tn["CZ", f"I{i}", f"I{(i + 1) % n}"]) == depth

        assert all(isinstance(t, qtn.PTensor) for t in tn["U3"])

    def test_qaoa(self):
        G = rand_reg_graph(3, 10, seed=666)
        terms = {(i, j): 1.0 for i, j in G.edges}
        ZZ = qu.pauli("Z") & qu.pauli("Z")

        gammas = [-0.6]
        betas = [-0.4]

        circ1 = qtn.circ_qaoa(terms, 1, gammas, betas)

        energy1 = sum(circ1.local_expectation(ZZ, edge) for edge in terms)
        assert energy1 < -4

        gammas = [-0.4]
        betas = [0.3]

        circ2 = qtn.circ_qaoa(terms, 1, gammas, betas)

        energy2 = sum(circ2.local_expectation(ZZ, edge) for edge in terms)
        assert energy2 > 4
