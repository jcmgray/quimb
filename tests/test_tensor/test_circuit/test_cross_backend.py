"""Unified (same input and final call) circuit tests for making sure
all effectively exact simulators match.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import quimb as qu
import quimb.tensor as qtn

# exact (or effectively-exact at this size/shallow depth) simulators
EXACT_SIMS = [
    qtn.Circuit,
    qtn.CircuitDense,
    qtn.CircuitMPS,
    qtn.CircuitMPSLazy,
    qtn.CircuitPermMPS,
]
SIM_IDS = [S.__name__ for S in EXACT_SIMS]
N = 6
BITSTRINGS = ["0" * N, "1" * N, "010101", "110010"]


def get_gates_ref():
    """A fixed, deterministic, low-entanglement circuit with 1q, 2q,
    parametrized, controlled and long-range gates, shallow enough to stay
    exact for the MPS representations at default truncation. The long-range
    gates induce a non-trivial qubit permutation in ``CircuitPermMPS``."""
    rng = np.random.default_rng(1234)
    one_q = ["RX", "RY", "RZ"]
    gs = [("H", i) for i in range(N)]
    for layer in range(3):
        for i in range(layer % 2, N - 1, 2):
            gs.append(("CX", i, i + 1))
        for i in range(N):
            gs.append(
                (one_q[(i + layer) % 3], float(rng.uniform(0, 2 * np.pi)), i)
            )
    gs.append(("RZZ", 0.5, 0, 1))
    gs.append(("CZ", 2, 3))
    # long-range (non-adjacent) gates
    gs.append(("CX", 0, 4))
    gs.append(("RZZ", 0.3, 1, 5))
    return gs


@pytest.fixture(scope="module")
def circuit_ref():
    return qtn.Circuit.from_gates(get_gates_ref())


@pytest.mark.parametrize("Sim", EXACT_SIMS, ids=SIM_IDS)
def test_to_dense_matches_exact(Sim, circuit_ref):
    c = Sim.from_gates(get_gates_ref())
    assert_allclose(c.to_dense(), circuit_ref.to_dense(), atol=1e-10)


@pytest.mark.parametrize("Sim", EXACT_SIMS, ids=SIM_IDS)
def test_amplitude_matches_exact(Sim, circuit_ref):
    c = Sim.from_gates(get_gates_ref())
    for b in BITSTRINGS:
        assert_allclose(c.amplitude(b), circuit_ref.amplitude(b), atol=1e-10)


@pytest.mark.parametrize("Sim", EXACT_SIMS, ids=SIM_IDS)
def test_local_expectation_matches_exact(Sim, circuit_ref):
    c = Sim.from_gates(get_gates_ref())
    Z = qu.pauli("Z")
    for i in range(N):
        assert_allclose(
            c.local_expectation(Z, i),
            circuit_ref.local_expectation(Z, i),
            atol=1e-10,
        )
    ZZ = qu.pauli("Z") & qu.pauli("Z")
    for where in [(0, 1), (0, 5)]:  # adjacent and long-range pairs
        assert_allclose(
            c.local_expectation(ZZ, where),
            circuit_ref.local_expectation(ZZ, where),
            atol=1e-10,
        )


@pytest.mark.parametrize("Sim", EXACT_SIMS, ids=SIM_IDS)
def test_sample_observables_match_exact(Sim, circuit_ref):
    """Estimate <Z_i> from samples and compare to exact, a statistical, but
    deterministic (fixed seed) with loose tolerance, sampling check."""
    c = Sim.from_gates(get_gates_ref())
    n_samples = 1000
    acc = np.zeros(N)
    for b in c.sample(n_samples, seed=2024):
        for i, x in enumerate(b):
            acc[i] += 1 if int(x) == 0 else -1
    z_emp = acc / n_samples
    z_exact = np.array(
        [
            circuit_ref.local_expectation(qu.pauli("Z"), i).real
            for i in range(N)
        ]
    )
    assert_allclose(z_emp, z_exact, atol=0.15)
