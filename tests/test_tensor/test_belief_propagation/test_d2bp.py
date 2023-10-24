import pytest

import quimb.tensor as qtn
from quimb.experimental.belief_propagation.d2bp import (
    contract_d2bp,
    compress_d2bp,
    sample_d2bp,
)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_contract(damping, dtype):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # normalize exactly
    peps /= (peps.H @ peps) ** 0.5
    info = {}
    N_ap = contract_d2bp(peps, damping=damping, info=info, progbar=True)
    assert info["converged"]
    assert N_ap == pytest.approx(1.0, rel=0.3)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_tree_exact(dtype):
    psi = qtn.TN_rand_tree(20, 3, 2, dtype=dtype, seed=42)
    norm2 = psi.H @ psi
    info = {}
    norm2_bp = contract_d2bp(psi, info=info, progbar=True)
    assert info["converged"]
    assert norm2_bp == pytest.approx(norm2, rel=5e-6)


@pytest.mark.parametrize("damping", [0.0, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_compress(damping, dtype):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # test that using the BP compression gives better fidelity than purely
    # local, naive compression scheme
    peps_c1 = peps.compress_all(max_bond=2)
    info = {}
    peps_c2 = compress_d2bp(
        peps, max_bond=2, damping=damping, info=info, progbar=True
    )
    assert info["converged"]
    fid1 = peps_c1.H @ peps_c2
    fid2 = peps_c2.H @ peps_c2
    assert abs(fid2) > abs(fid1)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_sample(dtype):
    peps = qtn.PEPS.rand(3, 4, 3, seed=42, dtype=dtype)
    # normalize exactly
    peps /= (peps.H @ peps) ** 0.5
    config, peps_config, omega = sample_d2bp(peps, seed=42, progbar=True)
    assert all(ix in config for ix in peps.site_inds)
    assert 0.0 < omega < 1.0
    assert peps_config.outer_inds() == ()

    ptotal = 0.0
    nrepeat = 4
    for _ in range(nrepeat):
        _, peps_config, _ = sample_d2bp(peps, seed=42, progbar=True)
        ptotal += abs(peps_config.contract()) ** 2

    # check we are doing better than random guessing
    assert ptotal > nrepeat * 2**-peps.nsites
