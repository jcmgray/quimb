import importlib

import pytest

found_torch = importlib.util.find_spec("torch") is not None
found_autograd = importlib.util.find_spec("autograd") is not None
found_jax = importlib.util.find_spec("jax") is not None
found_tensorflow = importlib.util.find_spec("tensorflow") is not None

if found_tensorflow:
    import tensorflow.experimental.numpy as tnp

    tnp.experimental_enable_numpy_behavior()

jax_case = pytest.param(
    "jax", marks=pytest.mark.skipif(not found_jax, reason="jax not installed")
)
autograd_case = pytest.param(
    "autograd",
    marks=pytest.mark.skipif(
        not found_autograd, reason="autograd not installed"
    ),
)
tensorflow_case = pytest.param(
    "tensorflow",
    marks=pytest.mark.skipif(
        not found_tensorflow, reason="tensorflow not installed"
    ),
)
pytorch_case = pytest.param(
    "torch",
    marks=pytest.mark.skipif(not found_torch, reason="pytorch not installed"),
)

requires_symmray = pytest.mark.skipif(
    importlib.util.find_spec("symmray") is None,
    reason="symmray not installed",
)

# scalar 2D networks for exact compression tests
symmetry_cases = [
    "abelian",
    "fermionic-even",
    "fermionic-odd-checker",
    "fermionic-odd-diag",
]

# uniform and mixed bond orientations
bond_orientations = ["reversed", "canonical", "random"]

site_charge_cases = {
    # odd checkerboard sites; even total charge
    "fermionic-odd-checker": lambda site: (site[0] + site[1]) % 2,
    # odd diagonal sites; even total charge
    "fermionic-odd-diag": lambda site: int(site[0] == site[1]),
}


def make_symmetric_2d_tn(
    symmetry,
    duals="reversed",
    Lx=4,
    Ly=4,
    bond_dim=2,
    seed=42,
    dist="normal",
):
    """Make a scalar Z2-symmetric 2D tensor network for tests."""
    import symmray as sr

    return sr.TN2D_abelian_rand(
        symmetry="Z2",
        Lx=Lx,
        Ly=Ly,
        bond_dim=bond_dim,
        phys_dim=None,
        fermionic=symmetry != "abelian",
        site_charge=site_charge_cases.get(symmetry),
        duals=duals,
        seed=seed,
        dist=dist,
    )
