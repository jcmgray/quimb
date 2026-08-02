import quimb.tensor as qtn
from quimb.tensor.tnag.compress import tensor_network_ag_compress


def test_compress_projector_bp_canonize():
    psi = qtn.PEPS.rand(2, 2, bond_dim=2, phys_dim=2, seed=42)
    info = {}

    compressed = tensor_network_ag_compress(
        psi,
        max_bond=1,
        method="projector",
        canonize="bp",
        power=0.75,
        gauge_power=0.75,
        canonize_opts={
            "max_iterations": 100,
            "tol": 1e-8,
            "info": info,
        },
    )

    assert info["converged"]
    assert compressed is not psi
    assert compressed.num_tensors == len(psi.site_tags)
    assert compressed.max_bond() == 1
    assert psi.max_bond() == 2
