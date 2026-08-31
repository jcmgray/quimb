import pytest

import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tnag.compress import tensor_network_ag_compress

from .. import (
    bond_orientations,
    make_symmetric_2d_tn,
    requires_symmray,
    symmetry_cases,
)


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


@requires_symmray
@pytest.mark.parametrize("symmetry", symmetry_cases)
@pytest.mark.parametrize("canonize", [False, True, "layered", "bp"])
@pytest.mark.parametrize("bond_orientation", bond_orientations)
@pytest.mark.parametrize("direction", ["xmin", "xmax", "ymin", "ymax"])
def test_compress_projector_symmetric(
    symmetry, canonize, bond_orientation, direction, request
):
    """Check exact boundary contraction with local projectors.

    Cover each preconditioner, bond orientation, and contraction direction.
    Use enough bond dimension to prevent truncation.
    """
    canonize_opts = None
    if canonize == "bp":
        # isolate phase errors with a tightly converged BP gauge
        canonize_opts = {"max_iterations": 1000, "tol": 5e-13}

        if symmetry != "abelian":
            # BP messages do not retain the full fermionic phase frame
            request.applymarker(
                pytest.mark.xfail(
                    reason="BP messages lose the fermionic phase frame",
                    strict=False,
                )
            )

    # use distinct random data for each case
    seed = qu.utils.hash_kwargs_to_int(
        symmetry=symmetry,
        canonize=canonize,
        bond_orientation=bond_orientation,
        direction=direction,
    )
    tn = make_symmetric_2d_tn(symmetry, duals=bond_orientation, seed=seed)

    expected = tn.contract(all, optimize="auto-hq")

    # `contract_boundary` routes this to `tensor_network_ag_compress`
    value = tn.contract_boundary(
        # prevent truncation
        max_bond=8,
        cutoff=0.0,
        mode="projector",
        canonize=canonize,
        canonize_opts=canonize_opts,
        sequence=(direction,),
    )

    assert value == pytest.approx(expected, rel=1e-10)
