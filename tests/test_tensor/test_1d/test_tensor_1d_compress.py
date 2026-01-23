import pytest

import quimb as qu
import quimb.tensor as qtn

dtypes = ["float32", "float64", "complex64", "complex128"]


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "fit",
        "zipup",
        "zipup-first",
    ],
)
@pytest.mark.parametrize("dtype", dtypes)
def test_mps_partial_mpo_apply(method, dtype):
    mps = qtn.MPS_rand_state(10, 7, dtype=dtype)
    A = qu.rand_uni(2**3, dtype=dtype)
    where = [8, 4, 5]
    mpo = qtn.MatrixProductOperator.from_dense(A, sites=where)
    new = mps.gate_with_op_lazy(mpo)
    assert (
        qtn.tensor_network_1d_compress(new, method=method, inplace=True) is new
    )
    assert new.num_tensors == 10
    assert new.distance_normalized(mps.gate(A, where)) == pytest.approx(
        0.0, abs=1e-3 if dtype in ("float32", "complex64") else 1e-6
    )


@pytest.mark.parametrize(
    "method",
    [
        "direct",
        "dm",
        "fit",
        "zipup",
        "zipup-first",
        "src",
        "src-first",
    ],
)
@pytest.mark.parametrize("sweep_reverse", [False, True])
def test_mpo_compress_opts(method, sweep_reverse):
    L = 6
    A = qtn.MPO_rand(L, 2, phys_dim=3, tags="A")
    B = qtn.MPO_rand(L, 3, phys_dim=3, tags="B")
    AB = A.gate_upper_with_op_lazy(B)
    assert AB.num_tensors == 2 * L
    ABc = qtn.tensor_network_1d_compress(
        AB,
        method=method,
        max_bond=5,
        cutoff=1e-6,
        sweep_reverse=sweep_reverse,
        inplace=False,
    )
    assert ABc.num_tensors == L
    assert ABc.num_indices == 2 * L + L - 1
    assert ABc.max_bond() == 5
    if sweep_reverse:
        assert ABc.calc_current_orthog_center() == (L - 1, L - 1)
    else:
        assert ABc.calc_current_orthog_center() == (0, 0)

    for site in range(L):
        assert set(ABc[site].tags) == {"A", "B", f"I{site}"}
