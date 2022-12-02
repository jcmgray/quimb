import pytest
import numpy as np

import quimb.tensor as qtn
from quimb.tensor.contraction import _CONTRACT_BACKEND, _TENSOR_LINOP_BACKEND


class TestContractOpts:

    def test_contract_strategy(self):
        assert qtn.get_contract_strategy() == 'greedy'
        with qtn.contract_strategy('auto'):
            assert qtn.get_contract_strategy() == 'auto'
        assert qtn.get_contract_strategy() == 'greedy'

    def test_contract_backend(self):
        assert qtn.get_contract_backend() == _CONTRACT_BACKEND
        with qtn.contract_backend('cupy'):
            assert qtn.get_contract_backend() == 'cupy'
        assert qtn.get_contract_backend() == _CONTRACT_BACKEND

    def test_tensor_linop_backend(self):
        assert qtn.get_tensor_linop_backend() == _TENSOR_LINOP_BACKEND
        with qtn.tensor_linop_backend('cupy'):
            assert qtn.get_tensor_linop_backend() == 'cupy'
        assert qtn.get_tensor_linop_backend() == _TENSOR_LINOP_BACKEND

    def test_contract_cache(self):
        import tempfile
        import os
        from opt_einsum.paths import register_path_fn

        info = {'num_calls': 0}

        def my_custom_opt(inputs, output, size_dict, memory_limit=None):
            info['num_calls'] += 1
            return [(0, 1)] * (len(inputs) - 1)

        register_path_fn('quimb_test_opt', my_custom_opt)

        tn = qtn.MPS_rand_state(4, 3) & qtn.MPS_rand_state(4, 3)
        assert (
            tn.contract(all, optimize='quimb_test_opt', get='expression')
            is
            tn.contract(all, optimize='quimb_test_opt', get='expression'))
        assert info['num_calls'] == 1

        # contraction pathinfo objects are now cached together
        assert (
            tn.contract(all, optimize='quimb_test_opt', get='path-info')
            is
            tn.contract(all, optimize='quimb_test_opt', get='path-info'))
        assert info['num_calls'] == 1

        # set a directory cache - functions will be run fresh again
        with tempfile.TemporaryDirectory() as tdir:
            assert len(os.listdir(tdir)) == 0
            qtn.set_contract_path_cache(tdir)
            assert (
                tn.contract(all, optimize='quimb_test_opt', get='expression')
                is
                tn.contract(all, optimize='quimb_test_opt', get='expression'))
            assert info['num_calls'] == 2
            assert (
                tn.contract(all, optimize='quimb_test_opt', get='path-info')
                is
                tn.contract(all, optimize='quimb_test_opt', get='path-info'))
            assert info['num_calls'] == 2
            assert len(os.listdir(tdir)) != 0

            # need to release close the cache so the directory can be deleted
            qtn.set_contract_path_cache(None)


@pytest.mark.parametrize('around', ['I3,3', 'I0,0', 'I1,2'])
@pytest.mark.parametrize('equalize_norms', [False, True])
@pytest.mark.parametrize('gauge_boundary_only', [False, True])
def test_contract_approx_with_gauges(
    around,
    equalize_norms,
    gauge_boundary_only
):
    rng = np.random.default_rng(42)
    tn = qtn.TN2D_from_fill_fn(
        lambda shape: rng.uniform(size=shape, low=-0.5),
        7, 7, 4
    )
    Zex = tn ^ ...
    Z = tn.contract_around(
        around,
        max_bond=8,
        gauges=True,
        gauge_boundary_only=gauge_boundary_only,
        tree_gauge_distance=2,
        equalize_norms=equalize_norms,
        progbar=True,
    )
    assert Z == pytest.approx(Zex, rel=1e-2)
