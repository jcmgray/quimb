from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose
from quijy.misc import (
    param_runner,
    sub_split,
    np_param_runner,
    np_param_runner2,
    xr_param_runner,
)


class TestCaseRunner:
    def test_param_runner_simple(self):
        def foo(a, b, c):
            return a + b + c
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = [*param_runner(foo, params)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_param_runner_dict(self):
        def foo(a, b, c):
            return a + b + c
        params = OrderedDict((('a', [1, 2]),
                              ('b', [10, 20, 30]),
                              ('c', [100, 200, 300, 400])))
        x = [*param_runner(foo, params)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_sub_split(self):
        a = [[[('a', 1), ('b', 2)],
              [('c', 3), ('d', 4)]],
             [[('e', 5), ('f', 6)],
              [('g', 7), ('h', 8)]]]
        c, d = sub_split(a)
        assert c.tolist() == [[['a', 'b'],
                               ['c', 'd']],
                              [['e', 'f'],
                               ['g', 'h']]]
        assert d.tolist() == [[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]]

    def test_param_runner_np(self):
        def foo(a, b, c):
            return a + b + c
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = np_param_runner(foo, params)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_param_runner_np_multires(self):
        def foo(a, b, c):
            return a + b + c, a % 2 == 0
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = np_param_runner(foo, params)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x[0], xn)
        assert np.all(x[1][1, ...])

    def test_param_runner_np2(self):
        def foo(a, b, c):
            return a + b + c
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = np_param_runner2(foo, params)
        assert x.shape == (2, 3, 4)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_param_runner_np2_multires(self):
        def foo(a, b, c):
            return a + b + c, a % 2 == 0
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = np_param_runner2(foo, params)
        assert x[0].shape == (2, 3, 4)
        assert x[0].dtype == int
        assert x[1].shape == (2, 3, 4)
        assert x[1].dtype == bool
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x[0], xn)
        assert np.all(x[1][1, ...])

    def test_param_runner_np_dict(self):
        def foo(a, b, c):
            return a + b + c
        params = OrderedDict((('a', [1, 2]),
                              ('b', [10, 20, 30]),
                              ('c', [100, 200, 300, 400])))
        x = [*np_param_runner(foo, params)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_xr_param_runner(self):
        def foo(a, b, c):
            return a + b + c
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = xr_param_runner(foo, params, 'bananas')
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_xr_param_runner_multiresult(self):
        def foo(a, b, c):
            return a + b + c, a % 2 == 0
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = xr_param_runner(foo, params, ['bananas', 'cakes'])
        assert ds.bananas.data.dtype == int
        assert ds.cakes.data.dtype == bool
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
        assert ds.sel(a=1, b=10, c=100)['bananas'].data == 111
        assert ds.sel(a=2, b=30, c=400)['cakes'].data
        assert not ds.sel(a=1, b=10, c=100)['cakes'].data
