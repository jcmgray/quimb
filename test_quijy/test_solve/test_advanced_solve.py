import importlib
from pytest import fixture, mark
import numpy as np
from numpy.testing import assert_allclose
from quijy import ldmul, rand_uni, qjf
from quijy.solve.advanced_solve import aeigsys


slepc4py_spec = importlib.util.find_spec("slepc4py")
slepc4py_notfound = slepc4py_spec is None
slepc4py_notfound_msg = "No SLEPc4py installation"


@fixture
def prematsparse():
    u = rand_uni(4)
    a = u @ ldmul(np.array([-1, 2, 4, -3]), u.H)
    a = qjf(a, sparse=True)
    return u, a


@mark.skipif(slepc4py_notfound, reason=slepc4py_notfound_msg)
def test_internal_eigvals(prematsparse):
    u, a = prematsparse
    lk = aeigsys(a, k=2, sigma=0.5)
    assert_allclose(lk, [-1, 2])


@mark.skipif(slepc4py_notfound, reason=slepc4py_notfound_msg)
def test_aeigsys_groundenergy(prematsparse):
    u, a = prematsparse
    lk = aeigsys(a, k=1, which="SR")
    assert_allclose(lk, -3)
    lk = aeigsys(a, k=1, which="LR")
    assert_allclose(lk, 4)
