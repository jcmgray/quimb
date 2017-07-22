import pytest
import numpy as np
from numpy.testing import assert_allclose

from quimb import (
    rand_herm,
    eigsys,
)

from quimb.solve import SCALAPY_FOUND

if SCALAPY_FOUND:
    from quimb.solve.scalapy_solver import (
        eigsys_scalapy,
    )


scalapy_notfound_msg = "No scalapy installation"
needs_scalapy = pytest.mark.skipif(not SCALAPY_FOUND,
                                   reason=scalapy_notfound_msg)


@needs_scalapy
class TestScalapyEigsys:

    def test_simple(self):
        a = rand_herm(43)

        l, v = eigsys_scalapy(a)
        le, ve = eigsys(a)
        assert isinstance(v, np.matrix)
        assert_allclose(l, le)

    def test_partial(self):

        a = rand_herm(43)

        el = eigsys_scalapy(a, k=10, return_vecs=False)
        elex, evex = eigsys(a)

        assert_allclose(el, elex[:10])
