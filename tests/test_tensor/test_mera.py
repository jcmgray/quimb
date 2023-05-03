import pytest
import quimb as qu
import quimb.tensor as qt


class TestMERA:

    @pytest.mark.parametrize("dtype", [float, complex])
    def test_construct_random(self, dtype):

        mera = qt.MERA.rand(16)

        # test outer inds
        assert set(mera.outer_inds()) == {f'k{i}' for i in range(16)}

        # test normalized
        assert (mera.H & mera) ^ all == pytest.approx(1.0)

        # test auto contract all
        assert mera.H @ mera == pytest.approx(1.0)

        # test dense conversion
        md = mera.to_qarray()
        assert md.H @ md == pytest.approx(1.0)

    def test_1d_vector_methods(self):
        X = qu.spin_operator('X', sparse=True)

        mera = qt.MERA.rand(16)
        meraX = mera.gate(X.A, 7)
        assert mera is not meraX
        x1 = mera.H @ meraX

        md = mera.to_qarray()
        mdX = qu.ikron(X, [2] * 16, 7) @ md
        x2 = md.H @ mdX
        # check against dense
        assert x1 == pytest.approx(x2)

        # check 'outside lightcone' unaffected
        assert mera.select(3).H @ meraX.select(3) == pytest.approx(1.0)

        # check only need 'lightcone' to compute local
        assert mera.select(7).H @ meraX.select(7) == pytest.approx(x2)

    @pytest.mark.parametrize("method", ['qr', 'exp', 'cayley', 'mgs', 'svd'])
    def test_isometrize(self, method):
        mera = qt.MERA.rand(16, dangle=True)
        assert mera.H @ mera == pytest.approx(2.0)
        for t in mera:
            t.modify(data=qu.randn(t.shape))
        assert mera.H @ mera != pytest.approx(2.0)
        mera.isometrize_(method=method)
        assert mera.H @ mera == pytest.approx(2.0)
