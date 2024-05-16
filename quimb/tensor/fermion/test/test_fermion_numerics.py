import pytest
import numpy as np
from quimb.tensor.fermion.fermion_core import (
    FermionTensor,
    FermionTensorNetwork,
)
from quimb.tensor.tensor_core import tensor_contract
from quimb.tensor.fermion.block_gen import rand_all_blocks as rand
from quimb.tensor.fermion.block_interface import set_options

set_options(fermion=True)


@pytest.fixture(scope="class")
def u11setup(request):
    bond = [(0, 0), (1, 1), (1, -1), (2, 0)]
    set_options(symmetry="u11")
    request.cls.abc = abc = rand(
        (4, 2, 3), [bond] * 3, pattern="+--", dq=(1, 1)
    )
    request.cls.bcd = bcd = rand(
        (2, 3, 5), [bond] * 3, pattern="++-", dq=(-1, -1)
    )
    request.cls.ega = ega = rand(
        (3, 6, 4), [bond] * 3, pattern="+--", dq=(1, -1)
    )
    request.cls.deg = deg = rand(
        (5, 3, 6), [bond] * 3, pattern="+-+", dq=(-1, 1)
    )

    request.cls.Tabc = Tabc = FermionTensor(
        abc, inds=["a", "b", "c"], tags=["abc"]
    )
    request.cls.Tega = Tega = FermionTensor(
        ega, inds=["e", "g", "a"], tags=["ega"]
    )
    request.cls.Tbcd = Tbcd = FermionTensor(
        bcd, inds=["b", "c", "d"], tags=["bcd"]
    )
    request.cls.Tdeg = Tdeg = FermionTensor(
        deg, inds=["d", "e", "g"], tags=["deg"]
    )
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((2, 5), [bond] * 2, pattern="+-", dq=(0, 0))
    bc = rand((5, 4), [bond] * 2, pattern="++", dq=(1, -1))
    Tab = FermionTensor(ab, inds=["a", "b"], tags=["ab"])
    Tbc = FermionTensor(bc, inds=["b", "c"], tags=["bc"])
    Tab1 = FermionTensor(ab.dagger, inds=["b1", "a"], tags=["ab1"])
    Tbc1 = FermionTensor(bc.dagger, inds=["c", "b1"], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield


@pytest.fixture(scope="class")
def z22setup(request):
    bond = [(0, 0), (0, 1), (1, 0), (1, 1)]
    set_options(symmetry="z22")
    request.cls.abc = abc = rand(
        (4, 2, 3), [bond] * 3, pattern="+--", dq=(0, 1)
    )
    request.cls.bcd = bcd = rand(
        (2, 3, 5), [bond] * 3, pattern="++-", dq=(1, 0)
    )
    request.cls.ega = ega = rand(
        (3, 6, 4), [bond] * 3, pattern="+--", dq=(1, 0)
    )
    request.cls.deg = deg = rand(
        (5, 3, 6), [bond] * 3, pattern="+-+", dq=(0, 1)
    )

    request.cls.Tabc = Tabc = FermionTensor(
        abc, inds=["a", "b", "c"], tags=["abc"]
    )
    request.cls.Tega = Tega = FermionTensor(
        ega, inds=["e", "g", "a"], tags=["ega"]
    )
    request.cls.Tbcd = Tbcd = FermionTensor(
        bcd, inds=["b", "c", "d"], tags=["bcd"]
    )
    request.cls.Tdeg = Tdeg = FermionTensor(
        deg, inds=["d", "e", "g"], tags=["deg"]
    )
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((2, 5), [bond] * 2, pattern="+-", dq=(0, 0))
    bc = rand((5, 4), [bond] * 2, pattern="++", dq=(1, 0))

    Tab = FermionTensor(ab, inds=["a", "b"], tags=["ab"])
    Tbc = FermionTensor(bc, inds=["b", "c"], tags=["bc"])
    Tab1 = FermionTensor(ab.dagger, inds=["b1", "a"], tags=["ab1"])
    Tbc1 = FermionTensor(bc.dagger, inds=["c", "b1"], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield


@pytest.fixture(scope="class")
def u1setup(request):
    bond = (0, 1, 2, 3)
    set_options(symmetry="u1")

    request.cls.abc = abc = rand((4, 2, 3), [bond] * 3, pattern="+--", dq=1)
    request.cls.bcd = bcd = rand((2, 3, 5), [bond] * 3, pattern="++-", dq=2)
    request.cls.ega = ega = rand((3, 6, 4), [bond] * 3, pattern="+--", dq=-1)
    request.cls.deg = deg = rand((5, 3, 6), [bond] * 3, pattern="+-+", dq=-2)

    request.cls.Tabc = Tabc = FermionTensor(
        abc, inds=["a", "b", "c"], tags=["abc"]
    )
    request.cls.Tega = Tega = FermionTensor(
        ega, inds=["e", "g", "a"], tags=["ega"]
    )
    request.cls.Tbcd = Tbcd = FermionTensor(
        bcd, inds=["b", "c", "d"], tags=["bcd"]
    )
    request.cls.Tdeg = Tdeg = FermionTensor(
        deg, inds=["d", "e", "g"], tags=["deg"]
    )
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((2, 5), [bond] * 2, pattern="+-", dq=0)
    bc = rand((5, 4), [bond] * 2, pattern="++", dq=1)

    Tab = FermionTensor(ab, inds=["a", "b"], tags=["ab"])
    Tbc = FermionTensor(bc, inds=["b", "c"], tags=["bc"])
    Tab1 = FermionTensor(ab.dagger, inds=["b1", "a"], tags=["ab1"])
    Tbc1 = FermionTensor(bc.dagger, inds=["c", "b1"], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield


@pytest.fixture(scope="class")
def z4setup(request):
    bond = (0, 1, 2, 3)
    set_options(symmetry="z4")
    request.cls.abc = abc = rand((4, 2, 3), [bond] * 3, pattern="+--", dq=1)
    request.cls.bcd = bcd = rand((2, 3, 5), [bond] * 3, pattern="++-", dq=2)
    request.cls.ega = ega = rand((3, 6, 4), [bond] * 3, pattern="+--", dq=0)
    request.cls.deg = deg = rand((5, 3, 6), [bond] * 3, pattern="+-+", dq=1)

    request.cls.Tabc = Tabc = FermionTensor(
        abc, inds=["a", "b", "c"], tags=["abc"]
    )
    request.cls.Tega = Tega = FermionTensor(
        ega, inds=["e", "g", "a"], tags=["ega"]
    )
    request.cls.Tbcd = Tbcd = FermionTensor(
        bcd, inds=["b", "c", "d"], tags=["bcd"]
    )
    request.cls.Tdeg = Tdeg = FermionTensor(
        deg, inds=["d", "e", "g"], tags=["deg"]
    )
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((2, 5), [bond] * 2, pattern="+-", dq=0)
    bc = rand((5, 4), [bond] * 2, pattern="++", dq=1)

    Tab = FermionTensor(ab, inds=["a", "b"], tags=["ab"])
    Tbc = FermionTensor(bc, inds=["b", "c"], tags=["bc"])
    Tab1 = FermionTensor(ab.dagger, inds=["b1", "a"], tags=["ab1"])
    Tbc1 = FermionTensor(bc.dagger, inds=["c", "b1"], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield


@pytest.fixture(scope="class")
def z2setup(request):
    bond = (0, 1)
    set_options(symmetry="z2")
    request.cls.abc = abc = rand((4, 2, 3), [bond] * 3, pattern="+--", dq=0)
    request.cls.bcd = bcd = rand((2, 3, 5), [bond] * 3, pattern="++-", dq=1)
    request.cls.ega = ega = rand((3, 6, 4), [bond] * 3, pattern="+--", dq=1)
    request.cls.deg = deg = rand((5, 3, 6), [bond] * 3, pattern="+-+", dq=0)

    request.cls.Tabc = Tabc = FermionTensor(
        abc, inds=["a", "b", "c"], tags=["abc"]
    )
    request.cls.Tega = Tega = FermionTensor(
        ega, inds=["e", "g", "a"], tags=["ega"]
    )
    request.cls.Tbcd = Tbcd = FermionTensor(
        bcd, inds=["b", "c", "d"], tags=["bcd"]
    )
    request.cls.Tdeg = Tdeg = FermionTensor(
        deg, inds=["d", "e", "g"], tags=["deg"]
    )
    request.cls.tn = FermionTensorNetwork((Tabc, Tega, Tbcd, Tdeg))

    ab = rand((2, 5), [bond] * 2, pattern="+-", dq=0)
    bc = rand((5, 4), [bond] * 2, pattern="++", dq=1)

    Tab = FermionTensor(ab, inds=["a", "b"], tags=["ab"])
    Tbc = FermionTensor(bc, inds=["b", "c"], tags=["bc"])
    Tab1 = FermionTensor(ab.dagger, inds=["b1", "a"], tags=["ab1"])
    Tbc1 = FermionTensor(bc.dagger, inds=["c", "b1"], tags=["bc1"])
    request.cls.norm = FermionTensorNetwork((Tab, Tbc, Tbc1, Tab1))
    yield


@pytest.mark.usefixtures("u11setup")
class TestU11:
    def test_backend(self):
        Tegbc = tensor_contract(
            self.Tega, self.Tabc, output_inds=("e", "g", "b", "c")
        )
        egbc = np.tensordot(self.ega, self.abc, axes=[(2,), (0,)])
        err = (egbc - Tegbc.data).norm()
        assert err < 1e-10

    def test_contract_between(self):
        tn1 = self.tn.copy()
        tn1.contract_between("abc", "ega")
        Tegbc = tn1["abc"].transpose("e", "g", "b", "c")
        egbc = np.tensordot(self.ega, self.abc, axes=[(2,), (0,)])
        err = (egbc - Tegbc.data).norm()
        assert err < 1e-10

    def test_contract_all(self):
        result = self.tn.contract(all)
        egbc = np.tensordot(self.ega, self.abc, axes=[(2,), (0,)])
        deg1 = np.tensordot(self.bcd, egbc, axes=[(0, 1), (2, 3)])
        ref_val = np.tensordot(
            self.deg,
            deg1,
            axes=[
                (0, 1, 2),
            ]
            * 2,
        )
        err = abs(result - ref_val)
        assert err < 1e-10

    def test_contract_ind(self):
        tn1 = self.tn.copy()
        tn1.contract_ind("d")
        out = tn1["deg"].transpose("e", "g", "b", "c")
        egbc = np.tensordot(self.deg, self.bcd, axes=[(0,), (2,)])
        err = (egbc - out.data).norm()
        assert err < 1e-10

    def test_balance_bonds(self):
        norm = self.norm
        exact = norm.contract(all, optimize="auto-hq")
        norm1 = norm.balance_bonds()
        exact_bb = norm1.contract(all, optimize="auto-hq")
        assert exact_bb == pytest.approx(exact, rel=1e-2)
        for tid, tsr in norm.tensor_map.items():
            tsr1 = norm1.tensor_map[tid]
            assert (tsr1 - tsr).data.norm() > 1e-10

    def test_equlaize_norm(self):
        norm = self.norm
        exact = norm.contract(all, optimize="auto-hq")
        norm1 = norm.equalize_norms()
        exact_en = norm1.contract(all, optimize="auto-hq")
        assert exact_en == pytest.approx(exact, rel=1e-2)
        ref1 = list(norm1.tensor_map.values())[0].norm()
        for tid, tsr in norm.tensor_map.items():
            tsr1 = norm1.tensor_map[tid]
            assert tsr1.norm() == pytest.approx(ref1, rel=1e-2)

    def test_split(self):
        Tegbc = tensor_contract(
            self.Tabc, self.Tega, output_inds=("e", "g", "b", "c")
        )
        u, s, v = Tegbc.split(
            ("e", "b"), method="svd", absorb=None, get="tensors"
        )
        out = tensor_contract(u, s, v, output_inds=Tegbc.inds)
        assert (out.data - Tegbc.data).norm() < 1e-10

        for absorb in ["left", "right"]:
            for method in ["qr", "svd"]:
                l, r = Tegbc.split(
                    ("g", "c"), method=method, absorb=absorb, get="tensors"
                )
                out = tensor_contract(l, r, output_inds=Tegbc.inds)
                assert (out.data - Tegbc.data).norm() < 1e-10


@pytest.mark.usefixtures("u1setup")
class TestU1(TestU11):
    pass


@pytest.mark.usefixtures("z4setup")
class TestZ4(TestU11):
    pass


@pytest.mark.usefixtures("z22setup")
class TestZ22(TestU11):
    pass


@pytest.mark.usefixtures("z2setup")
class TestZ2(TestU11):
    pass
