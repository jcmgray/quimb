from pyblock3.algebra.fermion_symmetry import U11, U1, Z2, Z4
from pyblock3.algebra.symmetry import BondInfo
from pyblock3.algebra.fermion import eye, SparseFermionTensor
from pyblock3.algebra import fermion_setting as setting
from pyblock3.algebra import fermion_ops

DEFAULT_SYMMETRY = U11
symmetry_map = setting.symmetry_map

def set_symmetry(symmetry_string):
    global DEFAULT_SYMMETRY
    symmetry_string = symmetry_string.upper()
    if symmetry_string not in symmetry_map:
        raise KeyError("input symmetry %s not supported"%symmetry_string)
    DEFAULT_SYMMETRY = symmetry_map[symmetry_string]
    setting.set_symmetry(symmetry_string)

to_exponential = fermion_ops.get_flat_exponential
H1 = fermion_ops.H1
Hubbard = fermion_ops.Hubbard
onsite_U = fermion_ops.onsite_U
measure_SZ = fermion_ops.measure_SZ
ParticleNumber = fermion_ops.ParticleNumber
