import sys
from pyblock3.algebra.fermion_symmetry import U11, U1, Z2, Z4, Z22
from pyblock3.algebra.symmetry import BondInfo
from pyblock3.algebra.fermion import eye, SparseFermionTensor
from pyblock3.algebra import fermion_setting as setting
from pyblock3.algebra import fermion_ops

this = sys.modules[__name__]
this.DEFAULT_SYMMETRY = "U1"
this.USE_CPP = True
this.USE_FERMION = True
symmetry_map = setting.symmetry_map

def set_symmetry(symmetry):
    symmetry = symmetry.upper()
    if symmetry not in symmetry_map:
        raise KeyError("input symmetry %s not supported"%symmetry)
    this.DEFAULT_SYMMETRY = symmetry
    setting.set_symmetry(symmetry)

def set_backend(use_cpp):
    this.USE_CPP = use_cpp
    setting.set_flat(use_cpp)

def set_fermion(use_fermion):
    this.USE_FERMION = use_fermion
    setting.set_fermion(use_fermion)

def set_options(**kwargs):
    symmetry = kwargs.pop("symmetry", this.DEFAULT_SYMMETRY)
    use_fermion = kwargs.pop("fermion", this.USE_FERMION)
    use_cpp = kwargs.pop("use_cpp", this.USE_CPP)
    set_symmetry(symmetry)
    set_fermion(use_fermion)
    set_backend(use_cpp)

def dispatch_settings(*keys):
    dict = {"symmetry": "DEFAULT_SYMMETRY",
            "fermion": "USE_FERMION",
            "use_cpp": "USE_CPP"}
    _settings = []
    for ikey in keys:
        if ikey not in dict:
            raise KeyError("%s not a valid backend setting"%ikey)
        _settings.append(getattr(this, dict[ikey]))
    if len(_settings) == 1:
        _settings = _settings[0]
    return _settings

def get_symmetry():
    return setting.symmetry_map[this.DEFAULT_SYMMETRY]

to_exponential = fermion_ops.get_exponential
H1 = fermion_ops.H1
Hubbard = fermion_ops.Hubbard
onsite_U = fermion_ops.onsite_U
measure_SZ = fermion_ops.measure_SZ
ParticleNumber = fermion_ops.ParticleNumber
