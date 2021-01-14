import numpy as np
from quimb.tensor.fermion_2d import FPEPS, FermionTensorNetwork2DVector
from pyblock3.algebra.symmetry import SZ, BondInfo
from pyblock3.algebra.fermion import (SparseFermionTensor,
                                      FlatFermionTensor)
import time

def compute_norm(psi, max_bond):
    ket = psi.copy()
    layer_tags=('KET', 'BRA')

    ket.add_tag(layer_tags[0])

    bra = ket.H.retag_({layer_tags[0]: layer_tags[1]})
    bra.mangle_inner_("*")

    norm = bra & ket

    envs = norm._compute_plaquette_environments_col_first(x_bsz=1, y_bsz=1, layer_tags=layer_tags, max_bond=max_bond)
    for key, val in envs.items():
        fs = val.fermion_space
        ntsr = len(val.tensor_map)
        for i in range(ntsr-1):
            out = fs._contract_pairs(0,1)
        print("Col:", key, out)

    envs = norm._compute_plaquette_environments_row_first(x_bsz=1, y_bsz=1, layer_tags=layer_tags, max_bond=max_bond)
    for key, val in envs.items():
        fs = val.fermion_space
        ntsr = len(val.tensor_map)
        for i in range(ntsr-1):
            out = fs._contract_pairs(0,1)
        print("Row:",key, out)

Lx = Ly = 4
D = 2

np.random.seed(3)
infox = BondInfo({SZ(0,0,0):2, SZ(1,0,0): 2})
G = SparseFermionTensor.random((infox,infox,infox,infox), dq=SZ(1)).to_flat()
TG = FlatFermionTensor.eye(infox)
psi = FPEPS.rand(Lx, Ly, bond_dim=D, seed=666)
psi.view_as_(FermionTensorNetwork2DVector, like=psi)

max_bond=None
cutoff = 1e-10

site = ((0,0), (0,1))

t0 = time.time()
psi0 = psi.gate_(TG,((1,1)), contract=True, absorb=None, max_bond=max_bond, info=dict())
t1 = time.time()
psi1 = psi.gate(G, site, contract="split", absorb=None, max_bond=max_bond, cutoff=cutoff, info=dict())
t2 = time.time()
psi2 = psi.gate(G, site, contract="reduce-split", absorb=None, max_bond=max_bond, cutoff=cutoff, info=dict())
t3 = time.time()
print(t1-t0, t2-t1, t3-t2)

max_bond = 16
print("chi=%i"%max_bond)
print("Checking split gate norm")
compute_norm(psi1, max_bond)
print("Checking reduce-split gate norm")
compute_norm(psi2, max_bond)
