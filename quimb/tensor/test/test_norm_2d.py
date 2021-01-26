import quimb as qu
import numpy as np
from quimb.tensor.tensor_core import rand_uuid
from quimb.tensor.tensor_2d import is_lone_coo
from quimb.tensor.fermion_2d import FermionTensorNetwork2DVector, gen_mf_peps
from quimb.tensor.fermion import FermionTensorNetwork, FermionTensor, tensor_contract, FermionSpace
from quimb.tensor import fermion_ops as ops
import itertools

def compute_env(psi, max_bond, bra=None, x_bsz=1, y_bsz=1):
    ket = psi.copy()
    layer_tags=('KET', 'BRA')

    ket.add_tag(layer_tags[0])
    if bra is None: bra = ket.H
    bra = bra.retag_({layer_tags[0]: layer_tags[1]})
    bra.mangle_inner_("*")

    norm = ket & bra
    envs = norm._compute_plaquette_environments_row_first(x_bsz=x_bsz, y_bsz=y_bsz, layer_tags=layer_tags, max_bond=max_bond)
    return envs


def compute_expectation(env, psi, op, where, max_bond):
    ket = psi.copy()
    layer_tags=('KET', 'BRA')

    ket.add_tag(layer_tags[0])
    bra = ket.H.retag_({layer_tags[0]: layer_tags[1]})
    bra.mangle_inner_("*")

    if is_lone_coo(where):
        where = (where,)
    else:
        where = tuple(where)

    ng = len(where)
    site_ix = [bra.site_ind(i, j) for i, j in where]
    bnds = [rand_uuid() for _ in range(ng)]
    reindex_map = dict(zip(site_ix, bnds))

    TG = FermionTensor(op.copy(), inds=site_ix+bnds, left_inds=site_ix)
    newTG = bra.fermion_space.move_past(TG, (0, len(bra.fermion_space.tensor_order)))
    if ng==1:
        tn = env[where+((1,1),)].copy()
        fs = tn.fermion_space
        ntsr = len(fs.tensor_order)
        for i in range(ntsr-2*ng, ntsr):
            tsr = fs[i][2]
            if layer_tags[0] in tsr.tags:
                tsr.reindex_(reindex_map)
        tn.add_tensor(newTG, virtual=True)
        out = tn.contract(all, optimize='auto-hq')
        return out



Lx = Ly = 6
max_bond = 8
state_array = np.random.randint(0,4, [Lx, Ly])
psi = gen_mf_peps(state_array)
psi.view_as_(FermionTensorNetwork2DVector, like=psi)

U = 4.
t = 2.
uop = ops.onsite_u(U)
nop = ops.count_n()
sz = ops.measure_sz()
hop = ops.hopping(t)

out = psi.compute_norm(max_bond=max_bond)
print(out)
