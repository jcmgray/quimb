import quimb as qu
import numpy as np
from quimb.tensor.tensor_core import rand_uuid
from quimb.tensor.tensor_2d import is_lone_coo
from quimb.tensor.fermion_2d import FermionTensorNetwork2DVector, gen_mf_peps, FPEPS
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
        id = where + ((1,1),)
    else:
        x_bsz = abs(where[1][0] - where[0][0]) + 1
        y_bsz = abs(where[1][1] - where[0][1]) + 1
        id = (where[0], ) + ((x_bsz, y_bsz),)
        if id not in env.keys():
            id = (where[1], ) + ((x_bsz, y_bsz),)
            if id not in env.keys():
                raise KeyError("env does not fit with operator")
    tn = env[id].copy()
    fs = tn.fermion_space
    ntsr = len(fs.tensor_order)
    for i in range(ntsr-2*ng, ntsr):
        tsr = fs[i][2]
        if layer_tags[0] in tsr.tags:
            tsr.reindex_(reindex_map)
    tn.add_tensor(newTG, virtual=True)
    out = tn.contract(all, optimize='auto-hq')
    return out

def contract_raw(psi, op, where):
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
    ket.reindex_(reindex_map)
    tn = ket & TG & bra
    out = tn.contract(all, optimize='auto-hq')
    return out

def contract_gate(psi, op, where):
    ket = psi.copy()
    layer_tags=('KET', 'BRA')

    ket.add_tag(layer_tags[0])
    bra = ket.H.retag_({layer_tags[0]: layer_tags[1]})
    bra.mangle_inner_("*")
    newket = ket.gate(op, where)
    tn =  newket & bra
    out = tn.contract(all, optimize='auto-hq')
    return out

Lx = Ly = 4
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

env = compute_env(psi, max_bond)

print("testing U")
for ix in range(Lx):
    for iy in range(Ly):
        where = (ix, iy)
        out = compute_expectation(env, psi, uop, where, max_bond)
        if state_array[ix,iy]==3:
            print(U==out)
        else:
            print(0.==out)

print("testing N")
for ix in range(Lx):
    for iy in range(Ly):
        where = (ix, iy)
        out = compute_expectation(env, psi, nop, where, max_bond)
        if state_array[ix,iy] ==0:
            print(0.==out)
        elif state_array[ix, iy] in [1, 2]:
            print(1.==out)
        else:
            print(2.==out)

print("testing sz")
for ix in range(Lx):
    for iy in range(Ly):
        where = (ix, iy)
        out = compute_expectation(env, psi, sz, where, max_bond)
        if state_array[ix,iy] in [0,3]:
            print(0.==out)
        elif state_array[ix, iy] ==1:
            print(.5==out)
        else:
            print(-.5==out)

print("testing hopping")

Lx = Ly = 3
psi = FPEPS.rand(Lx, Ly, 1, seed=33)

psi.view_as_(FermionTensorNetwork2DVector, like=psi)
where = ((1,1),(1,2))
out = contract_raw(psi, hop, where)
out1 = contract_gate(psi, hop, where)
env = compute_env(psi, max_bond, y_bsz=2)
out2 = compute_expectation(env, psi, hop, where, max_bond)
print(out, out1, out2)
