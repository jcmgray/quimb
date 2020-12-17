import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_2d import PEPS
import numpy as np
from quimb.tensor.fermion_2d import FPEPS
from quimb.tensor.fermion import _fetch_fermion_space, FermionTensorNetwork
from quimb.tensor.tensor_core import oset
from itertools import product
Lx = 3
Ly = 3
D = 2
np.random.seed(3)

def tensor_compress_bond(
    T1,
    T2,
    reduced=True,
    absorb='both',
    info=None,
    **compress_opts
):
    fs, (tid1, tid2) = _fetch_fermion_space(T1, T2, inplace=True)

    site1, site2 = fs[tid1][1], fs[tid2][1]

    if site1 < site2:
        Tl, Tr = T1, T2
        tidl, tidr = tid1, tid2
    else:
        Tl, Tr = T2, T1
        tidl, tidr = tid2, tid1

    left_inds = [ind for ind in Tl.inds if ind not in Tr.inds]
    right_inds = [ind for ind in Tr.inds if ind not in Tl.inds]

    out = fs._contract_pairs(tidl, tidr, direction="left")
    l, r = out.split(left_inds=left_inds, right_inds=right_inds, absorb=absorb, get="tensors", **compress_opts)
    return l, r

def get_err(max_bond=None):
    if max_bond is None: max_bond = 2*D**2


    psi = FPEPS.rand(Lx, Ly, bond_dim=D, seed=666)
    tsr1 = psi[0,0]
    tsr2 = psi[1,0]

    for x in range(Lx):
        psi.contract_between((0,x), (1,x))
    tsr1 = psi[0,0]
    tsr2 = psi[0,1]


    inds_contr = [i for i in tsr1.inds if i in tsr2.inds]
    outinds = [i for i in tsr1.inds if i not in tsr2.inds]
    idxa = [tsr1.inds.index(i) for i in inds_contr]
    idxb = [tsr2.inds.index(i) for i in inds_contr]

    out = np.tensordot(tsr1.data, tsr2.data, axes=(idxa, idxb))

    l, r = tensor_compress_bond(tsr1, tsr2, max_bond=max_bond)

    inds_contr = [i for i in l.inds if i in r.inds]
    outinds = [i for i in l.inds if i not in r.inds]
    idxa = [l.inds.index(i) for i in inds_contr]
    idxb = [r.inds.index(i) for i in inds_contr]
    fidx = [i for i in l.inds+r.inds if i not in inds_contr]

    out1 = np.tensordot(l.data, r.data, axes=(idxa, idxb))


    nblk = out.shapes.shape[0]

    err = []
    for i in range(nblk):
        dlt = np.sum(abs(out.q_labels[i] - out1.q_labels), axis=1)
        j = np.where(dlt==0)[0][0]
        ist, ied = out.idxs[i], out.idxs[i+1]
        jst, jed = out1.idxs[j], out1.idxs[j+1]
        err.append(max(abs(out.data[ist:ied]-out1.data[jst:jed])))
    return max(err)

def contract_all(tn):
    Lx, Ly = tn._Lx, tn._Ly
    nsite = Lx * Ly * 2
    fs = tn.fermion_space
    for x in range(nsite-1):
        out = fs._contract_pairs(0, 1)
    return out




psi = FPEPS.rand(Lx, Ly, bond_dim=D, seed=666)

ket = psi.copy()

layer_tags=('KET', 'BRA')

ket.add_tag(layer_tags[0])


bra = ket.H.retag_({layer_tags[0]: layer_tags[1]})
bra.mangle_inner_("*")
norm = bra & ket

norm_ur = norm.reorder_upward_column(layer_tags=layer_tags)
out = contract_all(norm_ur)
norm_dl = norm.reorder_downward_column(direction="left", layer_tags=layer_tags)
norm_rd = norm.reorder_right_row(direction="down",layer_tags=layer_tags)
norm_lu = norm.reorder_left_row(direction="up",layer_tags=layer_tags)




row_envs = norm.compute_row_environments(layer_tags=layer_tags)
print("TESTING ROW ENVIRONMENTS")
for ix in range(Lx):
    tmp = row_envs["below", ix].copy()
    tmp.add_tensor_network(row_envs["mid", ix])
    tmp.add_tensor_network(row_envs["above", ix])
    fs = tmp.fermion_space
    for i in range(len(fs.tensor_order.keys())-1):
        out = fs._contract_pairs(0,1)
    print("ROW%i env + mid: %.6f"%(ix, out))

col_envs = norm.compute_col_environments(layer_tags=layer_tags)
print("TESTING COL ENVIRONMENTS")
for ix in range(Ly):
    tmp = col_envs["left", ix].copy()
    tmp.add_tensor_network(col_envs["mid", ix])
    tmp.add_tensor_network(col_envs["right", ix])
    fs = tmp.fermion_space
    for i in range(len(fs.tensor_order.keys())-1):
        out = fs._contract_pairs(0,1)
    print("COL%i env + mid: %.6f"%(ix, out))

out = contract_all(norm)
print(out)
out_ur = contract_all(norm_ur)
print(out_ur)
out_dl = contract_all(norm_dl)
print(out_dl)
out_rd = contract_all(norm_rd)
print(out_rd)
out_lu = contract_all(norm_lu)
print(out_lu)
