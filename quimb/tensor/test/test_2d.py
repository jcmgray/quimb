import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.tensor_2d import PEPS
import numpy as np
from quimb.tensor.fermion_2d import FPEPS
from quimb.tensor.fermion import _fetch_fermion_space
from quimb.tensor.tensor_core import oset
Lx = 2
Ly = 3
D = 4
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




psi = FPEPS.rand(Lx, Ly, bond_dim=D, seed=666)

ket = psi.copy()

layer_tags=('KET', 'BRA')

ket.add_tag(layer_tags[0])


bra = ket.H.retag_({layer_tags[0]: layer_tags[1]})
bra.mangle_inner_("*")

norm = bra & ket

def contract_all(tn):
    Lx, Ly = tn._Lx, tn._Ly
    for i in range(Lx):
        for j in range(Ly):
            x1, x2 = tn[i,j]
            tn.contract_between(x1.tags, x2.tags)
    for i in range(Lx):
        for j in range(Ly-1):
            x1 = tn[i,j]
            x2 = tn[i,j+1]
            tn.contract_between(x1.tags, x2.tags)
    for i in range(Lx-1):
        x1 = tn[i,0]
        x2 = tn[i+1,0]
        out = tn.contract_between(x1.tags, x2.tags)
    return out

def contract_left(tn):
    Lx, Ly = tn._Lx, tn._Ly
    for i in range(Lx):
        for j in range(Ly):
            x1, x2 = tn[i,j]
            tn.contract_between(x1.tags, x2.tags)
    for j in range(Ly):
        for i in range(Lx-1):
            x1 = tn[i,j]
            x2 = tn[i+1,j]
            tn.contract_between(x1.tags, x2.tags)
    for i in range(Ly-1):
        x1 = tn[0,i]
        x2 = tn[0,i+1]
        out = tn.contract_between(x1.tags, x2.tags)
    return out


fs = norm.fermion_space
norm1 = norm.copy()

size = Lx * Ly
for i in range(size):
    norm1.fermion_space.move(2*size-1, 2*i+1)

out1 = contract_all(norm1)

tag1 = norm.site_tag(0, 0)#, self.site_tag(i + 1, j)
tag2 = norm.site_tag(0, 1)

out2 = contract_left(norm)

print(out1, out2)
#print(hash(x[0]), hash(x[1]))
#norm.contract_boundary()

#x1, x2 = norm[0,0]
#norm.contract_between(x1.tags, x2.tags)

#tid1, = norm._get_tids_from_tags(x1.tags, which='all')
#print(tid1)
#tid2, = norm._get_tids_from_tags(x2.tags, which='all')
#print(tid2)
#norm.contract_between(tagged_tids[0], tagged_tids[1])
#x = norm[0,0]
#print(x[0].fermion_owner[2],x[1].fermion_owner[2])#, type(x[1]))

#print(tid1)
#print(tid2)
#print(hash(bra[0,0]), hash(ket[0,0]))
exit()
for x in range(Lx-1):
    for y in range(Ly-1):
        out = psi.contract_between((0,x), (1,x))
        print(x, y, "done")
