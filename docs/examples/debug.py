import numpy as np
import itertools
from quimb.tensor.fermion_2d_tebd import Hubbard2D, SimpleUpdate
from pyblock3.algebra import fermion_operators as ops

t=1
u=4
Lx = 2
Ly = 2
mu = -0.9
mu = 0
Ham = Hubbard2D(t, u, Lx, Ly, mu=mu)
#efci = -5.702748483462062

state_array = np.zeros([Lx,Ly])
#state_array[0,0] = state_array[2,0] = 1
#`state_array[1,0] = state_array[3,0] = 2

state_array[0,0] = state_array[1,1] = 1
state_array[0,1] = state_array[1,0] = 2
from quimb.tensor.fermion_2d import gen_mf_peps

psi = gen_mf_peps(state_array) # this is now a 2d mean field PEPS


sz = ops.measure_sz()
nop = ops.count_n()

sz_ops = {(ix,iy): sz for ix, iy in itertools.product(range(Lx), range(Ly))}
n_ops = {(ix,iy): nop for ix, iy in itertools.product(range(Lx), range(Ly))}


book = {(0,0):"0", (0,1):"+-", (1,0):"+", (1,1):"-"}

def print_block(blk):
    qlab= [iq.n for iq in blk.q_labels]
    ind = np.where(abs(np.asarray(blk)) > 1e-20)
    need_print = False
    for ixs in zip(*ind):
        val = np.asarray(blk)[ixs]
        desc = "|"
        for ix, s in enumerate(ixs):
            desc += book[(qlab[ix], s)]
            if ix != len(ixs)-1:
                desc += ","
            else:
                desc += ">"
            
        
        if (desc.count("+"), desc.count("-")) != (2,2):
            desc = str(val) + desc
            print(desc)
            need_print = True
    return need_print


ket = psi.contract(all)
tsr = ket.data.to_sparse()
for iblk in tsr:
    print(np.asarray(iblk))
   
from pyblock3.algebra.fermion import SparseFermionTensor
np.random.seed(3)

            
def callback(su):
    psi1 = su.get_state()
    #for itsr in psi1:
     #   itsr.data.data[abs(itsr.data.data)<1e-9] =0
    state = psi1.contract(all)
    tsr = state.data.to_sparse()
    
    need_print = False
    for blk in tsr.blocks:
        need_print = need_print or print_block(blk)
    if need_print:
        print("Ending Cycle%i"%su._n)

            
su = SimpleUpdate(
    psi,
    Ham,
    chi=128,  # boundary contraction bond dim for computing energy
    D = 4,
    compute_energy_every=100,
    compute_energy_per_site=False,
    keep_best=True,
    ordering = 'sort',
    gauge_smudge = 1e-6,
    callback=callback#,
    #gate_opts = {'cutoff': 1e-6}
)
tau = 0.01
su.evolve(2, tau=tau)

second_dense = min(Lx, Ly)>1
sz_expecs = su.get_state().compute_local_expectation(sz_ops, return_all=True, second_dense=second_dense,normalized=True)
n_expecs = su.get_state().compute_local_expectation(n_ops, return_all=True, second_dense=second_dense,normalized=True)

print(su.get_state()[0,0].shape)
for ix, iy in itertools.product(range(Lx), range(Ly)):
    print("(%i, %i): SZ=%.2f, N=%.2f"%(ix,iy,sz_expecs[(ix,iy)][0]/sz_expecs[(ix,iy)][1], n_expecs[(ix,iy)][0]/n_expecs[(ix,iy)][1]))

norm = su.get_state().compute_norm()
print(norm)
