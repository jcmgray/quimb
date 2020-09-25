"""This script illustrates how quimb's various MPI modes work.

It can be run as:

python ex_mpi_modes_expm_evo.py
quimb-mpi-python ex_mpi_modes_expm_evo.py
quimb-mpi-python --syncro ex_mpi_modes_expm_evo.py

And will display slightly different output for each that explains the
three modes.
"""
import quimb as qu
from mpi4py import MPI

# Get some MPI information
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
print(f"I am worker {rank} of total {size} running main script...")


# setup a verbose version of the ham_heis constructor, and make it Lazy
n = 18
shape = (2**n, 2**n)

# this makes the function print some information when called
#     - in order to be pickled is has to be located in the main package
ham_heis_verbose = qu.utils.Verbosify(qu.ham_heis,
                                      highlight='ownership', mpi=True)

H = qu.Lazy(ham_heis_verbose, n=n, sparse=True, shape=shape)

# random initial state
#     - must make sure all processes have the same seed to be pure
psi0 = qu.rand_ket(2**n, seed=42)

# evolve the system, processes split 'hard' work (slepc computations)
#     - should see each worker gets given a different ownership rows
#     - but all end up with the results.
evo = qu.Evolution(psi0, H, method='expm', expm_backend='slepc')
evo.update_to(5)


print(f"{rank}: I have final state norm {qu.expec(evo.pt, evo.pt)}")

# Now lets demonstrate using the MPI pool construct
pool = qu.get_mpi_pool()

dims = [2] * n
bsz = 5
logneg_subsys_verbose = qu.utils.Verbosify(qu.logneg_subsys,
                                           highlight='sysb', mpi=True)

# each process only computes its own fraction of these
#     - should see each process calls logneg with different ``sysb``.
fs = [pool.submit(logneg_subsys_verbose, evo.pt, dims=dims,
                  sysa=range(0, bsz), sysb=range(i, i + bsz))
      for i in range(bsz, n - bsz)]

# but then the results are comminucated to everyone
rs = [f.result() for f in fs]

print(f"{rank}: I have logneg results: {rs}")
