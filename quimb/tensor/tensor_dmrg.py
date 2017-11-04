"""DMRG-like variational algorithms, but in tensor network language.
"""
import scipy.sparse.linalg as spla

from ..utils import progbar
from ..accel import prod
from .tensor_core import Tensor, TensorNetwork, tensor_contract
from .tensor_gen import MPS_rand
from .tensor_1d import align_inner


class EffectiveHamLinearOperator(spla.LinearOperator):
    """Get a linear operator - something that replicates the matrix-vector
    operation - for an arbitrary *uncontracted* hamiltonian operator, e.g:

         / | | \
        L--H-H--R
         \ | | /

    This can then be supplied to scipy's sparse linear algebra routines.

    Parameters
    ----------
    TN_ham : TensorNetwork
        A representation of the hamiltonian
    upper_inds : sequence of hashable
        The upper inds of the effective hamiltonian network.
    lower_inds : sequence of hashable
        The lower inds of the effective hamiltonian network. These should be
        ordered the same way as ``upper_inds``.
    dims : tuple of int
        The dimensions corresponding to the inds.
    """

    def __init__(self, TN_ham, upper_inds, lower_inds, dims):
        self.eff_ham_tensors = TN_ham["__ham__"]
        self.upper_inds = upper_inds
        self.lower_inds = lower_inds
        self.dims = dims
        self.d = prod(dims)
        super().__init__(dtype=complex, shape=(self.d, self.d))

    def _matvec(self, vec):
        v = Tensor(vec.reshape(*self.dims), inds=self.upper_inds)
        v_out = tensor_contract(*self.eff_ham_tensors, v,
                                output_inds=self.lower_inds).data
        return v_out.reshape(*vec.shape)


class DMRG1:
    """Single site, fixed bond-dimension variational groundstate search.

    Parameters
    ----------
    ham : MatrixProductOperator
        The hamiltonian in MPO form.
    bond_dim : int
        The bond-dimension of the MPS to optimize.
    """

    def __init__(self, ham, bond_dim):
        self.n = ham.nsites
        self.bond_dim = bond_dim
        self.k = MPS_rand(self.n, bond_dim)
        self.b = self.k.H

        # Tag the various bits for contraction.
        ham.add_tag("__ham__")

        # Line up and overlap
        align_inner(self.k, self.b, ham)

        # want to contract this multiple times while
        #   manipulating k -> make virtual
        self.TN_energy = TensorNetwork([self.b, ham, self.k], virtual=True)
        self.energies = []
        self.site_id = ham.site_tag_id

    def update_with_eff_gs(self, eff_ham, i, dense=False):
        """Find the effective tensor groundstate of:


                      /|\
            >->->->->- | -<-<-<-<-<-<-<-<          |||
            | | | | |  |  | | | | | | | |         / | \
            H-H-H-H-H--H--H-H-H-H-H-H-H-H   =    L--H--R
            | | | | | i|  | | | | | | | |         \i| /
            >->->->->- | -<-<-<-<-<-<-<-<          |||
                      \|/

        And insert it back into the states ``k`` and ``b``, and thus
        ``TN_energy``.
        """
        if dense:
            # contract remaining hamiltonian and get its dense representation
            eff_ham = (eff_ham ^ '__ham__')['__ham__']
            eff_ham.fuse((('lower', self.b.site[i].inds),
                          ('upper', self.k.site[i].inds)), inplace=True)
            op = eff_ham.data
        else:
            op = EffectiveHamLinearOperator(eff_ham, dims=self.k.site[i].shape,
                                            upper_inds=self.k.site[i].inds,
                                            lower_inds=self.b.site[i].inds)

        eff_e, eff_gs = spla.eigs(op, k=1, which='SR')
        self.k.site[i].data = eff_gs
        self.b.site[i].data = eff_gs.conj()
        return eff_e

    def sweep_right(self, canonize=True, eff_ham_dense=False):
        """Perform a sweep of optimizations rightwards:

              optimize -->
                .
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->-o-<-<-<-<-<-<-<-<-<-<-<-<-<

        After the sweep the state is left canonized.

        Parameters
        ----------
        canonize : bool, optional
            Right canonize first. Set to False if already right-canonized.
        eff_ham_dense : bool, optional
            Solve the inner eigensystem using a dense representation of the
            effective hamiltonian. Can be quicker for small bond_dim.
        """
        if canonize:
            self.k.right_canonize(bra=self.b)

        # build right envs iteratively, saving each step to be re-used
        envs = {self.n - 1: self.TN_energy.copy(virtual=True)}
        for i in reversed(range(0, self.n - 1)):
            env = envs[i + 1].copy(virtual=True)
            # contract the previous env with one more site
            env ^= slice(min(self.n - 1, i + 2), i)
            envs[i] = env

        for i in range(0, self.n):
            eff_ham = envs[i]
            if i >= 2:
                # replace left env with new effective left env
                for j in range(i - 1):
                    del eff_ham.site[j]
                eff_ham |= envs[i - 1].site[i - 2]

            if i >= 1:
                # contract left env with new minimized, canonized site
                eff_ham ^= slice(max(0, i - 2), i)

            eff_e = self.update_with_eff_gs(eff_ham, i, dense=eff_ham_dense)

            if i < self.n - 1:
                self.k.left_canonize_site(i, bra=self.b)

        return eff_e

    def sweep_left(self, canonize=True, eff_ham_dense=False):
        """Perform a sweep of optimizations leftwards:

                            <-- optimize
                                      .
            >->->->->->->->->->->->->-o-<-<
            | | | | | | | | | | | | | | | |
            H-H-H-H-H-H-H-H-H-H-H-H-H-H-H-H
            | | | | | | | | | | | | | | | |
            >->->->->->->->->->->->->-o-<-<

        After the sweep the state is right canonized.

        Parameters
        ----------
        canonize : bool, optional
            Left canonize first. Set to False if already right-canonized.
        eff_ham_dense : bool, optional
            Solve the inner eigensystem using a dense representation of the
            effective hamiltonian. Can be quicker for small bond_dim.
        """
        if canonize:
            self.k.left_canonize(bra=self.b)

        # build left envs iteratively, saving each step to be re-used
        envs = {0: self.TN_energy.copy(virtual=True)}
        for i in range(1, self.n):
            env = envs[i - 1].copy(virtual=True)
            # contract the previous env with one more site
            env ^= slice(max(0, i - 2), i)
            envs[i] = env

        for i in reversed(range(0, self.n)):
            eff_ham = envs[i]
            if i <= self.n - 3:
                # replace right env with new effective right env
                for j in reversed(range(i + 2, self.n)):
                    del eff_ham.site[j]
                eff_ham |= envs[i + 1].site[i + 2]

            if i <= self.n - 2:
                # contract right env with new minimized, canonized site
                eff_ham ^= slice(min(self.n - 1, i + 2), i)

            eff_e = self.update_with_eff_gs(eff_ham, i, dense=eff_ham_dense)
            if i > 0:
                self.k.right_canonize_site(i, bra=self.b)

        return eff_e

    def solve(self, num_sweeps=4, eff_ham_dense="AUTO"):
        """Sweep a number of times.
        """
        # choose a rough value at which dense effective ham should not be used
        if eff_ham_dense == "AUTO":
            eff_ham_dense = self.bond_dim < 20

        for _ in progbar(range(num_sweeps)):
            self.energies.append(self.sweep_right(eff_ham_dense=eff_ham_dense))

        return self.energies[-1], self.k
