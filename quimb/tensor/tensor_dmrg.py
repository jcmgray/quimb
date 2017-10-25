"""DMRG-like variational algorithms, but in tensor network language.
"""
import scipy.sparse.linalg as spla

from ..utils import progbar
from ..accel import prod
from .tensor_core import Tensor, tensor_contract
from .tensor_gen import MPS_rand
from .tensor_1d import align_inner


class EffectiveHamLinearOperator(spla.LinearOperator):

    def __init__(self, eff_ham, upper_inds, lower_inds, dims):
        self.eff_ham_ts = eff_ham["__ham__"]
        self.upper_inds = upper_inds
        self.lower_inds = lower_inds
        self.dims = dims
        self.d = prod(dims)
        super().__init__(dtype=complex, shape=(self.d, self.d))

    def _matvec(self, vec):
        k = Tensor(vec.reshape(*self.dims), inds=self.upper_inds)
        k_out = tensor_contract(*self.eff_ham_ts, k,
                                output_inds=self.lower_inds).data
        return k_out.reshape(*vec.shape)


def update_with_eff_gs(energy_tn, k, b, i, dense=False):
    """Find the effective tensor groundstate of:

                  /|\
        o-o-o-o-o- | -o-o-o-o-o-o-o-o
        | | | | |  |  | | | | | | | |
        O-O-O-O-O--O--O-O-O-O-O-O-O-O
        | | | | | i|  | | | | | | | |
        o-o-o-o-o- | -o-o-o-o-o-o-o-o
                  \|/

    And insert it back into the states ``k`` and ``b``, and thus ``energy_tn``.
    """
    # contract left and right environments
    eff_ham = energy_tn ^ slice(0, i) ^ slice(..., i)

    if dense:
        # also contract remaining hamiltonian and get its dense representation
        eff_ham = (eff_ham ^ '__ham__')['__ham__']
        eff_ham.fuse((('lower', b.site[i].inds),
                      ('upper', k.site[i].inds)), inplace=True)
        eff_ham = eff_ham.data
    else:
        eff_ham = EffectiveHamLinearOperator(eff_ham, dims=k.site[i].shape,
                                             upper_inds=k.site[i].inds,
                                             lower_inds=b.site[i].inds)

    eff_e, eff_gs = spla.eigs(eff_ham, k=1, which='SR')
    k.site[i].data = eff_gs
    b.site[i].data = eff_gs.conj()
    return eff_e


def dmrg1_sweep_right(energy_tn, k, b, canonize=True, eff_ham_dense=False):
    """
    """
    if canonize:
        k.right_canonize(bra=b)

    for i in range(0, k.nsites):
        eff_e = update_with_eff_gs(energy_tn, k, b, i, dense=eff_ham_dense)
        if i < k.nsites - 1:
            k.left_canonize_site(i, bra=b)

    return eff_e


def dmrg1_sweep_left(energy_tn, k, b, canonize=True, eff_ham_dense=False):
    """
    """
    if canonize:
        k.left_canonize(bra=b)

    for i in reversed(range(0, k.nsites)):
        eff_e = update_with_eff_gs(energy_tn, k, b, i, dense=eff_ham_dense)
        if i > 0:
            k.right_canonize_site(i, bra=b)

    return eff_e


def dmrg1(ham, bond_dim, num_sweeps=4, eff_ham_dense="AUTO"):
    """
    """
    k = MPS_rand(ham.nsites, bond_dim)
    b = k.H
    ham.add_tag("__ham__")
    k.add_tag("__ket__")
    b.add_tag("__bra__")

    align_inner(k, b, ham)

    energy_tn = b & ham & k

    if eff_ham_dense == "AUTO":
        eff_ham_dense = bond_dim < 20

    for _ in progbar(range(num_sweeps)):
        eff_e = dmrg1_sweep_right(energy_tn, k, b, eff_ham_dense=eff_ham_dense)

    ham.drop_tags("__ham__")

    return eff_e, k
