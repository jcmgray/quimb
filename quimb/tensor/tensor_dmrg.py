"""DMRG-like variational algorithms, but in tensor network language.
"""
import scipy.sparse.linalg as spla
from .tensor_gen import MPS_rand
from .tensor_1d import align_inner


def update_with_eff_gs(energy_tn, k, b, i):
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
    eff_ham = (energy_tn ^ slice(0, i) ^ slice(..., i) ^ '__ham__')['__ham__']
    eff_ham.fuse((('lower', b.site[i].inds),
                  ('upper', k.site[i].inds)), inplace=True)
    eff_e, eff_gs = spla.eigs(eff_ham.data, k=1)
    k.site[i].data = eff_gs
    b.site[i].data = eff_gs.conj()
    return eff_e


def dmrg1_sweep_right(energy_tn, k, b, canonize=True):
    """
    """
    if canonize:
        k.right_canonize(bra=b)

    for i in range(0, k.nsites):
        eff_e = update_with_eff_gs(energy_tn, k, b, i)
        if i < k.nsites - 1:
            k.left_canonize_site(i, bra=b)

    return eff_e


def dmrg1_sweep_left(energy_tn, k, b, canonize=True):
    """
    """
    if canonize:
        k.left_canonize(bra=b)

    for i in reversed(range(0, k.nsites)):
        eff_e = update_with_eff_gs(energy_tn, k, b, i)
        if i > 0:
            k.right_canonize_site(i, bra=b)

    return eff_e


def dmrg1(ham, bond_dim, num_sweeps=4):
    """
    """
    k = MPS_rand(ham.nsites, bond_dim)
    b = k.H
    ham.add_tag("__ham__")
    k.add_tag("__ket__")
    b.add_tag("__bra__")

    align_inner(k, b, ham)

    energy_tn = (b & ham & k)

    for _ in range(num_sweeps):
        eff_e = dmrg1_sweep_right(energy_tn, k, b)

    ham.remove_tag("__ham__")

    return eff_e, k
