import numpy as np

from .tensor_core import rand_uuid, Tensor
from .tensor_1d import MatrixProductState
from .tensor_gen import MPO_rand, MPO_zeros_like, randn


def construct_lanczos_tridiag_MPO(
        A,
        K,
        v0=None,
        initial_bond_dim=None,
        beta_tol=1e-6,
        max_bond=None,
        seed=None):
    """
    """
    if initial_bond_dim is None:
        initial_bond_dim = 2
    if max_bond is None:
        max_bond = 4

    if v0 is None:
        V = MPO_rand(A.nsites, initial_bond_dim, phys_dim=A.phys_dim(0))
    else:  # normalize
        V = v0 / (v0.H @ v0)**0.5
    Vm1 = MPO_zeros_like(V)

    alpha = np.zeros(K + 1)
    beta = np.zeros(K + 2)

    bsz = A.phys_dim()**A.nsites
    beta[1] = A.phys_dim()**A.nsites

    compress_kws = {'max_bond': max_bond, 'method': 'eig'}

    for j in range(1, K + 1):
        Vt = A.apply(V, compress=True, **compress_kws)
        Vt.add_MPO(-beta[j] * Vm1, inplace=True, compress=True, **compress_kws)
        alpha[j] = (V.H @ Vt).real
        Vt.add_MPO(-alpha[j] * V, inplace=True, compress=True, **compress_kws)
        beta[j + 1] = (Vt.H @ Vt)**0.5

        # check for convergence
        if abs(beta[j + 1]) < beta_tol:
            yield alpha[1:j + 1], beta[2:j + 2], beta[1]**2 / bsz
            break

        Vm1 = V.copy()
        V = Vt / beta[j + 1]

        if j > 3:
            yield (np.copy(alpha[1:j + 1]),
                   np.copy(beta[2:j + 2]),
                   np.copy(beta[1])**2 / bsz)


class EMPS(MatrixProductState):
    """Environment matrix product state::

         -------------E--------------
        /   sysa     / \    sysb     \
        o-o-o-o-o-o-o   o-o-o-o-o-o-o-o
        | | | | | | |   | | | | | | | |


    """

    def __init__(self, arrays, env=None, sysa=None, sysb=None,
                 nsites=None, **mps_opts):

        # short-circuit for copying EMPSs
        if isinstance(arrays, EMPS):
            super().__init__(arrays)
            self.sysa, self.sysb = list(arrays.sysa), list(arrays.sysb)
            return

        if any(x is None for x in [env, sysa, sysb, nsites]):
            raise ValueError("[env, sysa, sysb, nsites] are all required.")

        self.sysa, self.sysb = sorted(sysa), sorted(sysb)

        super().__init__(
            arrays, sites=(*self.sysa, *self.sysb), nsites=nsites, **mps_opts)

        # generate the four indices for the env
        ai, af, bi, bf = rand_uuid(), rand_uuid(), rand_uuid(), rand_uuid()

        # cut bond between sysa start and sysb end
        t1, t2 = self.site[self.sysa[0]], self.site[self.sysb[-1]]
        old_ind, = t1.shared_inds(t2)
        t1.reindex({old_ind: ai}, inplace=True)
        t2.reindex({old_ind: bf}, inplace=True)

        # cut bond between sysa end and sysb start
        t1, t2 = self.site[self.sysa[-1]], self.site[self.sysb[0]]
        old_ind, = t1.shared_inds(t2)
        t1.reindex({old_ind: af}, inplace=True)
        t2.reindex({old_ind: bi}, inplace=True)

        # add the env in that gap
        self.add_tensor(Tensor(env, inds=(ai, af, bi, bf), tags={'_ENV'}))

        # tag the subsystems
        self.add_tag('_SYSA', where=map(self.site_tag, sysa), mode='any')
        self.add_tag('_SYSB', where=map(self.site_tag, sysb), mode='any')

    def contract_structured_all(self, old, inplace=False):
        new = old if inplace else old.copy()
        new ^= slice(self.sysa[0], self.sysa[-1])
        new ^= slice(self.sysb[0], self.sysb[-1])
        return new.contract_tags(...)

    def __mul__(self, x):
        pass

    def __div__(self, x):
        pass

    def add_EMPS(self, other):
        pass


def EMPS_rand_state(sysa, sysb, nsites, bond_dim, phys_dim=2,
                    normalize=True, dtype=complex, **mps_opts):
    """
    """

    # physical sites
    arrays = (randn((bond_dim, bond_dim, phys_dim), dtype=dtype)
              for _ in range(len(sysa) + len(sysb)))

    # environment tensor
    env = randn((bond_dim,) * 4, dtype=dtype)

    emps = EMPS(arrays, env=env, sysa=sysa, sysb=sysb,
                nsites=nsites, **mps_opts)

    if normalize:
        emps['_ENV'] /= (emps.H @ emps)**0.5

    return emps


def EMPS_zeros(sysa, sysb, nsites, phys_dim=2, **mps_opts):
    pass


class MPSPTPT:
    """Turn a MPS into an effective operator by partially tracing and partially
    transposing like so::

                 sysa            sysb
        >->->-A-A-A-A-A-o-o-o-o-B-B-B-B-<-<-<-<-<
        | | | | | | | | | | | | | | | | | | | | |

                         ->

               | | | | |     | | | |
               A-A-A-A-A\   /B-B-B-B
               |         E=x       |
               A-A-A-A-A/   \B-B-B-B
               | | | | |     | | | |
    """

    def __init__(self, mps, sysa, sysb, upper_ind_id='b{}'):
        n = mps.nsites

        # parse sysa and sysb
        self.sysa_i, self.sysa_f = min(sysa), max(sysa) + 1
        self.sysb_i, self.sysb_f = min(sysb), max(sysb) + 1

        if len(sysa) != len(range(self.sysa_i, self.sysa_f)):
            raise ValueError("``sysa`` must be contiguous.")

        if len(sysb) != len(range(self.sysb_i, self.sysb_f)):
            raise ValueError("``sysb`` must be contiguous.")

        ket = mps.copy()
        for i in sysa:
            ket.add_tag('sysa', where=ket.site_tag(i))

        for i in sysb:
            ket.add_tag('sysb', where=ket.site_tag(i))

        # mixed canonize
        ket.left_canonize(stop=self.sysa_i)
        ket.right_canonize(stop=self.sysb_f)

        # make bra and reindex non traced out sites and do partial transpose
        bra = ket.H
        bra.reindex({bra.site_ind(i): upper_ind_id.format(i)
                     for i in sysa}, inplace=True)
        ket.reindex({ket.site_ind(i): upper_ind_id.format(i)
                     for i in sysb}, inplace=True)

        ket.add_tag('_KET')
        bra.add_tag('_BRA')
        self.X = ket & bra

        # replace left and right envs with identity since canonized
        le = [self.X.structure.format(i) for i in range(0, self.sysa_i)]
        re = [self.X.structure.format(i) for i in range(self.sysb_f, n)]
        self.X.replace_with_identity(le, inplace=True)
        self.X.replace_with_identity(re, inplace=True)

        # contract middle env
        self.X ^= slice(self.sysa_f, self.sysb_i)

    def apply(self, other):
        """Apply this operator to a vector.
        """
        v = other.copy()
        v.add_tag('_VEC')
