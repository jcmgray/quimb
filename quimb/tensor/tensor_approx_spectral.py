"""Approximating spectral functions with tensor networks.
"""

import numpy as np
import random

import quimb as qu

from .tensor_core import rand_uuid, Tensor
from .tensor_1d import MatrixProductState
from .tensor_gen import MPO_rand, MPO_zeros_like, randn


def construct_lanczos_tridiag_MPO(A, K, v0=None, initial_bond_dim=None,
                                  beta_tol=1e-6, max_bond=None, seed=False,
                                  v0_opts=None, k_min=10):
    """
    """
    if initial_bond_dim is None:
        initial_bond_dim = 8
    if max_bond is None:
        max_bond = 8

    if v0 is None:
        if seed:
            # needs to be truly random so MPI processes don't overlap
            qu.seed_rand(random.SystemRandom().randint(0, 2**32 - 1))

        V = MPO_rand(A.nsites, initial_bond_dim,
                     phys_dim=A.phys_dim(), dtype=A.dtype)
    else:  # normalize
        V = v0 / (v0.H @ v0)**0.5
    Vm1 = MPO_zeros_like(V)

    alpha = np.zeros(K + 1)
    beta = np.zeros(K + 2)

    bsz = A.phys_dim()**A.nsites
    beta[1] = bsz  # == sqrt(prod(A.shape))

    compress_kws = {'max_bond': max_bond, 'method': 'svd'}

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

        if j >= k_min:
            yield (np.copy(alpha[1:j + 1]),
                   np.copy(beta[2:j + 2]),
                   np.copy(beta[1])**2 / bsz)


class EEMPS(MatrixProductState):
    r"""Environment matrix product state::

         -------------E--------------
        /   sysa     / \    sysb     \
        o-o-o-o-o-o-o   o-o-o-o-o-o-o-o
        | | | | | | |   | | | | | | | |

    Used to estimate spectral quantities in subsystems of MPS.
    """

    def __init__(self, arrays, env=None, sysa=None, sysb=None,
                 nsites=None, **mps_opts):

        # short-circuit for copying EEMPSs
        if isinstance(arrays, EEMPS):
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

        # cut bond between sysa start and sysb end, and sysa end and sysb start
        self.cut_bond(self.sysa[0], self.sysb[-1], ai, bf)
        self.cut_bond(self.sysa[-1], self.sysb[0], af, bi)

        # add the env in that gap
        self |= Tensor(env, inds=(ai, af, bi, bf), tags={'_ENV'})

        # tag the subsystems
        self.add_tag('_SYSA', where=map(self.site_tag, sysa), which='any')
        self.add_tag('_SYSB', where=map(self.site_tag, sysb), which='any')

    _EXTRA_PROPS = ('sysa', 'sysb')

    def imprint(self, other):
        """Coerce
        """
        for p in EEMPS._EXTRA_PROPS:
            setattr(other, p, getattr(self, p))
        super().imprint(other)
        other.__class__ = EEMPS

    @staticmethod
    def contract_structured_all(old, inplace=False, **opts):
        new = old if inplace else old.copy()
        return new.contract_tags(all, **opts)

    def add_EEMPS(self, other, inplace=False):
        """Add another EEMPS.
        """
        self = self if inplace else self.copy()

        # for storing the bonds to the effective envinroment
        env_reindex_map = {}

        for i in self.sites:

            t1, t2 = self[i], other[i]

            # Check if need to use bonds to match indices
            if set(t1.inds) != set(t2.inds):
                reindex_map = {}
                edge = True

                # bond to left
                if i not in (self.sysa[0], self.sysb[0]):
                    reindex_map[other.bond(i - 1, i)] = self.bond(i - 1, i)
                else:
                    edge = True

                # bond to right
                if i not in (self.sysa[-1], self.sysb[-1]):
                    reindex_map[other.bond(i, i + 1)] = self.bond(i, i + 1)
                else:
                    edge = True

                # bond to eff env
                if edge:
                    s_env_bnds = t1.bonds(self['_ENV'])
                    o_env_bnds = t2.bonds(other['_ENV'])

                    for sb, ob in zip(s_env_bnds, o_env_bnds):
                        reindex_map[ob] = sb
                        env_reindex_map[ob] = sb

                t2 = t2.reindex(reindex_map)

            t1.direct_product(t2, inplace=True, sum_inds=self.site_ind(i))

        # treat effective envinroment tensor last
        t1, t2 = self['_ENV'], other['_ENV']
        t1.direct_product(t2.reindex(env_reindex_map), inplace=True)

        return self

    def __add__(self, other):
        """EEMPS addition.
        """
        return self.add_EEMPS(other, inplace=False)

    def __iadd__(self, other):
        """In-place EEMPS addition.
        """
        return self.add_EEMPS(other, inplace=True)

    def __sub__(self, other):
        """EEMPS subtraction.
        """
        return self.add_EEMPS(other * -1, inplace=False)

    def __isub__(self, other):
        """In-place EEMPS subtraction.
        """
        return self.add_EEMPS(other * -1, inplace=True)

    def to_dense(self):
        t = self.contract_tags(...)
        t.fuse_([('k', list(map(self.site_ind_id.format, self.sites)))])
        return qu.qarray(t.data.reshape(-1, 1))


def EEMPS_rand_state(sysa, sysb, nsites, bond_dim, phys_dim=2,
                     normalize=True, dtype=float, **mps_opts):
    """
    """
    # physical sites
    arrays = (randn((bond_dim, bond_dim, phys_dim), dtype=dtype)
              for _ in range(len(sysa) + len(sysb)))

    # environment tensor
    env = randn((bond_dim,) * 4, dtype=dtype)

    emps = EEMPS(arrays, env=env, sysa=sysa, sysb=sysb,
                 nsites=nsites, **mps_opts)

    if normalize:
        emps['_ENV'] /= (emps.H @ emps)**0.5

    return emps


def EEMPS_rand_like(other, bond_dim, **mps_opts):
    """Return a random EEMPS state with the same parameters as ``other`` and
    bond dimension ``bond_dim``.
    """
    return EEMPS_rand_state(sysa=other.sysa, sysb=other.sysb,
                            nsites=other.nsites, phys_dim=other.phys_dim(),
                            dtype=other.dtype, bond_dim=bond_dim, **mps_opts)


def EEMPS_zeros(sysa, sysb, nsites, phys_dim=2, dtype=float, **mps_opts):
    """Return the 'zero' EEMPS state.
    """
    # physical sites
    arrays = (np.zeros((1, 1, phys_dim), dtype=dtype)
              for _ in range(len(sysa) + len(sysb)))

    # environment tensor
    env = np.zeros((1,) * 4, dtype=dtype)

    return EEMPS(arrays, env=env, sysa=sysa, sysb=sysb,
                 nsites=nsites, **mps_opts)


def EEMPS_zeros_like(other, **mps_opts):
    """Return the 'zero' EEMPS state with the same parameters as ``other``.
    """
    return EEMPS_zeros(sysa=other.sysa, sysb=other.sysb, dtype=other.dtype,
                       phys_dim=other.phys_dim(), nsites=other.nsites,
                       **mps_opts)


class PTPTLazyMPS:
    r"""Turn a MPS into an effective operator by partially tracing and
    partially transposing like so::

                 sysa            sysb
        >->->-A-A-A-A-A-o-o-o-o-B-B-B-B-<-<-<-<-<
        | | | | | | | | | | | | | | | | | | | | |

                         ->

               | | | | |     | | | |
               A-A-A-A-A\   /B-B-B-B
               |         E=x       |
               A-A-A-A-A/   \B-B-B-B
               | | | | |     | | | |

    Which can then act as an operator on EEMPS.
    """

    def __init__(self, mps, sysa, sysb, upper_ind_id='b{}'):
        n = mps.nsites

        # parse sysa and sysb ranges
        self.sysa_i, self.sysa_f = min(sysa), max(sysa) + 1
        self.sysb_i, self.sysb_f = min(sysb), max(sysb) + 1
        self.sysa = range(self.sysa_i, self.sysa_f)
        self.sysb = range(self.sysb_i, self.sysb_f)

        if len(sysa) != len(range(self.sysa_i, self.sysa_f)):
            raise ValueError("``sysa`` must be contiguous.")

        if len(sysb) != len(range(self.sysb_i, self.sysb_f)):
            raise ValueError("``sysb`` must be contiguous.")

        ket = mps.copy()
        for i in sysa:
            ket.add_tag('_SYSA', where=ket.site_tag(i))

        for i in sysb:
            ket.add_tag('_SYSB', where=ket.site_tag(i))

        # mixed canonize
        ket.left_canonize(stop=self.sysa_i)
        ket.right_canonize(stop=self.sysb_f - 1)

        # make bra and reindex non traced out sites and do partial transpose
        bra = ket.H
        bra.reindex_({bra.site_ind(i): upper_ind_id.format(i) for i in sysa})
        ket.reindex_({ket.site_ind(i): upper_ind_id.format(i) for i in sysb})

        self.lower_ind_id = ket.site_ind_id
        self.upper_ind_id = upper_ind_id
        self._phys_dim = ket.phys_dim()

        for i in self.sysa:
            ket.add_tag('_KET', where=ket.site_tag(i))
            bra.add_tag('_BRA', where=ket.site_tag(i))
        for i in self.sysb:
            # reverse tagging for partial transpose
            ket.add_tag('_BRA', where=ket.site_tag(i))
            bra.add_tag('_KET', where=ket.site_tag(i))

        self.TN = ket | bra

        # replace left and right envs with identity since canonized
        #    but also make sure all env bonds exist regardless of geometry
        le = [ket.site_tag(i) for i in range(0, self.sysa_i)]
        re = [ket.site_tag(i) for i in range(self.sysb_f, n)]
        if le:
            self.TN.replace_with_identity(le, inplace=True)
        else:
            self.TN.add_bond([ket.site_tag(0), '_KET'],
                             [ket.site_tag(0), '_BRA'])
        if re:
            self.TN.replace_with_identity(re, inplace=True)
        else:
            self.TN.add_bond([ket.site_tag(n - 1), '_KET'],
                             [ket.site_tag(n - 1), '_BRA'])

        # contract middle env if there is one
        if self.sysa_f != self.sysb_i:
            self.TN ^= slice(self.sysa_f, self.sysb_i)
            self.TN.add_tag('_ENV', where=ket.site_tag(self.sysa_f))

        # drop its site tags
        self.TN.drop_tags(map(ket.site_tag, range(self.sysa_f, self.sysb_i)))
        self.TN.sites = self.TN.calc_sites()

    def phys_dim(self):
        return self._phys_dim

    def to_dense(self):
        """Convert this TN to a dense array.
        """
        t = self.TN.contract_tags(...)
        t.fuse_([('k', list(map(self.lower_ind_id.format, self.TN.sites))),
                 ('b', list(map(self.upper_ind_id.format, self.TN.sites)))])
        return qu.qarray(t.data)

    @property
    def shape(self):
        d = self.phys_dim() ** len(self.TN.sites)
        return (d, d)

    def dot(self, vector):
        """Dot with a dense vector.
        """
        dims = [self.phys_dim()] * len(self.TN.sites)
        u_ix = [self.upper_ind_id.format(i) for i in self.TN.sites]
        l_ix = [self.lower_ind_id.format(i) for i in self.TN.sites]
        vecT = Tensor(vector.reshape(*dims), u_ix)
        outT = (self.TN | vecT).contract_tags(...)
        outT.fuse_({'k': l_ix})
        return outT.data.reshape(*vector.shape)

    def apply(self, other):
        r"""Apply this operator to a EEMPS vector::

                   -----------X----------   :
                  /   sysa   / \    sysb \  : _VEC
                  a-a-a-a-a-a   b-b-b-b-b-b :         -----------Y----------
                : | | | | | |   | | | | | |          /   sysa   / \    sysb \
            _BRA: A-A-A-A-A-A\ /B-B-B-B-B-B   ==>    A-A-A-A-A-A   B-B-B-B-B-B
                : |           E           | :        | | | | | |   | | | | | |
                  A-A-A-A-A-A/ \B-B-B-B-B-B : _KET
                  | | | | | |   | | | | | | :               *New vector*

        To create a new EEMPS.
        """
        v = other.copy()
        v.add_tag('_VEC')

        # align them
        v.site_ind_id = self.upper_ind_id

        # split bra and operator env off to be contracted with vector
        leave, remove = (self.TN | v).partition(['_BRA', '_ENV', '_VEC'])

        remove = remove.contract_tags(..., inplace=True)
        remove.drop_tags()
        remove.add_tag('_ENV')

        leave |= remove

        # 'upcast' leave from TensorNetwork to EEMPS
        #     already has the correct structure
        v.imprint(leave)
        leave.site_ind_id = other.site_ind_id
        return leave

    def rand_state(self, bond_dim, **mps_opts):
        """
        """
        return EEMPS_rand_state(self.sysa, self.sysb, self.TN.nsites,
                                phys_dim=self.phys_dim(), dtype=self.TN.dtype,
                                bond_dim=bond_dim, **mps_opts)


def construct_lanczos_tridiag_PTPTLazyMPS(A, K, v0=None, initial_bond_dim=None,
                                          beta_tol=1e-6, max_bond=None,
                                          v0_opts=None, k_min=10, seed=False):
    """
    """
    if initial_bond_dim is None:
        initial_bond_dim = 16
    if max_bond is None:
        max_bond = 16

    if v0 is None:
        if seed:
            # needs to be truly random so MPI processes don't overlap
            qu.seed_rand(random.SystemRandom().randint(0, 2**32 - 1))

        V = A.rand_state(bond_dim=initial_bond_dim)
    else:  # normalize
        V = v0 / (v0.H @ v0)**0.5
    Vm1 = EEMPS_zeros_like(V)

    alpha = np.zeros(K + 1)
    beta = np.zeros(K + 2)

    beta[1] = A.phys_dim()**((len(A.sysa) + len(A.sysb)) / 2)

    compress_kws = {'max_bond': max_bond, 'method': 'svd'}

    for j in range(1, K + 1):
        Vt = A.apply(V)
        Vt -= beta[j] * Vm1
        Vt.compress_all(**compress_kws)
        alpha[j] = (V.H @ Vt).real
        Vt -= alpha[j] * V
        beta[j + 1] = ((Vt.H @ Vt)**0.5).real
        Vt.compress_all(**compress_kws)

        # check for convergence
        if abs(beta[j + 1]) < beta_tol:
            yield alpha[1:j + 1], beta[2:j + 2], beta[1]**2
            break

        Vm1 = V.copy()
        V = Vt / beta[j + 1]

        if j >= k_min:
            yield (np.copy(alpha[1:j + 1]),
                   np.copy(beta[2:j + 2]),
                   np.copy(beta[1])**2)
