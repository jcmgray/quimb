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


class EEMPS(MatrixProductState):
    """Environment matrix product state::

         -------------E--------------
        /   sysa     / \    sysb     \
        o-o-o-o-o-o-o   o-o-o-o-o-o-o-o
        | | | | | | |   | | | | | | | |


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
        self |= Tensor(env, inds=(ai, af, bi, bf), tags={'_ENV'})

        # tag the subsystems
        self.add_tag('_SYSA', where=map(self.site_tag, sysa), mode='any')
        self.add_tag('_SYSB', where=map(self.site_tag, sysb), mode='any')

    _EXTRA_PROPS = ('sysa', 'sysb')

    def imprint(self, other):
        """Coerce
        """
        for p in EEMPS._EXTRA_PROPS:
            setattr(other, p, getattr(self, p))
        super().imprint(other)
        other.__class__ = EEMPS

    def contract_structured_all(self, old, inplace=False):
        new = old if inplace else old.copy()
        new ^= slice(self.sysa[0], self.sysa[-1])
        new ^= slice(self.sysb[0], self.sysb[-1])
        return new.contract_tags(...)

    def add_EEMPS(self, other, inplace=False):
        """Add another EEMPS.
        """
        self = self if inplace else self.copy()

        # for storing the bonds to the effective envinroment
        env_reindex_map = {}

        for i in self.sites:

            t1, t2 = self.site[i], other.site[i]

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
                    s_env_bnds = t1.shared_inds(self['_ENV'])
                    o_env_bnds = t2.shared_inds(other['_ENV'])

                    for sb, ob in zip(s_env_bnds, o_env_bnds):
                        reindex_map[ob] = sb
                        env_reindex_map[ob] = sb

            t1.direct_product(t2.reindex(reindex_map), inplace=True,
                              sum_inds=self.site_ind(i))

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
        t.fuse([('k', list(map(self.site_ind_id.format, self.sites)))],
               inplace=True)
        return np.asmatrix(t.data.reshape(-1, 1))


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
                            nsites=other.nsites, phys_dim=other.phys_dim,
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
    return EEMPS_zeros(sysa=other.sysa, sysb=other.sysb, nsites=other.nsites,
                       phys_dim=other.phys_dim, dtype=other.dtype, **mps_opts)


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
        ket.right_canonize(stop=self.sysb_f)

        # make bra and reindex non traced out sites and do partial transpose
        bra = ket.H
        bra.reindex({bra.site_ind(i): upper_ind_id.format(i)
                     for i in sysa}, inplace=True)
        ket.reindex({ket.site_ind(i): upper_ind_id.format(i)
                     for i in sysb}, inplace=True)

        self.lower_ind_id = ket.site_ind_id
        self.upper_ind_id = upper_ind_id

        for i in self.sysa:
            ket.add_tag('_KET', where=ket.site_tag(i))
            bra.add_tag('_BRA', where=ket.site_tag(i))
        for i in self.sysb:
            # reverse tagging for partial transpose
            ket.add_tag('_BRA', where=ket.site_tag(i))
            bra.add_tag('_KET', where=ket.site_tag(i))

        self.X = ket | bra

        # replace left and right envs with identity since canonized
        le = [self.X.structure.format(i) for i in range(0, self.sysa_i)]
        re = [self.X.structure.format(i) for i in range(self.sysb_f, n)]
        self.X.replace_with_identity(le, inplace=True)
        self.X.replace_with_identity(re, inplace=True)

        # contract middle env if there is one
        if self.sysa_f != self.sysb_i:
            self.X ^= slice(self.sysa_f, self.sysb_i)
            self.X.add_tag('_ENV', where=ket.site_tag(self.sysa_f))

        # drop its site tags
        self.X.drop_tags(map(ket.site_tag, range(self.sysa_f, self.sysb_i)))
        self.X.sites = self.X.calc_sites()

    def to_dense(self):
        t = self.X.contract_tags(...)
        t.fuse([('k', list(map(self.lower_ind_id.format, self.X.sites))),
                ('b', list(map(self.upper_ind_id.format, self.X.sites)))],
               inplace=True)
        return np.asmatrix(t.data)

    def apply(self, other):
        """Apply this operator to a EEMPS vector::

               -----------X----------   :
              /   sysa   / \    sysb \  : _VEC
              a-a-a-a-a-a   b-b-b-b-b-b :         -----------Y----------
            : | | | | | |   | | | | | |          /   sysa   / \    sysb \
        _BRA: A-A-A-A-A-A\ /B-B-B-B-B-B   -->    A-A-A-A-A-A   B-B-B-B-B-B
            : |           E           | :        | | | | | |   | | | | | |
              A-A-A-A-A-A/ \B-B-B-B-B-B : _KET
              | | | | | |   | | | | | | :               *New vector*

        """
        # import pdb; pdb.set_trace()

        v = other.copy()
        v.add_tag('_VEC')

        # align them
        v.site_ind_id = self.upper_ind_id

        # split bra and operator env off to be contracted with vector
        leave, remove = self.X.partition(['_BRA', '_ENV'])

        remove |= v
        remove = remove.contract(..., inplace=True)
        remove.drop_tags()
        remove.tags.add('_ENV')

        leave |= remove

        # import pdb; pdb.set_trace()

        # 'upcast' leave from TensorNetwork to EEMPS
        v.imprint(leave)
        leave.site_ind_id = other.site_ind_id
        return leave
