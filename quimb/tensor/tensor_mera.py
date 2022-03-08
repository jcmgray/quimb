from math import log2
import itertools

import numpy as np

import quimb as qu
from .tensor_core import rand_uuid, IsoTensor, TensorNetwork
from .tensor_1d import TensorNetwork1D, TensorNetwork1DVector


def is_power_of_2(x):
    return ((x & (x - 1)) == 0) and x > 0


class MERA(TensorNetwork1DVector,
           TensorNetwork1D,
           TensorNetwork):
    r"""The Multi-scale Entanglement Renormalization Ansatz (MERA) state::

            ...     ...     ...     ...     ...     ...
         |       |       |       |       |       |       |
        ISO     ISO     ISO     ISO     ISO     ISO     ISO   :
           \   /   \   /   \   /   \   /   \   /   \   /      : '_LAYER1'
            UNI     UNI     UNI     UNI     UNI     UNI       :
           /   \   /   \   /   \   /   \   /   \   /   \
        O ISO ISO ISO ISO ISO ISO ISO ISO ISO ISO ISO ISO I   :
        | | | | | | | | | | | | | | | | | | | | | | | | | |   : '_LAYER0'
        UNI UNI UNI UNI UNI UNI UNI UNI UNI UNI UNI UNI UNI   :
        | | | | | | | | | | | | | | | | | | | | | | | | | |  <-- phys_dim
        0 1 2 3 4 ....                            ... L-2 L-1

    Parameters
    ----------
    L : int
        The number of phyiscal sites. Shoule be a power of 2.
    uni : array or sequence of arrays of shape (d, d, d, d).
        The unitary operator(s). These will be cycled over and placed from
        bottom left to top right in diagram above.
    iso : array or sequence of arrays of shape (d, d, d)
        The isometry operator(s). These will be cycled over and placed from
        bottom left to top right in diagram above.
    phys_dim : int, optional
        The dimension of the local hilbert space.
    dangle : bool, optional
        Whether to leave a dangling index on the final isometry, in order to
        maintain perfect scale invariance, else join the final unitaries just
        with an indentity.
    """

    _EXTRA_PROPS = ('_site_ind_id', '_site_tag_id', 'cyclic', '_L')
    _CONTRACT_STRUCTURED = False

    def __init__(self, L, uni=None, iso=None, phys_dim=2, dangle=False,
                 site_ind_id="k{}", site_tag_id="I{}", **tn_opts):

        # short-circuit for copying MERA
        if isinstance(L, MERA):
            super().__init__(L)
            for ep in MERA._EXTRA_PROPS:
                setattr(self, ep, getattr(L, ep))
            return

        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self.cyclic = True
        self._L = L

        if not is_power_of_2(L):
            raise ValueError("``L`` should be a power of 2.")

        nlayers = round(log2(L))

        if hasattr(uni, 'shape'):
            uni = (uni,)

        if hasattr(iso, 'shape'):
            iso = (iso,)

        unis = itertools.cycle(uni)
        isos = itertools.cycle(iso)

        def gen_mera_tensors():
            u_ind_id = site_ind_id

            for i in range(nlayers):

                # index id connecting to layer below
                l_ind_id = u_ind_id
                # index id connecting to isos to unis
                m_ind_id = rand_uuid() + "_{}"
                # index id connecting to layer above
                u_ind_id = rand_uuid() + "_{}"

                # number of tensor sites in this layer
                eff_L = L // 2**i

                for j in range(0, eff_L, 2):

                    # generate the unitary:
                    #  ul | | ur
                    #     UNI
                    #  ll | | lr
                    #     j j+1
                    ll, lr = map(l_ind_id.format, (j, (j + 1) % eff_L))
                    ul, ur = map(m_ind_id.format, (j, (j + 1) % eff_L))
                    inds = (ll, lr, ul, ur)

                    tags = ("_UNI", f"_LAYER{i}")
                    if i == 0:
                        tags += (site_tag_id.format(j),
                                 site_tag_id.format(j + 1))

                    yield IsoTensor(next(unis), inds=inds,
                                    tags=tags, left_inds=(ll, lr))

                    # generate the isometry (offset by one effective site):
                    #      | ui
                    #     ISO
                    #  ll | | lr
                    #   j+1 j+2
                    ll, lr = map(m_ind_id.format, (j + 1, (j + 2) % eff_L))
                    ui = u_ind_id.format(j // 2)
                    inds = (ll, lr, ui)
                    tags = ("_ISO", f"_LAYER{i}")

                    if i < nlayers - 1 or dangle:
                        yield IsoTensor(next(isos), inds=inds,
                                        tags=tags, left_inds=(ll, lr))
                    else:
                        # don't leave dangling index at top
                        iso_f = next(isos)
                        yield IsoTensor(
                            np.eye(iso_f.shape[0], dtype=iso_f.dtype) / 2**0.5,
                            inds=inds[:-1], tags=tags, left_inds=(ll, lr)
                        )

        super().__init__(gen_mera_tensors(), virtual=True)

        # tag the MERA with the 'causal-cone' of each site
        for i in range(nlayers):
            for j in range(L):
                # get isometries in the same layer
                for t in self.select_neighbors(j):
                    if f'_LAYER{i}' in t.tags:
                        t.add_tag(f'I{j}')

                # get unitaries in layer above
                for t in self.select_neighbors(j):
                    if f'_LAYER{i + 1}' in t.tags:
                        t.add_tag(f'I{j}')

    @classmethod
    def rand(cls, L, max_bond=None, phys_dim=2, dtype=float, **mera_opts):

        d = phys_dim
        if max_bond is None:
            max_bond = d

        def gen_unis():
            D = d
            m = L // 2

            while True:
                for _ in range(m):
                    uni = qu.rand_iso(D**2, D**2, dtype=dtype)
                    uni.shape = (D, D, D, D)
                    yield uni

                D = min(D**2, max_bond)
                m //= 2

        def gen_isos():
            Dl = d
            Du = min(Dl**2, max_bond)
            m = L // 2

            while True:
                for _ in range(m):
                    iso = qu.rand_iso(Dl**2, Du, dtype=dtype)
                    iso.shape = (Dl, Dl, Du)
                    yield iso

                Dl = Du
                Du = min(Dl**2, max_bond)
                m //= 2

        return cls(L, gen_unis(), gen_isos(), phys_dim=d, **mera_opts)

    @classmethod
    def rand_invar(cls, L, phys_dim=2, dtype=float, **mera_opts):
        """Generate a random translational and scale invariant MERA.
        """
        d = phys_dim

        iso = qu.rand_iso(d**2, d, dtype=dtype)
        iso.shape = (d, d, d)

        uni = qu.rand_iso(d**2, d**2, dtype=dtype)
        uni.shape = (d, d, d, d)

        return cls(L, uni, iso, phys_dim=d, **mera_opts)
