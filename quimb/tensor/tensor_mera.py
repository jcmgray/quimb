from math import log2
import itertools

import numpy as np

from .tensor_core import rand_uuid, Tensor, TensorNetwork
from .tensor_1d import TensorNetwork1D, TensorNetwork1DVector


def rand_iso(n, m, dtype=complex):
    """Generate a random isometry.
    """
    data = np.random.randn(n, m)

    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1.0j * np.random.randn(n, m)

    q, r = np.linalg.qr(data if n > m else data.T)
    return q if n > m else q.T


def is_power_of_2(x):
    return ((x & (x - 1)) == 0) and x > 0


class MERA(TensorNetwork, TensorNetwork1D, TensorNetwork1DVector):
    r"""The Multi-scale Entanglement Renormalization Ansatz (MERA) state:

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
        0 1 2 3 4 ....                            ... n-2 n-1

    Parameters
    ----------
    n : int
        The number of phyiscal sites. Shoule be a power of 2.
    uni : array or sequence of arrays of shape (d, d, d, d).
        The unitary operator(s). These will be cycled over and placed from
        bottom left to top right in diagram above.
    iso : array or sequence of arrays of shape (d, d, d)
        The isometry operator(s). These will be cycled over and placed from
        bottom left to top right in diagram above.
    phys_dim : int, optional
        The dimension of the local hilbert space.
    """

    _EXTRA_PROPS = ('_site_ind_id', '_site_tag_id', 'cyclic')

    def __init__(self, n, uni=None, iso=None, phys_dim=2,
                 site_ind_id="k{}", site_tag_id="I{}", **tn_opts):

        # short-circuit for copying MPSs
        if isinstance(n, MERA):
            super().__init__(n)
            for ep in MERA._EXTRA_PROPS:
                setattr(self, ep, getattr(n, ep))
            return

        self._site_ind_id = site_ind_id
        self._site_tag_id = site_tag_id
        self.cyclic = True

        if not is_power_of_2(n):
            raise ValueError("``n`` should be a power of 2.")

        nlayers = int(log2(n))

        if isinstance(uni, np.ndarray):
            uni = (uni,)

        if isinstance(iso, np.ndarray):
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
                eff_n = n // 2**i

                for j in range(0, eff_n, 2):

                    # generate the unitary:
                    #  ul | | ur
                    #     UNI
                    #  ll | | lr
                    #     j j+1
                    ll, lr = map(l_ind_id.format, (j, (j + 1) % eff_n))
                    ul, ur = map(m_ind_id.format, (j, (j + 1) % eff_n))
                    inds = (ll, lr, ul, ur)

                    tags = {"_UNI", "_LAYER{}".format(i)}
                    if i == 0:
                        tags.add(site_tag_id.format(j))
                        tags.add(site_tag_id.format(j + 1))

                    yield Tensor(next(unis), inds, tags=tags)

                    # generate the isometry (offset by one effective site):
                    #      | ui
                    #     ISO
                    #  ll | | lr
                    #   j+1 j+2
                    ll, lr = map(m_ind_id.format, (j + 1, (j + 2) % eff_n))
                    ui = u_ind_id.format(j // 2)
                    inds = (ll, lr, ui)
                    tags = {"_ISO", "_LAYER{}".format(i)}

                    if i == nlayers - 1:
                        # don't leave dangling index at top
                        yield Tensor(np.eye(phys_dim) / 2**0.5,
                                     inds[:-1], tags)
                    else:
                        yield Tensor(next(isos), inds, tags)

        super().__init__(gen_mera_tensors(), check_collisions=False,
                         structure=site_tag_id)

        # tag the MERA with the 'causal-cone' of each site
        for i in range(nlayers):
            for j in range(n):
                # get isometries in the same layer
                for t in self.select_neighbors(j):
                    if f'_LAYER{i}' in t.tags:
                        t.add_tag(f'I{j}')

                # get unitaries in layer above
                for t in self.select_neighbors(j):
                    if f'_LAYER{i + 1}' in t.tags:
                        t.add_tag(f'I{j}')

    @classmethod
    def rand_invar(cls, n, phys_dim=2):
        """Generate a random invariant MERA.
        """
        iso_shape = (phys_dim, phys_dim, phys_dim)
        iso = rand_iso(phys_dim**2, phys_dim).reshape(iso_shape)

        uni_shape = (phys_dim, phys_dim, phys_dim, phys_dim)
        uni = rand_iso(phys_dim**2, phys_dim**2).reshape(uni_shape)

        return cls(n, uni, iso, phys_dim=phys_dim)

    @staticmethod
    def contract_structured_all(old, inplace=False, **opts):
        new = old if inplace else old.copy()
        return new.contract_tags(all, **opts)

    def to_dense(self):
        """Convert this MERA to dense matrix form.
        """
        return np.asmatrix(super().to_dense(self.site_inds).reshape(-1, 1))
