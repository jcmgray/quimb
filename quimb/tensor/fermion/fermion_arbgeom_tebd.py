import itertools
import collections

from ..tensor_arbgeom_tebd import LocalHamGen, TEBDGen, SimpleUpdateGen
from .block_interface import eye, to_exponential, Hubbard

class LocalHamGen(LocalHamGen):

    def __init__(self, H2, H1=None):
        # caches for not repeating operations / duplicating tensors
        self._op_cache = collections.defaultdict(dict)

        self.terms = dict(H2)

        self.sites = tuple(
            sorted(set(itertools.chain.from_iterable(self.terms)))
        )

        # first combine terms to ensure coo1 < coo2
        for where in tuple(filter(bool, self.terms)):
            coo1, coo2 = where
            
            new_where = coo2, coo1
            if new_where in self.terms:
                X12 = self.terms.pop(new_where).transpose([1,0,3,2])
                self.terms[where] = self.terms[where]+X12

        # parse one site terms
        if H1 is None:
            H1s = dict()
        elif hasattr(H1, "shape"):
            # set a default site term
            H1s = {None: H1}
        else:
            H1s = dict(H1)

        # possibly set the default single site term
        default_H1 = H1s.pop(None, None)
        if default_H1 is not None:
            for site in self.sites:
                H1s.setdefault(site, default_H1)

        self.terms.update(H1s)

    def _flip_cached(self, x):
        cache = self._op_cache["flip"]
        key = id(x)
        if key not in cache:
            cache[key] = x.transpose([1,0,3,2])
        return cache[key]
    
    def _expm_cached(self, x, y):
        cache = self._op_cache['expm']
        key = (id(x), y)
        if key not in cache:
            out = to_exponential(x, y)
            cache[key] = out
        return cache[key]

def Hubbard_from_TN(tn, t, u, mu=0.):
    H2 = dict()
    for i, isite in enumerate(tn.sites):
        ix = tn.site_ind(isite)
        itid, = tn.ind_map[ix]
        Ti, = tn._inds_get(ix)
        ni = len(tn._get_neighbor_tids([itid]))
        for jsite in tn.sites[i+1:]:
            jx = tn.site_ind(jsite)
            jtid, = tn.ind_map[jx]
            Tj, = tn._inds_get(jx)
            if not Ti.bonds(Tj):
                continue
            nj = len(tn._get_neighbor_tids([jtid]))
            if Ti.get_fermion_info()[1] < Tj.get_fermion_info()[1]:
                key = (isite, jsite)
                factors = (1./ni, 1./nj)
            else:
                key = (jsite, isite)
                factors = (1./nj, 1./ni)
            H2[key] = Hubbard(t,u, mu, factors)
    return LocalHamGen(H2)

class TEBDGen(TEBDGen):
    """Generic class for performing time evolving block decimation on an
    arbitrary graph, i.e. applying the exponential of a Hamiltonian using
    a product formula that involves applying local exponentiated gates only.
    """

    def compute_energy(self):
        """Compute and return the energy of the current state. Subclasses can
        override this with a custom method to compute the energy.
        """
        raise NotImplementedError
        return self._psi.compute_local_expectation_simple(
            terms=self.ham.terms,
            **self.compute_energy_opts
        )


class SimpleUpdateGen(SimpleUpdateGen):

    def gate(self, U, where):
        raise NotImplementedError
        self._psi.gate_simple_(
            U, where, gauges=self.gauges, **self.gate_opts
        )

    def compute_energy(self):
        raise NotImplementedError
        return self._psi.compute_local_expectation_simple(
            terms=self.ham.terms,
            gauges=self.gauges,
            **self.compute_energy_opts,
        )

    def get_state(self, absorb_gauges=True):
        psi = self._psi.copy()

        if absorb_gauges:
            psi.gauge_simple_insert(self.gauges)
        else:
            raise NotImplementedError("gauge has to be absorbed")

        return psi

    def set_state(self, psi):
        raise NotImplementedError
        self._psi = psi.copy()
        self.gauges = {}
        self._psi.gauge_all_simple_(gauges=self.gauges)
