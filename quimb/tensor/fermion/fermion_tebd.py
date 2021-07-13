import numpy as np
from .block_interface import eye, Hubbard
from ..tensor_core import tensor_contract, rand_uuid
from .fermion_core import FermionTensor
from . import block_tools
from ..tensor_2d_tebd import SimpleUpdate as _SimpleUpdate
from ..tensor_2d_tebd import conditioner
from .fermion_2d_tebd import LocalHam2D
import collections
INVERSE_CUTOFF = 1e-10

def _tid_to_phys_ind(tn, tid):
    ind = None
    T = tn.tensor_map[tid]
    for ix in T.inds:
        if len(tn.ind_map[ix])==1:
            ind = ix
            break
    return ind

def gen_neighbor_pairs(tn):
    all_tids = list(tn.tensor_map)
    for i, itid in enumerate(all_tids):
        Ti = tn.tensor_map[itid]
        isite = Ti.get_fermion_info()[1]
        for jtid in all_tids[i+1:]:
            Tj = tn.tensor_map[jtid]
            jsite = Tj.get_fermion_info()[1]
            if Ti.bonds(Tj):
                ix = _tid_to_phys_ind(tn, itid)
                jx = _tid_to_phys_ind(tn, jtid)
                if isite<jsite:
                    yield (itid, ix), (jtid, jx)
                else:
                    yield (jtid, jx), (itid, ix)

def HubbardHam(tn, t, u, mu=0):
    ind_map = tn.ind_map
    ham = dict()
    for (itid, ix), (jtid, jx) in gen_neighbor_pairs(tn):
        ni = len(tn._get_neighbor_tids([itid]))
        nj = len(tn._get_neighbor_tids([jtid]))
        ham[(ix, jx)] = Hubbard(t,u, mu, (1./ni, 1./nj))
    return ham


class LocalHam(LocalHam2D):
    def __init__(self, H2, H1=None):
        # caches for not repeating operations / duplicating tensors
        self._op_cache = collections.defaultdict(dict)
        self.terms = dict(H2)

        # first combine terms to ensure coo1 < coo2
        for where in tuple(filter(bool, self.terms)):
            coo1, coo2 = where
            new_where = coo2, coo1
            if new_where in self.terms:
                raise KeyError("appearing twice", where, new_where)

        if H1 is None:
            return
        self.terms.update(H1)

def _get_location(Ti, Tj):
    if Ti.get_fermion_info()[1]<Tj.get_fermion_info()[1]:
        return "front", "back"
    else:
        return "back", "front"

class SimpleUpdate(_SimpleUpdate):

    def setup(
        self,
        gauge_renorm=True,
        gauge_smudge=1e-6,
        condition_tensors=True,
        condition_balance_bonds=True
    ):
        self.gauge_renorm = gauge_renorm
        self.gauge_smudge = gauge_smudge
        self.condition_tensors = condition_tensors
        self.condition_balance_bonds = condition_balance_bonds
        self.neighbor_pairs = dict()
        for (_, ix), (_, iy) in gen_neighbor_pairs(self._psi):
            if ix not in self.neighbor_pairs:
                self.neighbor_pairs[ix] = [iy]
            else:
                self.neighbor_pairs[ix].append(iy)
            if iy not in self.neighbor_pairs:
                self.neighbor_pairs[iy] = [ix]
            else:
                self.neighbor_pairs[iy].append(ix)

    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        # create the gauges like whatever data array is in the first site.
        self._gauges = dict()
        inv_dict = {"+":"-", "-":"+"}
        for (itid, ix), (jtid, jx) in gen_neighbor_pairs(self._psi):
            Ti = self._psi.tensor_map[itid]
            Tj = self._psi.tensor_map[jtid]
            bnd, = Ti.bonds(Tj)
            sign_i = Ti.data.pattern[Ti.inds.index(bnd)]
            bond_info = Ti.bond_info(bnd)
            Tsval = eye(bond_info)
            Tsval.pattern = sign_i + inv_dict[sign_i]
            self._gauges[(ix, jx)] = Tsval

    def _unpack_gauge(self, ix, iy):
        Ta, Tb = self._psi._inds_get(ix, iy)
        if (ix, iy) in self.gauges:
            Tsval = self.gauges[(ix, iy)]
            location = _get_location(Ta, Tb)
        else:
            Tsval = self.gauges[(iy, ix)]
            location = _get_location(Tb, Ta)
        return Ta, Tb, Tsval, location

    def compute_energy(self):
        ket = self.state
        ket.add_tag("KET")
        bra = ket.retag({"KET": "BRA"})
        bra = bra.H
        bra.mangle_inner_("*")
        norm = ket & bra
        phased_ham = dict()
        for where, U in self.ham.terms.items():
            inds = (rand_uuid(), rand_uuid()) + where
            TU = FermionTensor(U.copy(), inds=inds)
            new_U = ket.fermion_space.move_past(TU).data
            phased_ham[where] = new_U
        return phased_ham

    def gate(self, U, where):
        """Like ``TEBD2D.gate`` but absorb and extract the relevant gauges
        before and after each gate application.
        """
        ija, ijb = where

        def env_neighbours(i, exclude=None):
            return [ix for ix in self.neighbor_pairs[i] if ix != exclude]

        # get the relevant neighbours for string of sites
        neighbours = {ija: env_neighbours(ija, exclude=ijb),
                      ijb: env_neighbours(ijb, exclude=ija)}

        string = (ija, ijb)

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            for neighbour in neighbours[site]:
                Ta, Tb, Tsval, location = self._unpack_gauge(site, neighbour)
                bond_ind, = Ta.bonds(Tb)
                mult_val = block_tools.add_with_smudge(Tsval,
                            INVERSE_CUTOFF, self.gauge_smudge)
                Ta.multiply_index_diagonal_(ind=bond_ind,
                            x=mult_val, location=location[0])

        # absorb the inner bond gauges equally into both sites along string
        Ta, Tb, Tsval, location = self._unpack_gauge(ija, ijb)
        mult_val = block_tools.sqrt(Tsval)
        bnd, = Ta.bonds(Tb)
        Ta.multiply_index_diagonal_(ind=bnd, x=mult_val, location=location[0])
        Tb.multiply_index_diagonal_(ind=bnd, x=mult_val, location=location[1])

        info = dict()
        self._psi.gate_inds_(U, where,
                    absorb=None, info=info, **self.gate_opts)

        # set the new singualar values all along the chain
        ((_, bond_pair), s),  = info.items()
        if self.gauge_renorm:
            s = s / s.norm()
        if bond_pair not in self.gauges:
            del self.gauges[(bond_pair[1], bond_pair[0])]
        self.gauges[bond_pair] = s

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            for neighbour in neighbours[site]:
                Ta, Tb, Tsval, location = self._unpack_gauge(site, neighbour)
                bond_ind, = Ta.bonds(Tb)
                mult_val = block_tools.inv_with_smudge(Tsval,
                            INVERSE_CUTOFF, self.gauge_smudge)
                Ta.multiply_index_diagonal_(ind=bond_ind,
                            x=mult_val, location=location[0])

    def get_state(self, absorb_gauges=True):
        psi = self._psi.copy()
        if not absorb_gauges:
            raise NotImplementedError
        else:
            for (ix, jx), Tsval in self.gauges.items():
                Ti, Tj = psi._inds_get(ix, jx)
                bnd, = Ti.bonds(Tj)
                loci, locj = _get_location(Ti, Tj)
                mult_val = block_tools.sqrt(Tsval)
                Ti.multiply_index_diagonal_(bnd, mult_val, location=loci)
                Tj.multiply_index_diagonal_(bnd, mult_val, location=locj)

        if self.condition_tensors:
            conditioner(psi, balance_bonds=self.condition_balance_bonds)
        return psi

    def _check_energy(self):
        if self.its and (self._n == self.its[-1]):
            # only compute if haven't already
            return self.energies[-1]

        if self.compute_energy_fn is not None:
            en = self.compute_energy_fn(self)
        else:
            en = self.compute_energy()

        if self.compute_energy_per_site:
            nsites = len(self._psi.tensor_map)
            en = en / nsites

        self.energies.append(float(en))
        self.taus.append(float(self.tau))
        self.its.append(self._n)

        if self.keep_best and en < self.best['energy']:
            self.best['energy'] = en
            self.best['state'] = self.state
            self.best['it'] = self._n

        return self.energies[-1]
