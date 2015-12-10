from quijy.core import (isbra, isket, isop, esys, qonvert, ldmul, rdmul)
from numpy.linalg import multi_dot as mdot
from numexpr import evaluate as evl
from numpy import multiply


class QuEvo(object):
    """A class for evolving """
    def __init__(self,
                 p0=None,  # Starting state
                 dop=None,  # evolve as density operator
                 ham=None,  # hamiltonian
                 solve=False,  # whether to immediately solve for
                 evals=None,  # energy eigenvalues
                 evecs=None,  # energy eigenvectors
                 method=None):  # TODO diagonalise or use iterative method ...
        super(QuEvo, self).__init__()

        # Convert state to ket or dop and mark as such
        self.p0 = qonvert(p0)
        if isop(self.p0):
            if dop is False:
                raise ValueError('Cannot convert dop to ket.')
            else:
                self.dop = True
        else:
            if dop is True:
                self.p0 = qonvert(p0, 'dop')
                self.dop = True
            else:
                self.p0 = qonvert(p0, 'ket')
                self.dop = False
        self.pt = p0  # Current state (start with same as initial)

        self.ham = ham
        if solve:
            self.solve_ham()
            self.solved = True
        elif evals is not None and evecs is not None:
            self.evals = evals
            self.evecs = evecs
            # Solve for initial state in energy basis
            if self.dop:
                self.pe0 = mdot([self.evecs.H * p0 * self.evecs])
            else:
                self.pe0 = self.evecs.H * p0
            self.solved = True
        else:
            self.solved = False
        self.method = method

    def solve_ham(self):
        # Diagonalise hamiltonian
        self.evals, self.evecs = esys(self.ham)
        # Find initial state in energy eigenbasis
        if self.dop:
            self.pe0 = mdot([self.evecs.H * self.p0 * self.evecs])
        else:
            self.pe0 = self.evecs.H * self.p0
        # Mark solved
        self.solved = True

    def update_to(self, t):
        l = self.evals
        exptl = evl('exp(-1.0j*t*l)')
        if self.dop:
            lvpvl = rdmul(ldmul(exptl, self.pe0), evl('conj(exptl)'))
            self.pt = mdot([self.evecs, lvpvl, self.evecs.H])
        else:
            self.pt = self.evecs * ldmul(exptl, self.pe0)
