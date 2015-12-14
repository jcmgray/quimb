"""
Contains an evolution class, QuEvo to efficiently manage time evolution of
quantum states according to schrodingers' equation, and related functions.
"""

from quijy.core import (isbra, isket, isop, esys, qonvert, ldmul, rdmul)
from numpy.linalg import multi_dot as mdot
from numexpr import evaluate as evl
from numpy import multiply


class QuEvo(object):
    """
    A class for evolving quantum systems according to schro equation
    Note that vector states are converted to kets always.
    """
    # TODO diagonalise or use iterative method ...
    def __init__(self,
                 ham=None,
                 p0=None,
                 dop=None,
                 solve=False,
                 evals=None,
                 evecs=None):
        """
        Inputs:
            ham: Governing Hamiltonian
            p0: inital state, either vector or operator
            dop: whether to force evolution as density operator
            solve: whether to immediately solve hamiltonian
            evals: eigenvalues if ham already solved
            evecs: eigevvectors if ham already solved
        """
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
                self.p0 = qonvert(p0, 'ket')  # make sure ket
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


def rk4_step(y0, f, dt, t=None):
    """
    Performs a 4th order runge-kutta step of length dt according to the
    relation dy/dt = f(t, y). If t is not specified then assumes f = f(y)
    """
    if t is None:
        k1 = f(y0)
        k2 = f(y0 + k1 * dt / 2.0)
        k3 = f(y0 + k2 * dt / 2.0)
        k4 = f(y0 + k3 * dt)
    else:
        k1 = f(y0, t)
        k2 = f(y0 + k1 * dt / 2.0, t + dt / 2.0)
        k3 = f(y0 + k2 * dt / 2.0, t + dt / 2.0)
        k4 = f(y0 + k3 * dt, t + dt)
    return y0 + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
