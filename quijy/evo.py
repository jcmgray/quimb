"""
Contains an evolution class, QuEvo to efficiently manage time evolution of
quantum states according to schrodingers' equation, and related functions.
TODO: iterative method, sparse etc., turn off optimzations for small n
"""

from numpy.linalg import multi_dot as mdot
from numexpr import evaluate as evl
from quijy.core import isop, quijify, ldmul, rdmul
from quijy.solve import eigsys


class QuEvo(object):
    """
    A class for evolving quantum systems according to schro equation
    Note that vector states are converted to kets always.
    """
    # TODO diagonalise or use iterative method ...
    def __init__(self,
                 p0,
                 ham,
                 dop=None,
                 solve=False,
                 t0=0):
        """
        Inputs:
            p0: inital state, either vector or operator
            ham: Governing Hamiltonian, can be tuple (eigvals, eigvecs)
            dop: whether to force evolution as density operator
            solve: whether to immediately solve hamiltonian
            t0: initial time
            l: eigenvalues if ham already solved
            v: eigevvectors if ham already solved
        """
        super(QuEvo, self).__init__()

        # Convert state to ket or dop and mark as such
        self.p0 = quijify(p0)
        if isop(self.p0):
            if dop is False:
                raise ValueError('Cannot convert dop to ket.')
            else:
                self.dop = True
        else:
            if dop is True:
                self.p0 = quijify(p0, 'dop')
                self.dop = True
            else:
                self.p0 = quijify(p0, 'ket')  # make sure ket
                self.dop = False
        self.pt = p0  # Current state (start with same as initial)
        self.t0 = t0  # initial time
        self.t = t0  # current time

        # Set Hamiltonian and infer if solved already from tuple
        if type(ham) is tuple:  # Eigendecomposition already supplied
            self.l, self.v = ham
            # Solve for initial state in energy basis
            if self.dop:
                self.pe0 = mdot([self.v.H, p0, self.v])
            else:
                self.pe0 = self.v.H @ p0
            self.solved = True
        elif solve:
            self.ham = ham
            self.solve_ham()
            self.solved = True
        else:
            self.ham = ham
            self.solved = False

    def solve_ham(self):
        # Diagonalise hamiltonian
        self.l, self.v = eigsys(self.ham)
        # Find initial state in energy eigenbasis at t0
        if self.dop:
            self.pe0 = mdot([self.v.H, self.p0, self.v])
        else:
            self.pe0 = self.v.H @ self.p0
        # Mark solved
        self.solved = True

    def update_to(self, t):
        self.t = t
        dt = self.t - self.t0
        exptl = evl('exp(-1.0j*dt*l)', local_dict={'l': self.l, 'dt': dt})
        if self.dop:
            lvpvl = rdmul(ldmul(exptl, self.pe0), evl('conj(exptl)'))
            self.pt = mdot([self.v, lvpvl, self.v.H])
        else:
            self.pt = self.v @ ldmul(exptl, self.pe0)


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
