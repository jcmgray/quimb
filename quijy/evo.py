"""
Contains an evolution class, QuEvo to efficiently manage time evolution of
quantum states according to schrodingers' equation, and related functions.
TODO: iterative method, sparse etc., turn off optimzations for small n
"""

import numpy as np
from scipy.integrate import complex_ode
from .accel import (
    isop,
    ldmul,
    rdmul,
)
from .accel import issparse
from .core import qjf, eye
from .solve import eigsys, norm


# --------------------------------------------------------------------------- #
# Quantum evolution equations                                                 #
# --------------------------------------------------------------------------- #

def schrodinger_eq_ket(ham):
    """ Wavefunction schrodinger equation.

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution

    Returns
    -------
        psi_dot(t, y): function to calculate psi_dot(t) at psi(t). """
    def psi_dot(t, y):
        return -1.0j * (ham.dot(y))

    return psi_dot


def schrodinger_eq_dop(ham):
    """ Density operator schrodinger equation, but with flattened input/output.

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form). """
    d = ham.shape[0]
    ham = ham.A

    def rho_dot(t, y):
        rho = y.reshape(d, d)
        hrho = ham.dot(rho)
        return -1.0j * (hrho - hrho.T.conj()).reshape(-1)

    return rho_dot


def schrodinger_eq_dop_vec(ham):
    """ Density operator schrodinger equation, but with flattened input/output
    and vectorised superoperation mode (no reshaping required).

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form). """
    d = ham.shape[0]
    sparse = issparse(ham)
    evo = -1.0j * ((ham & eye(d, sparse=sparse)) -
                   (eye(d, sparse=sparse) & ham.T))

    def rho_dot(t, y):
        return evo.dot(y)

    return rho_dot


def lindblad_eq(ham, ls, gamma):
    """ Lindblad equation, but with flattened input/output.

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution
        ls: lindblad operators
        gamma: dampening strength

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form). """
    d = ham.shape[0]
    lls = [l.H @ l for l in ls]

    def gen_l_terms(rho):
        for l, ll in zip(ls, lls):
            yield (l @ rho @ l.H) - 0.5 * ((rho @ ll) + (ll @ rho))

    def rho_dot(t, y):
        rho = y.reshape(d, d)
        rd = -1.0j * (ham @ rho - rho @ ham)
        rd += gamma * sum(gen_l_terms(rho))
        return np.asarray(rd).reshape(-1)

    return rho_dot


def lindblad_eq_vec(ham, ls, gamma, sparse=False):
    """ Lindblad equation, but with flattened input/output and vectorised
    superoperation mode (no reshaping required).

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution
        ls: lindblad operators
        gamma: dampening strength

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form). """
    d = ham.shape[0]
    ham_sparse = issparse(ham) or sparse
    evo = -1.0j * ((ham & eye(d, sparse=ham_sparse)) -
                   (eye(d, sparse=ham_sparse) & ham.T))

    def gen_lb_terms():
        for l in ls:
            lb_sparse = issparse(l) or sparse
            yield ((l & l.conj()) -
                   0.5*((eye(d, sparse=lb_sparse) & (l.H @ l).T) +
                        ((l.H @ l) & eye(d, sparse=lb_sparse))))

    evo += gamma * sum(gen_lb_terms())

    def rho_dot(t, y):
        return evo.dot(y)

    return rho_dot


# --------------------------------------------------------------------------- #
# Quantum Evolution Class                                                     #
# --------------------------------------------------------------------------- #

class QuEvo(object):
    # TODO: solout method with funcyt
    # TODO: QuEvoTimeDepend
    """
    A class for evolving quantum systems according to schro equation
    Note that vector states are converted to kets always.
    """
    def __init__(self,
                 p0,
                 ham,
                 solve=False,
                 t0=0,
                 small_step=False):
        """
        Parameters
        ----------
            p0: inital state, either vector or operator
            ham: Governing Hamiltonian, can be tuple (eigvals, eigvecs)
            dop: whether to force evolution as density operator
            solve: whether to immediately solve hamiltonian
            t0: initial time (i.e. time of state p0)
            l: eigenvalues if ham already solved
            v: eigevvectors if ham already solved

        Members
        -------
            t: current time
            pt: current state
        """
        super(QuEvo, self).__init__()

        self.p0 = qjf(p0)
        self.t0 = t0  # initial time
        self.dop = isop(self.p0)
        self.t = t0  # current time
        self.d = p0.shape[0]  # Hilbert space dimensions

        # Set Hamiltonian and infer if solved already from tuple
        if type(ham) is tuple:
            # Eigendecomposition already supplied
            self.l, self.v = ham
            self.solve_ham()
        elif solve:
            # Eigendecomposition not supplied, but wanted
            self.ham = ham
            self.solve_ham()
            self.solved = True
        else:
            # Use definite integration to solve schrodinger equation
            self.ham = ham
            if self.dop:
                self.stepper = complex_ode(schrodinger_eq_dop(self.ham))
            else:
                self.stepper = complex_ode(schrodinger_eq_ket(self.ham))
            if small_step:
                self.stepper.set_integrator(
                    'dopri5', nsteps=1000,
                    first_step=norm(self.ham, 'f') / 150)
            else:
                self.stepper.set_integrator(
                    'dop853', nsteps=1000,
                    first_step=norm(self.ham, 'f') / 50)
            self.stepper.set_initial_value(np.asarray(p0).reshape(-1), t0)
            self.solved = False

    def solve_ham(self):
        try:  # Already set from tuple
            self.l, self.v
        except AttributeError:
            self.l, self.v = eigsys(self.ham)
        # Find initial state in energy eigenbasis at t0
        if self.dop:
            self.pe0 = self.v.H @ self.p0 @ self.v
        else:
            self.pe0 = self.v.H @ self.p0
        self._pt = self.p0  # Current state (start with same as initial)
        self.solved = True

    @property
    def pt(self):
        if self.solved:
            return self._pt
        else:
            return np.asmatrix(self.stepper.y.reshape(self.d, -1))

    def update_to(self, t):
        self.t = t
        if self.solved:
            exptl = np.exp((-1.0j * (self.t - self.t0)) * self.l)
            if self.dop:
                lvpvl = rdmul(ldmul(exptl, self.pe0), exptl.conj())
                self._pt = self.v @ lvpvl @ self.v.H
            else:
                self._pt = self.v @ ldmul(exptl, self.pe0)
        else:
            self.stepper.integrate(t)
