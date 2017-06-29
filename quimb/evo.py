"""
Contains an evolution class, QuEvo to easily and efficiently manage time
evolution of quantum states according to the Schrodinger equation,
and related functions.
"""

# TODO: setter/update pt
# TODO: setter for compute?
# TODO: test known lindlbad evolution
# TODO: QuEvoTimeDepend?
# TODO: slepc krylov method

import functools

import numpy as np
from scipy.integrate import complex_ode

from .accel import isop, ldmul, rdmul, explt, dot, issparse, _dot_dense
from .core import qu, eye
from .solve.base_solver import eigsys, norm
from .utils import continuous_progbar, progbar


# --------------------------------------------------------------------------- #
# Quantum evolution equations                                                 #
# --------------------------------------------------------------------------- #

def schrodinger_eq_ket(ham):
    """Wavefunction schrodinger equation.

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution

    Returns
    -------
        psi_dot(t, y): function to calculate psi_dot(t) at psi(t).
    """
    def psi_dot(_, y):
        return -1.0j * dot(ham, y)

    return psi_dot


def schrodinger_eq_dop(ham):
    """Density operator schrodinger equation, but with flattened input/output.
    Note that this assumes both `ham` and `rho` are hermitian in order to speed
    up the commutator, non-hermitian hamiltonians as used to model loss should
    be treated explicilty or with `schrodinger_eq_dop_vectorized`.

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form).
    """
    d = ham.shape[0]

    def rho_dot(_, y):
        hrho = dot(ham, y.reshape(d, d))
        return -1.0j * (hrho - hrho.T.conj()).reshape(-1)

    return rho_dot


def schrodinger_eq_dop_vectorized(ham):
    """Density operator schrodinger equation, but with flattened input/output
    and vectorised superoperation mode (no reshaping required).

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form).
    """
    d = ham.shape[0]
    sparse = issparse(ham)
    idt = eye(d, sparse=sparse)
    evo_superop = -1.0j * ((ham & idt) - (idt & ham.T))

    def rho_dot(_, y):
        return dot(evo_superop, y)
    return rho_dot


def lindblad_eq(ham, ls, gamma):
    """Lindblad equation, but with flattened input/output.

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution
        ls: lindblad operators
        gamma: dampening strength

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form).
    """
    d = ham.shape[0]
    lls = tuple(dot(l.H, l) for l in ls)

    def gen_l_terms(rho):
        for l, ll in zip(ls, lls):
            yield (dot(l, dot(rho, l.H)) -
                   0.5 * (dot(rho, ll) + dot(ll, rho)))

    def rho_dot(_, y):
        rho = y.reshape(d, d)
        rho_d = dot(ham, rho)
        rho_d -= rho_d.T.conj()
        rho_d *= -1.0j
        rho_d += gamma * sum(gen_l_terms(rho))
        return np.asarray(rho_d).reshape(-1)

    return rho_dot


def lindblad_eq_vectorized(ham, ls, gamma, sparse=False):
    """Lindblad equation, but with flattened input/output and vectorised
    superoperation mode (no reshaping required).

    Parameters
    ----------
        ham: time-independant hamiltonian governing evolution
        ls: lindblad operators
        gamma: dampening strength

    Returns
    -------
        rho_dot(t, y): function to calculate rho_dot(t) at rho(t), input and
            output both in ravelled (1D form).
    """
    d = ham.shape[0]
    ham_sparse = issparse(ham) or sparse
    idt = eye(d, sparse=ham_sparse)
    evo_superop = -1.0j * ((ham & idt) - (idt & ham.T))

    def gen_lb_terms():
        for l in ls:
            lb_sparse = issparse(l) or sparse
            idt = eye(d, sparse=lb_sparse)
            yield ((l & l.conj()) - 0.5 * ((idt & dot(l.H, l).T) +
                                           (dot(l.H, l) & idt)))
    evo_superop += gamma * sum(gen_lb_terms())

    def rho_dot(_, y):
        return dot(evo_superop, y)
    return rho_dot


def calc_evo_eq(isdop, issparse, isopen=False):
    """Choose an appropirate dynamical equation to evolve with.
    """
    eq_chooser = {
        (0, 0, 0): schrodinger_eq_ket,
        (0, 1, 0): schrodinger_eq_ket,
        (1, 0, 0): schrodinger_eq_dop,
        (1, 1, 0): schrodinger_eq_dop_vectorized,
        (1, 0, 1): lindblad_eq,
        (1, 1, 1): lindblad_eq_vectorized,
    }
    return eq_chooser[(isdop, issparse, isopen)]


# --------------------------------------------------------------------------- #
# Quantum Evolution Class                                                     #
# --------------------------------------------------------------------------- #

class QuEvo(object):
    """A class for evolving quantum systems according to schro equation
    Note that vector states are converted to kets always.
    """

    def __init__(self, p0, ham,
                 solve=False,
                 t0=0,
                 small_step=False,
                 compute=None,
                 progbar=False):
        """
        Parameters
        ----------
            p0 : quantum state
                inital state, either vector or operator
            ham : matrix-like, or tuple (1d array, mattix-like).
                Governing Hamiltonian, if tuple then assumed to contain
                (eigvals, eigvecs) of presolved system.
            solve : bool, optional
                Whether to immediately solve hamiltonian.
            t0 : float, optional
                Initial time (i.e. time of state p0), defaults to zero.
            small_step : bool, optional
                If integrating, whether to use a low or high order
                integrator to give naturally small or large steps.
            compute : callable, or dict of callable
                Function(s) to compute on the state at each time step, called
                with args (t, pt). If supplied with:
                    * single callable, the results will be a list stored in
                        QuEvo.results
                    * dict of callables, results will a dict of lists with
                        corresponding keys in QuEvo.results.

        Members
        -------
            t : float
                Current time.
            pt : quantum state
                Current state.
            results : list, or dict of lists, optional
                Results of the compute callback(s) for each time step.
        """
        super(QuEvo, self).__init__()

        self.p0 = qu(p0)
        self._t = self.t0 = t0
        self.isdop = isop(self.p0)  # Density operator evolution?
        self.d = p0.shape[0]  # Hilbert space dimension

        self.progbar = progbar
        self._setup_callback(compute)

        # Hamiltonian
        if solve or isinstance(ham, (tuple, list)):
            self._solve_ham(ham)
        else:  # Use definite integration
            self._start_integrator(ham, small_step)

    def _setup_callback(self, fn):
        """
        """
        if fn is None:
            slv_step_callback = None
            int_step_callback = None

        # dict of funcs input -> dict of funcs output
        elif isinstance(fn, dict):
            self.results = {k: [] for k in fn}

            @functools.wraps(fn)
            def slv_step_callback(t, pt):
                for k, v in fn.items():
                    self.results[k].append(v(t, pt))

            # For the integration callback, additionally need to convert
            #   back to 'quantum' form
            @functools.wraps(fn)
            def int_step_callback(t, y):
                pt = np.asmatrix(y.reshape(self.d, -1))
                for k, v in fn.items():
                    self.results[k].append(v(t, pt))

        # else results -> single list of outputs of fn
        else:
            self.results = []

            @functools.wraps(fn)
            def slv_step_callback(t, pt):
                self.results.append(fn(t, pt))

            @functools.wraps(fn)
            def int_step_callback(t, y):
                pt = np.asmatrix(y.reshape(self.d, -1))
                self.results.append(fn(t, pt))

        self._slv_step_callback = slv_step_callback
        self._int_step_callback = int_step_callback

    def _solve_ham(self, ham):
        """Solve the supplied hamiltonian and find the initial state in the
        energy eigenbasis for quick evolution later.
        """
        try:  # See if already set from tuple
            self.evals, self.evecs = ham
        except ValueError:
            self.evals, self.evecs = eigsys(ham.A)

        # Find initial state in energy eigenbasis at t0
        self.pe0 = (dot(self.evecs.H, dot(self.p0,
                                          self.evecs)) if self.isdop else
                    dot(self.evecs.H, self.p0))
        self._pt = self.p0  # Current state (start with same as initial)

        # Set update method conditional on type of state
        self._update_method = (self._update_to_solved_dop if self.isdop else
                               self._update_to_solved_ket)
        self.solved = True

    def _start_integrator(self, ham, small_step):
        """Initialize a stepping integrator.
        """
        self.sparse_ham = issparse(ham)

        # set complex ode with governing equation
        evo_eq = calc_evo_eq(self.isdop, self.sparse_ham)
        self.stepper = complex_ode(evo_eq(ham))

        # 5th order stpper or 8th order stepper
        int_mthd, step_fct = ('dopri5', 150) if small_step else ('dop853', 50)
        first_step = norm(ham, 'f') / step_fct

        self.stepper.set_integrator(int_mthd, nsteps=0, first_step=first_step)

        # Set step_callback to be evaluated with args (t, y) at each step
        if self._int_step_callback is not None:
            self.stepper.set_solout(self._int_step_callback)

        self.stepper.set_initial_value(self.p0.A.reshape(-1), self.t0)

        # assign the correct update_to method
        self._update_method = self._update_to_integrate
        self.solved = False

    # Methods for updating the simulation ----------------------------------- #

    def _update_to_solved_ket(self, t):
        """Update simulation consisting of a solved hamiltonian and a
        wavefunction to time `t`.
        """
        self._t = t
        lt = explt(self.evals, t - self.t0)
        self._pt = _dot_dense(self.evecs, ldmul(lt, self.pe0))

        # compute any callbacks into ->> self.results
        if self._slv_step_callback is not None:
            self._slv_step_callback(t, self._pt)

    def _update_to_solved_dop(self, t):
        """Update simulation consisting of a solved hamiltonian and a
        density operator to time `t`.
        """
        self._t = t
        lt = explt(self.evals, t - self.t0)
        lvpvl = rdmul(ldmul(lt, self.pe0), lt.conj())
        self._pt = _dot_dense(self.evecs, _dot_dense(lvpvl, self.evecs.H))

        # compute any callbacks into ->> self.results
        if self._slv_step_callback is not None:
            self._slv_step_callback(t, self._pt)

    def _update_to_integrate(self, t):
        """Update simulation consisting of unsolved hamiltonian.
        """
        self.stepper.integrate(t)

    def update_to(self, t):
        """Update the simulation to time `t`.
        """
        if self.progbar and hasattr(self, 'stepper'):
            with continuous_progbar(self.t, t) as pbar:
                # def here for the pbar closure
                def pbar_compute(fn):
                    @functools.wraps(fn)
                    def wrapped_fn(t, y):  # pragma: no cover
                        res = fn(t, y)
                        pbar.cupdate(t)
                        return res

                    return wrapped_fn

                self.stepper.set_solout(
                    pbar_compute(self._int_step_callback))
                self._update_method(t)
        else:
            self._update_method(t)

    def at_times(self, ts):
        """Generator expression that will yield the state at each of the times
        in `ts`.

        Notes
        -----
            If integrating (solve=False), currently any compute callbacks will
            be called at every *integration* step, not just the times `ts` --
            i.e. in general len(QuEvo.results) != len(ts).

        """
        if self.progbar:
            ts = progbar(ts)

        for t in ts:
            self._update_method(t)
            yield self.pt

    # Simulation properties ------------------------------------------------- #

    @property
    def t(self):
        """Current time of simulation.
        """
        return (self._t if self.solved else
                self.stepper.t)

    @property
    def pt(self):
        """Return the state of the system at the current time (t).
        """
        return (self._pt if self.solved else
                np.asmatrix(self.stepper.y.reshape(self.d, -1)))
