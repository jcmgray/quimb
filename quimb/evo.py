"""Easy and efficient time evolutions.

Contains an evolution class, Evolution to easily and efficiently manage time
evolution of quantum states according to the Schrodinger equation,
and related functions.
"""

import functools

import numpy as np
from scipy.integrate import complex_ode

from .core import (qarray, isop, ldmul, rdmul, explt,
                   dot, issparse, qu, eye, dag)
from .linalg.base_linalg import eigh, norm, expm_multiply
from .utils import continuous_progbar, progbar


# --------------------------------------------------------------------------- #
# Quantum evolution equations                                                 #
# --------------------------------------------------------------------------- #
#
# This are mostly just to be used internally with the integrators


def schrodinger_eq_ket(ham):
    """Wavefunction schrodinger equation.

    Parameters
    ----------
    ham : operator
        Time-independant Hamiltonian governing evolution.

    Returns
    -------
    psi_dot(t, y) : callable
        Function to calculate psi_dot(t) at psi(t).
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
    ham : operator
        Time-independant Hamiltonian governing evolution.

    Returns
    -------
    rho_dot(t, y) : callable
        Function to calculate rho_dot(t) at rho(t), input and
        output both in ravelled (1D form).
    """
    d = ham.shape[0]

    def rho_dot(_, y):
        hrho = dot(ham, y.reshape(d, d))
        return -1.0j * (hrho - hrho.T.conj()).reshape(-1)

    return rho_dot


def schrodinger_eq_dop_vectorized(ham):
    """Density operator schrodinger equation, but with flattened input/output
    and vectorised superoperator mode (no reshaping required).

    Note that this is probably only more efficient for sparse Hamiltonians.

    Parameters
    ----------
    ham: time-independant hamiltonian governing evolution

    Returns
    -------
    rho_dot(t, y) : callable
        Function to calculate rho_dot(t) at rho(t), input and
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
    ham : operator
        Time-independant hamiltonian governing evolution.
    ls : sequence of matrices
        Lindblad operators.
    gamma : float
        Dampening strength.

    Returns
    -------
    rho_dot(t, y) : callable
        Function to calculate rho_dot(t) at rho(t), input and
        output both in ravelled (1D form).
    """
    d = ham.shape[0]
    lls = tuple(dot(dag(l), l) for l in ls)

    def gen_l_terms(rho):
        for l, ll in zip(ls, lls):
            yield (dot(l, dot(rho, dag(l))) -
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
    ham : operator
        Time-independant hamiltonian governing evolution.
    ls : sequence of matrices
        Lindblad operators.
    gamma : float
        Dampening strength.

    Returns
    -------
    rho_dot(t, y) : callable
        Function to calculate rho_dot(t) at rho(t), input and
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
            yield ((l & l.conj()) - 0.5 * ((idt & dot(dag(l), l).T) +
                                           (dot(dag(l), l) & idt)))

    evo_superop += gamma * sum(gen_lb_terms())

    def rho_dot(_, y):
        return dot(evo_superop, y)
    return rho_dot


def _calc_evo_eq(isdop, issparse, isopen=False):
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

class Evolution(object):
    """A class for evolving quantum systems according to Schrodinger equation.

    The evolution can be performed in a number of ways:

        - diagonalise the Hamiltonian (or use already diagonalised system).
        - integrate the complex ODE, that is, the Schrodinger equation, using
          scipy. Here either a mid- or high-order Dormand-Prince adaptive
          time stepping scheme is used (see
          :class:`scipy.integrate.complex_ode`).

    Parameters
    ----------
    p0 : quantum state
        Inital state, either vector or operator. If vector, converted to ket.
    ham : operator, or tuple (1d array, operator).
        Governing Hamiltonian, if tuple then assumed to contain
        ``(eigvals, eigvecs)`` of presolved system.
    t0 : float, optional
        Initial time (i.e. time of state ``p0``), defaults to zero.
    compute : callable, or dict of callable, optional
        Function(s) to compute on the state at each time step, called
        with args (t, pt). If supplied with:

            - single callable : ``Evolution.results`` will contain the results
              as a list,
            - dict of callables : ``Evolution.results`` will contain the
              results as a dict of lists with corresponding keys to those
              given in ``compute``.

    method : {'integrate', 'solve', 'expm'}
        How to evolve the system:

            - ``'integrate'``: use definite integration. Get system at each
              time step, only need action of Hamiltonian on state. Generally
              efficient.
            - ``'solve'``: diagonalise dense hamiltonian. Best for small
              systems and allows arbitrary time steps without loss of
              precision.
            - ``'expm'``: compute the evolved state using the action of the
              operator exponential in a 'single shot' style. Only needs action
              of Hamiltonian, for very large systems can use distributed MPI.

    int_small_step : bool, optional
        If ``method='integrate'``, whether to use a low or high order
        integrator to give naturally small or large steps.
    expm_backend : {'auto', 'scipy', 'slepc'}
        How to perform the expm_multiply function if ``method='expm'``. Can
        further specifiy ``'slepc-krylov'``, or ``'slepc-expokit'``.
    expm_opts : dict
        Supplied to :func:`~quimb.linalg.base_linalg.expm_multiply`
        function if ``method='expm'``.
    progbar : bool, optional
        Whether to show a progress bar when calling ``at_times`` or integrating
        with the ``update_to`` method.
    """

    def __init__(self, p0, ham, t0=0,
                 compute=None,
                 method='integrate',
                 int_small_step=False,
                 expm_backend='AUTO',
                 expm_opts=None,
                 progbar=False):

        self._p0 = qu(p0)
        self._t = self.t0 = t0
        self._isdop = isop(self._p0)  # Density operator evolution?
        self._d = p0.shape[0]  # Hilbert space dimension
        self._progbar = progbar

        self._setup_callback(compute)
        self._method = method

        if method == 'solve' or isinstance(ham, (tuple, list)):
            self._solve_ham(ham)
        elif method == 'integrate':
            self._start_integrator(ham, int_small_step)
        elif method == 'expm':
            self._update_method = self._update_to_expm_ket
            self._pt = self._p0
            self.ham = ham
            self.expm_backend = expm_backend
            self.expm_opts = {} if expm_opts is None else dict(expm_opts)
        else:
            raise ValueError("Did not understand evolution method: '{}'."
                             .format(method))

    def _setup_callback(self, fn):
        """Setup callbacks in the correct place to compute into _results
        """
        if fn is None:
            step_callback = None
            if not self._progbar:
                int_step_callback = None
            else:
                def int_step_callback(t, y):
                    pass

        # dict of funcs input -> dict of funcs output
        elif isinstance(fn, dict):
            self._results = {k: [] for k in fn}

            @functools.wraps(fn)
            def step_callback(t, pt):
                for k, v in fn.items():
                    self._results[k].append(v(t, pt))

            # For the integration callback, additionally need to convert
            #   back to 'quantum' (column vector) form
            @functools.wraps(fn)
            def int_step_callback(t, y):
                pt = qarray(y.reshape(self._d, -1))
                for k, v in fn.items():
                    self._results[k].append(v(t, pt))

        # else results -> single list of outputs of fn
        else:
            self._results = []

            @functools.wraps(fn)
            def step_callback(t, pt):
                self._results.append(fn(t, pt))

            @functools.wraps(fn)
            def int_step_callback(t, y):
                pt = qarray(y.reshape(self._d, -1))
                self._results.append(fn(t, pt))

        self._step_callback = step_callback
        self._int_step_callback = int_step_callback

    def _solve_ham(self, ham):
        """Solve the supplied hamiltonian and find the initial state in the
        energy eigenbasis for quick evolution later.
        """
        # See if already solved from tuple
        try:
            self._evals, self._evecs = ham
            self._method = 'solve'
        except ValueError:
            self._evals, self._evecs = eigh(ham.A)

        # Find initial state in energy eigenbasis at t0
        if self._isdop:
            self.pe0 = dot(dag(self._evecs), dot(self._p0, self._evecs))
            self._update_method = self._update_to_solved_dop
        else:
            self.pe0 = dot(dag(self._evecs), self._p0)
            self._update_method = self._update_to_solved_ket

        # Current state (start with same as initial)
        self._pt = self._p0
        self._solved = True

    def _start_integrator(self, ham, small_step):
        """Initialize a stepping integrator.
        """
        self._sparse_ham = issparse(ham)

        # set complex ode with governing equation
        evo_eq = _calc_evo_eq(self._isdop, self._sparse_ham)
        self._stepper = complex_ode(evo_eq(ham))

        # 5th order stpper or 8th order stepper
        int_mthd, step_fct = ('dopri5', 150) if small_step else ('dop853', 50)
        first_step = norm(ham, 'f') / step_fct

        self._stepper.set_integrator(int_mthd, nsteps=0, first_step=first_step)

        # Set step_callback to be evaluated with args (t, y) at each step
        if self._int_step_callback is not None:
            self._stepper.set_solout(self._int_step_callback)

        self._stepper.set_initial_value(self._p0.A.reshape(-1), self.t0)

        # assign the correct update_to method
        self._update_method = self._update_to_integrate
        self._solved = False

    # Methods for updating the simulation ----------------------------------- #

    def _update_to_expm_ket(self, t):
        """Update the simulation to time ``t``, without explicitly computing
        the operator exponential itself.
        """
        factor = -1j * (t - self.t)
        self._pt = expm_multiply(factor * self.ham, self._pt,
                                 backend=self.expm_backend, **self.expm_opts)
        self._t = t

        # compute any callbacks into -> self._results
        if self._step_callback is not None:
            self._step_callback(t, self._pt)

    def _update_to_solved_ket(self, t):
        """Update simulation consisting of a solved hamiltonian and a
        wavefunction to time `t`.
        """
        self._t = t
        lt = explt(self._evals, t - self.t0)
        self._pt = self._evecs @ ldmul(lt, self.pe0)

        # compute any callbacks into -> self._results
        if self._step_callback is not None:
            self._step_callback(t, self._pt)

    def _update_to_solved_dop(self, t):
        """Update simulation consisting of a solved hamiltonian and a
        density operator to time `t`.
        """
        self._t = t
        lt = explt(self._evals, t - self.t0)
        lvpvl = rdmul(ldmul(lt, self.pe0), lt.conj())
        self._pt = self._evecs @ (lvpvl @ dag(self._evecs))

        # compute any callbacks into -> self._results
        if self._step_callback is not None:
            self._step_callback(t, self._pt)

    def _update_to_integrate(self, t):
        """Update simulation consisting of unsolved hamiltonian.
        """
        self._stepper.integrate(t)

    def update_to(self, t):
        """Update the simulation to time ``t`` using relevant method.

        Parameters
        ----------
        t : float
            Time to update the evolution to.
        """
        if self._progbar and hasattr(self, '_stepper'):

            with continuous_progbar(self.t, t) as pbar:
                if self._int_step_callback is not None:
                    def solout(t, y):
                        self._int_step_callback(t, y)
                        pbar.cupdate(t)
                else:
                    def solout(t, _):
                        pbar.cupdate(t)

                self._stepper.set_solout(solout)
                self._update_method(t)
        else:
            self._update_method(t)

    def at_times(self, ts):
        """Generator expression to yield state af list of times.

        Parameters
        ----------
        ts : sequence of floats
            Times at which to evolve to, then yield the state.

        Yields
        ------
        pt : quantum state
            Quantum state of evolution at next time in ``ts``.

        Notes
        -----
        If integrating, currently any compute callbacks will be called at every
        *integration* step, not just the times `ts` -- i.e. in general
        len(Evolution.results) != len(ts) and if the adaptive step times are
        needed they should be added as a callback, e.g.
        ``compute['t'] = lambda t, _: return t``.
        """
        if self._progbar:
            ts = progbar(ts)

        for t in ts:
            self._update_method(t)
            yield self.pt

    # Simulation properties ------------------------------------------------- #

    @property
    def t(self):
        """float : Current time of simulation.
        """
        return self._stepper.t if self._method == 'integrate' else self._t

    @property
    def pt(self):
        """quantum state : State of the system at the current time (t).
        """
        if self._method == 'integrate':
            return qarray(self._stepper.y.reshape(self._d, -1))
        else:
            return self._pt

    @property
    def results(self):
        """list, or dict of lists, optional : Results of the compute
        callback(s) for each time step.
        """
        return self._results
