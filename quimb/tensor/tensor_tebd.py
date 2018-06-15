import numpy as np

import quimb as qu


class NNI:
    """An simple interacting hamiltonian object used, for instance, in TEBD.

    Parameters
    ----------
    H2 : array_like
        The sum of interaction terms.
    H1 : array_like, optional
        The sum of single site terms.
    n : int, optional
        The size of the hamiltonian.
    cyclic : bool, optional
        Whether the hamiltonian has periodic boundary conditions or not.
    """

    def __init__(self, H2, H1=None, n=None, cyclic=False):
        if (H1 is not None) and (n is None):
            raise ValueError("Need to specify length ``n`` if one "
                             "site term ``H1`` is supplied.")

        self.H1 = H1
        self.H2 = H2
        self.n = n
        self.cyclic = cyclic

        self._terms = {}

    def gen_term(self, sites=None):
        """Generate the interaction term for sites ``sites``.
        """
        term = self.H2

        # make sure have sites as (i, i + 1) if supplied
        if sites is not None:
            i, j = sorted(sites)
            if j - i != 1:
                raise ValueError("Only nearest neighbour interactions are "
                                 "supported for an ``NNI``.")

        # add single site term to left site if present
        if self.H1 is not None:
            I_2 = qu.eye(self.H1.shape[0], dtype=self.H1.dtype)
            term = term + qu.kron(self.H1, I_2)

            # for the last interaction, add to right site as well
            if sites and (i == self.n - 2) and (not self.cyclic):
                term = term + qu.kron(I_2, self.H1)

        return term

    def __call__(self, sites=None):
        """Get the cached term for sites ``sites``, generate if necessary.
        """
        try:
            return self._terms[sites]
        except KeyError:
            term = self.gen_term(sites)
            self._terms[sites] = term
            return term


class TEBD:
    """Class implementing Time Evolving Block Decimation (TEBD) [1].

    [1] Guifré Vidal, Efficient Classical Simulation of Slightly Entangled
    Quantum Computations, PRL 91, 147902 (2003)

    Parameters
    ----------
    p0 : MatrixProductState
        Initial state.
    H : NNI or array_like
        Dense hamiltonian representing the two body interaction. Should have
        shape ``(d * d, d * d)``, where ``d`` is the physical dimension of
        ``p0``.
    dt : float, optional
        Default time step, cannot be set as well as ``tol``.
    tol : float, optional
        Default target error for each evolution, cannot be set as well as
        ``dt``, which will instead be calculated from the trotter orderm length
        of time, and hamiltonian norm.
    t0 : float, optional
        Initial time. Defaults to 0.0.
    split_opts : dict, optional
        Compression options applied for splitting after gate application, see
        :func:`~quimb.tensor.tensor_core.tensor_split`.

    See Also
    --------
    quimb.Evolution
    """

    def __init__(self, p0, H, dt=None, tol=None, t0=0.0,
                 split_opts=None, progbar=True):
        # prepare initial state
        self._pt = p0.copy()
        self._pt.canonize(0)
        self.N = self._pt.nsites

        # handle hamiltonian -> convert array to NNI
        if isinstance(H, np.ndarray):
            H = NNI(H)
        if not isinstance(H, NNI):
            raise TypeError("``H`` should be a ``NNI`` or 2-site array, "
                            "not a TensorNetwork of any form.")
        self.H = H
        self._ham_norm = qu.norm(H(), 'fro')
        self._U_ints = {}
        self._err = 0.0

        # set time and tolerance defaults
        self.t0 = self.t = t0
        if dt and tol:
            raise ValueError("Can't set default for both ``dt`` and ``tol``.")
        self.dt = self._dt = dt
        self.tol = tol

        # misc other options
        self.progbar = progbar
        self.split_opts = {} if split_opts is None else dict(split_opts)

    @property
    def pt(self):
        """The MPS state of the system at the current time.
        """
        return self._pt.copy()

    @property
    def err(self):
        return self._err

    def choose_time_step(self, tol, T, order):
        """Trotter error is ``~ (T / dt) * dt^(order + 1)``. Invert to
        find desired time step, and scale by norm of interaction term.
        """
        return (tol / (T * self._ham_norm)) ** (1 / order)

    def get_gate(self, dt_frac, sites=None):
        """Get the unitary gate for fraction of timestep ``dt_frac`` and sites
        ``sites``, cached.
        """
        # XXX: currently trans-invar only -> reuse exp-gate apart from last
        if sites != (self.N - 2, self.N - 1):
            sites = None

        try:
            return self._U_ints[dt_frac, sites]
        except KeyError:
            U = qu.expm(-1.0j * self._dt * dt_frac * self.H(sites))
            self._U_ints[dt_frac, sites] = U
            return U

    def sweep(self, direction, dt_frac, dt=None, queue=False):
        """Perform a single sweep of gates and compression. This shifts the
        orthonognality centre along with the gates as they are applied and
        split.

        Parameters
        ----------
        direction : {'right', 'left'}
            Which direction to sweep. Right is even bonds, left is odd.
        dt_frac : float
            What fraction of dt substep to take.
        dt : float, optional
            Overide the current ``dt`` with a custom value.
        """

        # if custom dt set, scale the dt fraction
        if dt is not None:
            dt_frac *= (dt / self._dt)

        # ------ automatically combine consecutive sweeps of same time ------ #

        if not hasattr(self, '_queued_sweep'):
            self._queued_sweep = None

        if queue:
            # check for queued sweep
            if self._queued_sweep:
                # if matches, combine and continue
                if direction == self._queued_sweep[0]:
                    self._queued_sweep[1] += dt_frac
                    return
                # else perform the old, queue the new
                else:
                    new_queued_sweep = [direction, dt_frac]
                    direction, dt_frac = self._queued_sweep
                    self._queued_sweep = new_queued_sweep

            # just queue the new sweep
            else:
                self._queued_sweep = [direction, dt_frac]
                return

        # check if need to drain the queue first
        elif self._queued_sweep:
            queued_direction, queued_dt_frac = self._queued_sweep
            self._queued_sweep = None
            self.sweep(queued_direction, queued_dt_frac, queue=False)

        # ------------------------------------------------------------------- #

        if direction == 'right':
            # Apply even gates:
            #
            #     o-<-<-<-<-<-<-<-<-<-   -<-<
            #     | | | | | | | | | |     | |       >~>~>~>~>~>~>~>~>~>~>~o
            #     UUU UUU UUU UUU UUU ... UUU  -->  | | | | | | | | | | | |
            #     | | | | | | | | | |     | |
            #      1   2   3   4   5  ==>
            #
            for i in range(0, self.N - 1, 2):
                sites = (i, i + 1)
                U = self.get_gate(dt_frac, sites)
                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                self._pt.gate2split(
                    U, where=sites, absorb='right', **self.split_opts)

        elif direction == 'left':
            # Apply odd gates:
            #
            #     >->->-   ->->->->->->->->-o
            #     | | |     | | | | | | | | |       o~<~<~<~<~<~<~<~<~<~<~<
            #     | UUU ... UUU UUU UUU UUU |  -->  | | | | | | | | | | | |
            #     | | |     | | | | | | | | |
            #           <==  4   3   2   1
            #
            for i in reversed(range(1, self.N - 1, 2)):
                sites = (i, i + 1)
                U = self.get_gate(dt_frac, sites)
                self._pt.right_canonize(
                    start=min(self.N - 1, i + 2), stop=i + 1)
                self._pt.gate2split(
                    U, where=sites, absorb='left', **self.split_opts)

            # one extra canonicalization not included in last split
            self._pt.right_canonize_site(1)

    def _step_order2(self, tau=1, **sweep_opts):
        """Perform a single, second order step.
        """
        self.sweep('right', tau / 2, **sweep_opts)
        self.sweep('left', tau, **sweep_opts)
        self.sweep('right', tau / 2, **sweep_opts)

    def _step_order4(self, **sweep_opts):
        """Perform a single, fourth order step.
        """
        tau1 = tau2 = 1 / (4 * 4**(1 / 3))
        tau3 = 1 - 2 * tau1 - 2 * tau2
        self._step_order2(tau1, **sweep_opts)
        self._step_order2(tau2, **sweep_opts)
        self._step_order2(tau3, **sweep_opts)
        self._step_order2(tau2, **sweep_opts)
        self._step_order2(tau1, **sweep_opts)

    def step(self, order=2, dt=None, progbar=None, **sweep_opts):
        """Perform a single step of time ``self.dt``.
        """
        {2: self._step_order2,
         4: self._step_order4}[order](dt=dt, **sweep_opts)

        dt = self._dt if dt is None else dt
        self.t += dt
        self._err += self._ham_norm * dt ** (order + 1)

        if progbar is not None:
            progbar.cupdate(self.t)
            self._set_progbar_desc(progbar)

    def _compute_sweep_dt_tol(self, T, dt, tol, order):
        # Work out timestep, possibly from target tol, and checking defaults
        dt = self.dt if (dt is None) else dt
        tol = self.tol if (tol is None) else tol

        if not (dt or tol):
            raise ValueError("Must set one of ``dt`` and ``tol``.")
        if (dt and tol):
            raise ValueError("Can't set both ``dt`` and ``tol``.")

        if dt is None:
            self._dt = self.choose_time_step(tol, T - self.t, order)
        else:
            self._dt = dt

        return self._dt

    TARGET_TOL = 1e-13  # tolerance to have 'reached' target time

    def update_to(self, T, dt=None, tol=None, order=4, progbar=None):
        """Update the state to time ``T``.

        Parameters
        ----------
        T : float
            The time to evolve to.
        dt : float, optional
            Time step to use. Can't be set as well as ``tol``.
        tol : float, optional
            Tolerance for whole evolution. Can't be set as well as ``dt``.
        order : int, optional
            Trotter order to use.
        progbar : bool, optional
            Manually turn the progress bar off.
        """
        if T < self.t - self.TARGET_TOL:
            raise NotImplementedError

        self._compute_sweep_dt_tol(T, dt, tol, order)

        # set up progress bar and start evolution
        progbar = self.progbar if (progbar is None) else progbar
        progbar = qu.utils.continuous_progbar(self.t, T) if progbar else None

        while self.t < T - self.TARGET_TOL:
            if (T - self.t < self._dt):
                # set custom dt if within one step of final time
                dt = T - self.t
                # also make sure queued sweeps are drained
                queue = False
            else:
                dt = None
                queue = True

            # perform a step!
            self.step(order=order, progbar=progbar, dt=dt, queue=queue)

        if progbar:
            progbar.close()

    def _set_progbar_desc(self, progbar):
        msg = "t={:.4g}, max-bond={}".format(self.t, self._pt.max_bond())
        progbar.set_description(msg)

    def at_times(self, ts, dt=None, tol=None, order=4, progbar=None):
        """Generate the time evolved state at each time in ``ts``.

        Parameters
        ----------
        ts : sequence of float
            The times to evolve to and yield the state at.
        dt : float, optional
            Time step to use. Can't be set as well as ``tol``.
        tol : float, optional
            Tolerance for whole evolution. Can't be set as well as ``dt``.
        order : int, optional
            Trotter order to use.
        progbar : bool, optional
            Manually turn the progress bar off.

        Yields
        ------
        pt : MatrixProductState
            The state at each of the times in ``ts``. This is a copy of
            internal state used, so inplace changes can be made to it.
        """
        # convert ts to list, to to calc range and use progress bar
        ts = sorted(ts)
        T = ts[-1]

        # need to use dt always so tol applies over whole T sweep
        dt = self._compute_sweep_dt_tol(T, dt, tol, order)

        # set up progress bar
        progbar = self.progbar if (progbar is None) else progbar
        if progbar:
            ts = qu.utils.progbar(ts)

        for t in ts:
            self.update_to(t, dt=dt, tol=False, order=order, progbar=False)

            if progbar:
                self._set_progbar_desc(ts)

            yield self.pt
