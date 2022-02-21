import numpy as np

from ..utils import ensure_dict, continuous_progbar, deprecated
from ..utils import progbar as Progbar
from .array_ops import norm_fro
from .tensor_arbgeom_tebd import LocalHamGen


class LocalHam1D(LocalHamGen):
    """An simple interacting hamiltonian object used, for instance, in TEBD.
    Once instantiated, the ``LocalHam1D`` hamiltonian stores a single term
    per pair of sites, cached versions of which can be retrieved like
    ``H.get_gate_expm((i, i + 1), -1j * 0.5)`` etc.

    Parameters
    ----------
    L : int
        The size of the hamiltonian.
    H2 : array_like or dict[tuple[int], array_like]
        The sum of interaction terms. If a dict is given, the keys should be
        nearest neighbours like ``(10, 11)``, apart from any default term which
        should have the key ``None``, and the values should be the sum of
        interaction terms for that interaction.
    H1 : array_like or dict[int, array_like], optional
        The sum of single site terms. If a dict is given, the keys should be
        integer sites, apart from any default term which should have the key
        ``None``, and the values should be the sum of single site terms for
        that site.
    cyclic : bool, optional
        Whether the hamiltonian has periodic boundary conditions or not.

    Attributes
    ----------
    terms : dict[tuple[int], array]
        The terms in the hamiltonian, combined from the inputs such that there
        is a single term per pair.

    Examples
    --------
    A simple, translationally invariant, interaction-only ``LocalHam1D``::

        >>> XX = pauli('X') & pauli('X')
        >>> YY = pauli('Y') & pauli('Y')
        >>> ham = LocalHam1D(L=100, H2=XX + YY)

    The same, but with a translationally invariant field as well::

        >>> Z = pauli('Z')
        >>> ham = LocalHam1D(L=100, H2=XX + YY, H1=Z)

    Specifying a default interaction and field, with custom values set for some
    sites::

        >>> H2 = {None: XX + YY, (49, 50): (XX + YY) / 2}
        >>> H1 = {None: Z, 49: 2 * Z, 50: 2 * Z}
        >>> ham = LocalHam1D(L=100, H2=H2, H1=H1)

    Specifying the hamiltonian entirely through site specific interactions and
    fields::

        >>> H2 = {(i, i + 1): XX + YY for i in range(99)}
        >>> H1 = {i: Z for i in range(100)}
        >>> ham = LocalHam1D(L=100, H2=H2, H1=H1)

    See Also
    --------
    SpinHam1D
    """

    def __init__(self, L, H2, H1=None, cyclic=False):
        self.L = int(L)
        self.cyclic = cyclic

        # parse two site terms
        if hasattr(H2, 'shape'):
            # use as default nearest neighbour term
            H2 = {None: H2}
        else:
            H2 = dict(H2)

        default_H2 = H2.pop(None, None)
        if default_H2 is not None:
            for i in range(self.L + int(self.cyclic) - 1):
                coo_a = i
                coo_b = (i + 1) % self.L
                if (coo_a, coo_b) not in H2 and (coo_b, coo_a) not in H2:
                    H2[coo_a, coo_b] = default_H2

        super().__init__(H2=H2, H1=H1)

    def mean_norm(self):
        """Computes the average frobenius norm of local terms.
        """
        return sum(
            norm_fro(h)
            for h in self.terms.values()
        ) / len(self.terms)

    def __repr__(self):
        return f"<LocalHam1D(L={self.L}, cyclic={self.cyclic})>"


NNI = deprecated(LocalHam1D, 'NNI', 'LocalHam1D')


class TEBD:
    """Class implementing Time Evolving Block Decimation (TEBD) [1].

    [1] Guifré Vidal, Efficient Classical Simulation of Slightly Entangled
    Quantum Computations, PRL 91, 147902 (2003)

    Parameters
    ----------
    p0 : MatrixProductState
        Initial state.
    H : LocalHam1D or array_like
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
    imag : bool, optional
        Enable imaginary time evolution. Defaults to false.

    See Also
    --------
    quimb.Evolution
    """

    def __init__(self, p0, H, dt=None, tol=None, t0=0.0,
                 split_opts=None, progbar=True, imag=False):
        # prepare initial state
        self._pt = p0.copy()
        self._pt.canonize(0)
        self.L = self._pt.L

        # handle hamiltonian -> convert array to LocalHam1D
        if isinstance(H, np.ndarray):
            H = LocalHam1D(L=self.L, H2=H, cyclic=p0.cyclic)

        if not isinstance(H, LocalHam1D):
            raise TypeError("``H`` should be a ``LocalHam1D`` or 2-site "
                            "array, not a TensorNetwork of any form.")

        if p0.cyclic != H.cyclic:
            raise ValueError("Both ``p0`` and ``H`` should have matching OBC "
                             "or PBC.")

        self.H = H
        self.cyclic = H.cyclic
        self._ham_norm = H.mean_norm()
        self._err = 0.0

        # set time and tolerance defaults
        self.t0 = self.t = t0
        if dt and tol:
            raise ValueError("Can't set default for both ``dt`` and ``tol``.")
        self.dt = self._dt = dt
        self.tol = tol
        self.imag = imag

        # misc other options
        self.progbar = progbar
        self.split_opts = ensure_dict(split_opts)

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

    def _get_gate_from_ham(self, dt_frac, sites):
        """Get the unitary (exponentiated) gate for fraction of timestep
        ``dt_frac`` and sites ``sites``, cached.
        """
        imag_factor = 1.0 if self.imag else 1.0j
        return self.H.get_gate_expm(sites, -imag_factor * self._dt * dt_frac)

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
            start_site_ind = 0
            final_site_ind = self.L - 1
            # Apply even gates:
            #
            #     o-<-<-<-<-<-<-<-<-<-   -<-<
            #     | | | | | | | | | |     | |       >~>~>~>~>~>~>~>~>~>~>~o
            #     UUU UUU UUU UUU UUU ... UUU  -->  | | | | | | | | | | | |
            #     | | | | | | | | | |     | |
            #      1   2   3   4   5  ==>
            #
            for i in range(start_site_ind, final_site_ind, 2):
                sites = (i, (i + 1) % self.L)
                U = self._get_gate_from_ham(dt_frac, sites)
                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                self._pt.gate_split_(
                    U, where=sites, absorb='right', **self.split_opts)

            if (self.L % 2 == 1):
                self._pt.left_canonize_site(self.L - 2)
                if self.cyclic:
                    sites = (self.L - 1, 0)
                    U = self._get_gate_from_ham(dt_frac, sites)
                    self._pt.right_canonize_site(1)
                    self._pt.gate_split_(
                        U, where=sites, absorb='left', **self.split_opts)

        elif direction == 'left':

            if self.cyclic and (self.L % 2 == 0):
                sites = (self.L - 1, 0)
                U = self._get_gate_from_ham(dt_frac, sites)
                self._pt.right_canonize_site(1)
                self._pt.gate_split_(
                    U, where=sites, absorb='left', **self.split_opts)

            final_site_ind = 1
            # Apply odd gates:
            #
            #     >->->-   ->->->->->->->->-o
            #     | | |     | | | | | | | | |       o~<~<~<~<~<~<~<~<~<~<~<
            #     | UUU ... UUU UUU UUU UUU |  -->  | | | | | | | | | | | |
            #     | | |     | | | | | | | | |
            #           <==  4   3   2   1
            #
            for i in reversed(range(final_site_ind, self.L - 1, 2)):
                sites = (i, (i + 1) % self.L)
                U = self._get_gate_from_ham(dt_frac, sites)
                self._pt.right_canonize(
                    start=min(self.L - 1, i + 2), stop=i + 1)
                self._pt.gate_split_(
                    U, where=sites, absorb='left', **self.split_opts)

            # one extra canonicalization not included in last split
            self._pt.right_canonize_site(1)

        # Renormalise after imaginary time evolution
        if self.imag:
            factor = self._pt[final_site_ind].norm()
            self._pt[final_site_ind] /= factor

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
        progbar = continuous_progbar(self.t, T) if progbar else None

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
        msg = f"t={self.t:.4g}, max-bond={self._pt.max_bond()}"
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
            ts = Progbar(ts)

        for t in ts:
            self.update_to(t, dt=dt, tol=False, order=order, progbar=False)

            if progbar:
                self._set_progbar_desc(ts)

            yield self.pt


def OTOC_local(psi0, H, H_back, ts, i, A, j=None, B=None,
               initial_eigenstate='check', **tebd_opts):
    """ The out-of-time-ordered correlator (OTOC) generating by two local
    operator A and B acting on site 'i', note it's a function of time.

    Parameters
    ----------
    psi0 : MatrixProductState
        The initial state in MPS form.
    H : LocalHam1D
        The Hamiltonian for forward time-evolution.
    H_back : LocalHam1D
        The Hamiltonian for backward time-evolution, should have only
        sign difference with 'H'.
    ts : sequence of float
        The time to evolve to.
    i : int
        The site where the local operators acting on.
    A : array
        The operator to act with.
    initial_eigenstate: {'check', Flase, True}
        To check the psi0 is or not eigenstate of operator B. If psi0 is the
        eigenstate of B, it will run a simpler version of OTOC calculation
        automatically.

    Returns
    ----------
    The OTOC <A(t)B(0)A(t)B(0)>
    """

    if B is None:
        B = A
    if j is None:
        j = i

    if initial_eigenstate == 'check':
        psi = psi0.gate(B, j, contract=True)
        x = psi0.H.expec(psi)
        y = psi.H.expec(psi)
        if abs(x**2 - y) < 1e-10:
            initial_eigenstate = True
        else:
            initial_eigenstate = False

    if initial_eigenstate is True:
        tebd1 = TEBD(psi0, H, **tebd_opts)
        x = psi0.H.expec(psi0.gate(B, j, contract=True))
        for t in ts:
            # evolve forward
            tebd1.update_to(t)
            # apply first A-gate
            psi_t_A = tebd1.pt.gate(A, i, contract=True)
            # evolve backwards
            tebd2 = TEBD(psi_t_A, H_back, **tebd_opts)
            tebd2.update_to(t)
            # compute expectation with second B-gate
            psi_f = tebd2.pt
            yield x * psi_f.H.expec(psi_f.gate(B, j, contract=True))
    else:
        # set the initial TEBD and apply the first operator A to right
        psi0_L = psi0
        tebd1_L = TEBD(psi0_L, H, **tebd_opts)

        psi0_R = psi0.gate(B, j, contract=True)
        tebd1_R = TEBD(psi0_R, H, **tebd_opts)

        for t in ts:
            # evolve forward
            tebd1_L.update_to(t)
            tebd1_R.update_to(t)

            # apply the opertor A to both left and right states
            psi_t_L_A = tebd1_L.pt.gate(A, i, contract=True)
            psi_t_R_A = tebd1_R.pt.gate(A.H, i, contract=True)

            # set the second left and right TEBD
            tebd2_L = TEBD(psi_t_L_A, H_back, **tebd_opts)
            tebd2_R = TEBD(psi_t_R_A, H_back, **tebd_opts)

            # evolve backwards
            tebd2_L.update_to(t)
            tebd2_R.update_to(t)

            # apply the laste operator B to left and compute overlap
            psi_f_L = tebd2_L.pt.gate(B.H, j, contract=True)
            psi_f_R = tebd2_R.pt
            yield psi_f_L.H.expec(psi_f_R)
