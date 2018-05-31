import quimb as qu


class TEBD:
    """Class implementing Time Evolving Block Decimation (TEBD) [1].

    Parameters
    ----------
    p0 : MatrixProductState
        initial state
    ham_int : array_like
        Dense hamiltonian representing the two body interaction. Should have
        shape ``(d * d, d * d)``, where ``d`` is the physical dimension.
    dt : float
        Time step.
    t0 : float
        Initial time. Defaults to 0.0.
    split_opts : dict
        Compression options applied for splitting after gate application, see
        :func:`~quimb.tensor.tensor_core.tensor_split`.

    [1] Guifré Vidal, Efficient Classical Simulation of Slightly Entangled
    Quantum Computations, PRL 91, 147902 (2003)
    """

    def __init__(self, p0, ham_int, dt, t0=0.0,
                 split_opts=None, progbar=True):
        self.pt = p0.copy()
        self.ham_int = ham_int
        self.dt = dt
        self.t0 = self.t = t0
        self.progbar = progbar
        self.split_opts = {} if split_opts is None else dict(split_opts)

        self.pt.canonize(0)

        self.N = p0.nsites
        self.U_ints = {}

    def get_gate(self, dt_frac):
        """Get the unitary gate for fraction of timestep ``dt_frac``, cached.
        """
        try:
            return self.U_ints[dt_frac]
        except KeyError:
            U = qu.expm(-1.0j * self.dt * dt_frac * self.ham_int)
            self.U_ints[dt_frac] = U
            return U

    def sweep(self, direction, dt_frac):
        """Perform a single sweep of gates and compression. This shifts the
        orthonognality centre along with the gates as they are applied and
        split.

        Parameters
        ----------
        direction : {'right', 'left'}
            Which direction to sweep. Right is even bonds, left is odd.
        dt_frac : float
            What fraction of dt substep to take.
        """
        U = self.get_gate(dt_frac)
        N = self.N

        if direction == 'right':
            # Apply even gates:
            #
            #     o-<-<-<-<-<-<-<-<-<-<-<
            #     | | | | | | | | | | | |       >~>~>~>~>~>~>~>~>~>~>~o
            #     UUU UUU UUU UUU UUU UUU  -->  | | | | | | | | | | | |
            #     | | | | | | | | | | | |
            #      1   2   3   4   5  ==>
            #
            for i in range(0, N - 1, 2):
                self.pt.left_canonize(start=max(0, i - 1), stop=i)
                self.pt.gate2split(U, where=(i, i + 1),
                                   absorb='right', **self.split_opts)

        elif direction == 'left':
            # Apply odd gates:
            #
            #     >->->->->->->->->->->-o
            #     | | | | | | | | | | | |       o~<~<~<~<~<~<~<~<~<~<~<
            #     | UUU UUU UUU UUU UUU |  -->  | | | | | | | | | | | |
            #     | | | | | | | | | | | |
            #       <==  4   3   2   1
            #
            for i in reversed(range(1, N - 1, 2)):
                self.pt.right_canonize(start=min(N - 1, i + 2), stop=i + 1)
                self.pt.gate2split(U, where=(i, i + 1),
                                   absorb='left', **self.split_opts)

            # one extra canonicalization not included in last split
            self.pt.right_canonize_site(1)

    def step_order2(self, tau=1):
        """Perform a single, second order step.
        """
        self.sweep('right', tau / 2)
        self.sweep('left', tau)
        self.sweep('right', tau / 2)

    def step_order4(self):
        """Perform a single, fourth order step.
        """
        tau1 = tau2 = 1 / (4 * 4**(1 / 3))
        tau3 = 1 - 2 * tau1 - 2 * tau2
        self.step_order2(tau1)
        self.step_order2(tau2)
        self.step_order2(tau3)
        self.step_order2(tau2)
        self.step_order2(tau1)

    def step(self, order=2):
        """Perform a single step of time ``self.dt``.
        """
        {2: self.step_order2, 4: self.step_order4}[order]()
        self.t += self.dt

    def update_to(self, t, order=2):
        """Update the state to time ``t``.
        """
        if self.progbar:
            with qu.utils.continuous_progbar(self.t, t) as pbar:
                while self.t < t - 1e-13:
                    pbar.cupdate(self.t)
                    self.step(order=order)
                pbar.cupdate(t)

        else:
            while self.t < t - 1e-14:
                self.step(order=order)
