import itertools
import importlib

import pytest

import quimb as qu
import quimb.tensor as qtn

found_torch = importlib.util.find_spec('torch') is not None

pytorch_case = pytest.param(
    'torch', marks=pytest.mark.skipif(
        not found_torch, reason='pytorch not installed'))

class TestLocalHam2DConstruct:

    @pytest.mark.parametrize(
        'H2_type',
        ['default', 'manual'])
    @pytest.mark.parametrize(
        'H1_type',
        [None, 'default', 'manual'])
    @pytest.mark.parametrize('Lx', [3, 4])
    @pytest.mark.parametrize('Ly', [3, 4])
    def test_construct(self, Lx, Ly, H2_type, H1_type):
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.use('Template')

        if H2_type == 'default':
            H2 = qu.rand_herm(4)
        elif H2_type == 'manual':
            H2 = dict()
            for i, j in itertools.product(range(Lx), range(Ly)):
                if i + 1 < Lx:
                    H2[(i, j), (i + 1, j)] = qu.rand_herm(4)
                if j + 1 < Ly:
                    H2[(i, j), (i, j + 1)] = qu.rand_herm(4)

        if H1_type is None:
            H1 = None
        elif H1_type == 'default':
            H1 = qu.rand_herm(2)
        elif H1_type == 'manual':
            H1 = dict()
            for i, j in itertools.product(range(Lx), range(Ly)):
                H1[i, j] = qu.rand_herm(2)

        ham = qtn.LocalHam2D(Lx, Ly, H2, H1)
        assert len(ham.terms) == 2 * Lx * Ly - Lx - Ly

        # check that terms are being cached if possible
        if (H2_type == 'default') and (H1_type is None):
            assert len({id(x) for x in ham.terms.values()}) == 1

        print(ham)
        fig = ham.draw(return_fig=True)
        plt.close(fig)

    @pytest.mark.parametrize('Lx', [4, 5])
    @pytest.mark.parametrize('Ly', [4, 5])
    @pytest.mark.parametrize(
        'order', [None, 'sort', 'random', 'smallest_last'])
    def test_ordering(self, Lx, Ly, order):
        ham = qtn.ham_2d_j1j2(Lx, Ly)
        assert (
            len(ham.terms) ==
            2 * Lx * Ly - Lx - Ly + 2 * (Lx - 1) * (Ly - 1)
        )
        ordering = ham.get_auto_ordering(order)
        assert len(ordering) == len(ham.terms)
        assert set(ordering) == set(ham.terms)
        assert tuple(ordering) != tuple(ham.terms)

        # make sure first four pairs are in same commuting group at least
        first_four_pairs = tuple(itertools.chain(*ordering[:4]))
        assert len(first_four_pairs) == len(set(first_four_pairs))


class TestSimpleUpdate:

    @pytest.mark.parametrize('backend', ['numpy', pytorch_case])
    def test_heis_small(self, backend):
        Lx = 3
        Ly = 4
        D = 2

        ham = qtn.ham_2d_heis(Lx, Ly)
        psi0 = qtn.PEPS.rand(Lx, Ly, D)

        def to_backend(x):
            import autoray
            return autoray.do('array', x, like=backend)

        psi0.apply_to_arrays(to_backend)
        ham.apply_to_arrays(to_backend)

        su = qtn.SimpleUpdate(
            psi0, ham, progbar=True, keep_best=True, compute_energy_every=10)

        su.evolve(33, tau=0.3)
        su.state = su.best['state']
        su.evolve(33, tau=0.1)
        su.state = su.best['state']
        su.evolve(33, tau=0.03)
        su.state = su.best['state']

        assert su.best['energy'] < -6.25

class TestFullUpdate:

    @pytest.mark.parametrize('backend', ['numpy', pytorch_case])
    def test_heis_small(self, backend):
        Lx = 3
        Ly = 4
        D = 2

        psi0 = qtn.PEPS.rand(Lx, Ly, D)
        ham = qtn.ham_2d_heis(Lx, Ly)

        def to_backend(x):
            import autoray
            return autoray.do('array', x, like=backend)

        psi0.apply_to_arrays(to_backend)
        ham.apply_to_arrays(to_backend)

        su = qtn.FullUpdate(
            psi0, ham, progbar=True, keep_best=True, compute_energy_every=1)

        su.evolve(33, tau=0.3)
        su.state = su.best['state']
        su.evolve(33, tau=0.1)
        su.state = su.best['state']
        su.evolve(33, tau=0.03)
        su.state = su.best['state']

        assert su.best['energy'] < -6.30
