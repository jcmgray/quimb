import pytest
import itertools

import quimb as qu
import quimb.tensor as qtn


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
        fig = ham.graph(return_fig=True)
        plt.close(fig)

    @pytest.mark.parametrize('Lx', [4, 5])
    @pytest.mark.parametrize('Ly', [4, 5])
    @pytest.mark.parametrize(
        'order', [None, 'sort', 'random', 'smallest_last'])
    def test_ordering(self, Lx, Ly, order):
        H2 = {None: qu.ham_heis(2)}
        for i in range(Lx - 1):
            for j in range(Ly):
                if j + 1 < Ly:
                    H2[(i, j), (i + 1, j + 1)] = qu.ham_heis(2, j=0.5)
                if j - 1 >= 0:
                    H2[(i, j), (i + 1, j - 1)] = qu.ham_heis(2, j=0.5)

        ham = qtn.LocalHam2D(Lx, Ly, H2=H2)

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
