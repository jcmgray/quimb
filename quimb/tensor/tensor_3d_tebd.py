from .tensor_3d import gen_3d_bonds
from .tensor_arbgeom_tebd import LocalHamGen


class LocalHam3D(LocalHamGen):

    def __init__(self, Lx, Ly, Lz, H2, H1=None):
        self.Lx = int(Lx)
        self.Ly = int(Ly)
        self.Lz = int(Lz)

        # parse two site terms
        if hasattr(H2, 'shape'):
            # use as default nearest neighbour term
            H2 = {None: H2}
        else:
            H2 = dict(H2)

        # possibly fill in default gates
        default_H2 = H2.pop(None, None)
        if default_H2 is not None:
            for coo_a, coo_b in gen_3d_bonds(Lx, Ly, Lz, steppers=[
                lambda i, j, k: (i, j, k + 1),
                lambda i, j, k: (i, j + 1, k),
                lambda i, j, k: (i + 1, j, k),
            ]):
                if (coo_a, coo_b) not in H2 and (coo_b, coo_a) not in H2:
                    H2[coo_a, coo_b] = default_H2

        super().__init__(H2=H2, H1=H1)

    @property
    def nsites(self):
        """The number of sites in the system.
        """
        return self.Lx * self.Ly * self.Lz

    def __repr__(self):
        s = "<LocalHam3D(Lx={}, Ly={}, Lx={}, num_terms={})>"
        return s.format(self.Lx, self.Ly, self.Lz, len(self.terms))
