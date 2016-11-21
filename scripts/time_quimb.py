import timeit


# ----------------------------- dense dot ----------------------------------- #
setup = """
import quimb
a = quimb.rand_herm(2**4)
b = quimb.rand_herm(2**4)
"""
stmt = """
a @ b
"""
t = timeit.timeit(stmt, setup=setup, number=100000)
print("Small dot".ljust(20) + ":  {:.3} sec".format(t))

setup = """
import quimb
a = quimb.rand_herm(2**10)
b = quimb.rand_herm(2**10)
"""
stmt = """
a @ b
"""
t = timeit.timeit(stmt, setup=setup, number=10)
print("Big dot".ljust(20) + ":  {:.3} sec".format(t))


# ----------------------------- dense eigsys -------------------------------- #
setup = """
import quimb
mat = quimb.rand_herm(2**4) """
stmt = """
quimb.eigsys(mat) """
t = timeit.timeit(stmt, setup=setup, number=10000)
print("Small eigsys".ljust(20) + ":  {:.3} sec".format(t))

setup = """
import quimb
mat = quimb.rand_herm(2**10) """
stmt = """
quimb.eigsys(mat) """
t = timeit.timeit(stmt, setup=setup, number=10)
print("Big eigsys".ljust(20) + ":  {:.3} sec".format(t))


# ----------------------------- sparse eigsys ------------------------------- #
setup = """
import quimb
mat = quimb.rand_herm(2**14, sparse=True) """
stmt = """
quimb.seigsys(mat, backend='scipy') """
t = timeit.timeit(stmt, setup=setup, number=10)
print("Scipy seigsys".ljust(20) + ":  {:.3} sec".format(t))

setup = """
import quimb
mat = quimb.rand_herm(2**14, sparse=True) """
stmt = """
quimb.seigsys(mat, backend='slepc') """
t = timeit.timeit(stmt, setup=setup, number=10)
print("Slepc seigsys".ljust(20) + ":  {:.3} sec".format(t))

setup = """
import quimb
import qdmbl
mat = qdmbl.ham_qd(10, 1, sparse=True) """
stmt = """
quimb.seigsys(mat, sigma=0.01, backend='scipy') """
t = timeit.timeit(stmt, setup=setup, number=10)
print("Scipy seigsys int".ljust(20) + ":  {:.3} sec".format(t))

setup = """
import quimb
import qdmbl
mat = qdmbl.ham_qd(10, 1, sparse=True) """
stmt = """
quimb.seigsys(mat, sigma=1, backend='slepc') """
t = timeit.timeit(stmt, setup=setup, number=10)
print("Slepc seigsys int".ljust(20) + ":  {:.3} sec".format(t))
