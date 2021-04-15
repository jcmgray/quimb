from quimb.tensor import block_interface as bitf
import numpy as np

def apply(T, func):
    use_cpp = bitf.dispatch_settings("use_cpp")
    if use_cpp:
        new_T = T.copy()
        new_T.data = func(new_T.data)
    else:
        new_T = T.copy()
        for iblk in new_T:
            iblk[:] = func(iblk[:])
    return new_T

def sqrt(T):
    _sqrt = lambda x : x**.5
    return apply(T, _sqrt)

def inv_with_smudge(T, cutoff=1e-10, gauge_smudge=1e-6):
    def _inv_with_smudge(arr):
        new_arr = np.zeros_like(arr)
        ind = abs(arr) > cutoff
        new_arr[ind] = (arr[ind] + gauge_smudge) ** -1
        return new_arr
    return apply(T, _inv_with_smudge)

def add_with_smudge(T, cutoff=1e-10, gauge_smudge=1e-6):
    def _add_with_smudge(arr):
        ind = abs(arr) > cutoff
        arr[ind] += gauge_smudge
        return arr
    return apply(T, _add_with_smudge)

'''
bitf.set(symmetry="z22", use_cpp=True, fermion=True)
H = bitf.Hubbard(1,4,0.1)
H.data = abs(H.data)
H.data[H.data==0] = 2.0
Hsqrt = sqrt(H)
print("sqrt")
print((Hsqrt.data**2-H.data).sum())

Hi = inv_with_smudge(H)
print("inv")
print(Hi.data*H.data)
Ha = add_with_smudge(H)
print("add")
print(Ha.data-H.data)

bitf.set(symmetry="z2", use_cpp=False, fermion=True)
H = bitf.Hubbard(1,4,0.1).to_flat()
H.data = abs(H.data)
H.data[H.data==0] = 2.0
H = H.to_sparse()
Hsqrt = sqrt(H).to_flat()
print("sqrt")
print((Hsqrt.data**2-H.to_flat().data).sum())
Hi = inv_with_smudge(H).to_flat()
print("inv")
print(Hi.data*H.to_flat().data)
Ha = add_with_smudge(H).to_flat()
print("add")
print(Ha.data-H.to_flat().data)
'''
