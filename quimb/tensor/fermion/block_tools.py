from .block_interface import dispatch_settings
import numpy as np

def apply(T, func):
    use_cpp = dispatch_settings("use_cpp")
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
        new_arr = np.zeros(arr.shape, dtype=arr.dtype)
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

def get_smudge_balance(T1, T2, ix, smudge):
    flat = dispatch_settings("use_cpp")
    if flat:
        t1, t2 = T1.data.to_sparse(), T2.data.to_sparse()
    else:
        t1, t2 = T1.data, T2.data
    sign1 = t1.pattern[T1.inds.index(ix)]
    sign2 = t2.pattern[T2.inds.index(ix)]
    s1_pattern = {"+":"-+", "-":"+-"}[sign1]
    s2_pattern = {"-":"-+", "+":"+-"}[sign2]

    inv = (sign1 == sign2)
    block_cls = t1.blocks[0].__class__
    block_dict = {}
    for iblk1 in t1:
        q0 = iblk1.q_labels[0]
        block_dict[q0] = np.diag(np.asarray(iblk1)) + smudge
    for iblk2 in t2:
        q0 = -iblk2.q_labels[0] if inv else iblk2.q_labels[0]
        if q0 not in block_dict: continue
        block_dict[q0] = block_dict[q0] / (np.diag(np.asarray(iblk2)) + smudge)

    s1 = [block_cls(reduced=np.diag(s**-0.25), q_labels=(qlab,)*2) for qlab, s in block_dict.items()]
    s2 = [block_cls(reduced=np.diag(s** 0.25), q_labels=(qlab,)*2) for qlab, s in block_dict.items()]
    s1 = t1.__class__(blocks=s1, pattern=s1_pattern)
    s2 = t2.__class__(blocks=s2, pattern=s2_pattern)
    if flat:
        s1 = s1.to_flat()
        s2 = s2.to_flat()
    return s1, s2
