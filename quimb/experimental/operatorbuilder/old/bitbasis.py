import numpy as np
from numba import njit


@njit
def get_local_size(n, rank, world_size):
    """Given global size n, and a rank in [0, world_size), return the size of
    the portion assigned to this rank.
    """
    cutoff_rank = n % world_size
    return n // world_size + int(rank < cutoff_rank)


@njit
def get_local_range(n, rank, world_size):
    """Given global size n, and a rank in [0, world_size), return the range of
    indices assigned to this rank.
    """
    ri = 0
    for rank_below in range(rank):
        ri += get_local_size(n, rank_below, world_size)
    rf = ri + get_local_size(n, rank, world_size)
    return ri, rf


@njit
def get_nth_bit(val, n):
    """Get the nth bit of val.

    Examples
    --------

        >>> get_nth_bit(0b101, 1)
        0
    """
    return (val >> n) & 1


@njit
def flip_nth_bit(val, n):
    """Flip the nth bit of val.

    Examples
    --------

        >>> bin(flip_nth_bit(0b101, 1))
        0b111
    """
    return val ^ (1 << n)


@njit
def _recursively_fill_flatconfigs(flatconfigs, n, k, c, r):
    c0 = c * (n - k) // n
    # set the entries of the left binary subtree
    flatconfigs[r : r + c0, -n] = 0
    # set the entries of the right binary subtree
    flatconfigs[r + c0 : r + c, -n] = 1
    if n > 1:
        # process each subtree recursively
        _recursively_fill_flatconfigs(flatconfigs, n - 1, k, c0, r)
        _recursively_fill_flatconfigs(
            flatconfigs, n - 1, k - 1, c - c0, r + c0
        )


@njit
def get_all_equal_weight_flatconfigs(n, k):
    """Get every flat configuration of length n with k bits set."""
    c = comb(n, k)
    flatconfigs = np.empty((c, n), dtype=np.uint8)
    _recursively_fill_flatconfigs(flatconfigs, n, k, c, 0)
    return flatconfigs


@njit
def flatconfig_to_bit(flatconfig):
    """Given a flat configuration, return the corresponding bitstring."""
    b = 0
    for i, x in enumerate(flatconfig):
        if x:
            b |= 1 << i
    return b


@njit
def comb(n, k):
    """Compute the binomial coefficient n choose k."""
    r = 1
    for dk, dn in zip(range(1, k + 1), range(n, n - k, -1)):
        r *= dn
        r //= dk
    return r


@njit
def get_all_equal_weight_bits(n, k, dtype=np.int64):
    """Get an array of all 'bits' (integers), with n bits, and k of them set."""
    if k == 0:
        return np.array([0], dtype=dtype)

    m = comb(n, k)
    b = np.empty(m, dtype=dtype)
    i = 0
    val = (1 << k) - 1
    while val < (1 << (n)):
        b[i] = val
        c = val & -val
        r = val + c
        val = (((r ^ val) >> 2) // c) | r
        i += 1
    return b


@njit
def bit_to_rank(b, n, k):
    """Given a bitstring b, return the rank of the bitstring in the
    basis of all bitstrings of length n with k bits set. Adapted from
    https://dlbeer.co.nz/articles/kwbs.html.
    """
    c = comb(n, k)
    r = 0
    while n:
        c0 = c * (n - k) // n
        if (b >> n - 1) & 1:
            r += c0
            k -= 1
            c -= c0
        else:
            c = c0
        n -= 1
    return r


@njit
def rank_to_bit(r, n, k):
    """Given a rank r, return the bitstring of length n with k bits set
    that has rank r in the basis of all bitstrings of length n with k
    bits set. Adapted from https://dlbeer.co.nz/articles/kwbs.html.
    """
    b = 0
    c = comb(n, k)
    while n:
        c0 = c * (n - k) // n
        if r >= c0:
            b |= 1 << (n - 1)
            r -= c0
            k -= 1
            c -= c0
        else:
            c = c0
        n -= 1
    return b


@njit
def product_of_bits(b1, b2, n2, dtype=np.int64):
    """Get the outer product of two bit arrays.

    Parameters
    ----------
    b1, b2 : array
        The bit arrays to take the outer product of.
    n2 : int
        The number of bits in ``b2``
    """
    b = np.empty(len(b1) * len(b2), dtype=dtype)
    i = 0
    for x in b1:
        for y in b2:
            b[i] = (x << n2) | y
            i += 1
    return b


def get_number_bitbasis(*nk_pairs, dtype=np.int64):
    """Create a bit basis with number conservation.

    Parameters
    ----------
    nk_pairs : sequence of (int, int)
        Each element is a pair (n, k) where n is the number of bits, and k is
        the number of bits that are set. The basis will be the product of all
        supplied pairs.

    Returns
    -------
    basis : ndarray
        An array of integers, each representing a bit string. The size will be
        ``prod(comb(n, k) for n, k in nk_pairs)``.

    Examples
    --------

    A single number conserving basis:

        >>> for i, b in enumerate(get_number_bitbasis((4, 2))):
        >>>    print(f"{i}: {b:0>4b}")
        0: 0011
        1: 0101
        2: 0110
        3: 1001
        4: 1010
        5: 1100

    A product of two number conserving bases, e.g. n_up and n_down:

        >>> for b in get_number_bitbasis((3, 2), (3, 1)):
        >>>     print(f"{b:0>6b}")
        011001
        011010
        011100
        101001
        101010
        101100
        110001
        110010
        110100

    """
    configs = None
    for n, k in nk_pairs:
        next_configs = get_all_equal_weight_bits(n, k, dtype=dtype)
        if configs is None:
            configs = next_configs
        else:
            configs = product_of_bits(configs, next_configs, n, dtype=dtype)
    return configs


@njit
def build_bitmap(configs):
    """Build of map of bits to linear indices, suitable for use with numba."""
    return {b: i for i, b in enumerate(configs)}


@njit(nogil=True)
def coupled_bits_numba(bi, coupling_map):
    buf_ptr = 0
    bjs = np.empty(len(coupling_map), dtype=np.int64)
    cijs = np.empty(len(coupling_map), dtype=np.float64)
    bitmap = {}

    for coupling_t in coupling_map.values():
        cij = 1.0
        bj = bi
        for reg, coupling_t_reg in coupling_t.items():
            xi = get_nth_bit(bi, reg)
            if xi not in coupling_t_reg:
                # zero coupling - whole branch dead
                break
            # update coeff and config
            xj, cij = coupling_t_reg[xi]
            cij *= cij
            if xi != xj:
                bj = flip_nth_bit(bj, reg)
        else:
            # no break - all terms survived
            if bj in bitmap:
                # already seed this config - just add coeff
                loc = bitmap[bj]
                cijs[loc] += cij
            else:
                # TODO: check performance of numba exception catching
                bjs[buf_ptr] = bj
                cijs[buf_ptr] = cij
                bitmap[bj] = buf_ptr
                buf_ptr += 1

    return bjs[:buf_ptr], cijs[:buf_ptr]


@njit(nogil=True)
def _build_coo_numba_core(bits, coupling_map, bitmap=None, dtype=np.float64):
    # the bit map is needed if we only have a partial set of `bits`, which
    # might couple to other bits that are not in `bits` -> we need to look up
    # the linear register of the coupled bit
    if bitmap is None:
        bitmap = {bi: ci for ci, bi in enumerate(bits)}

    buf_size = len(bits)
    data = np.empty(buf_size, dtype=dtype)
    cis = np.empty(buf_size, dtype=np.int64)
    cjs = np.empty(buf_size, dtype=np.int64)
    buf_ptr = 0

    for bi in bits:
        ci = bitmap[bi]
        for coupling_t in coupling_map.values():
            hij = 1.0
            bj = bi
            for reg, coupling_t_reg in coupling_t.items():
                xi = get_nth_bit(bi, reg)
                if xi not in coupling_t_reg:
                    # zero coupling - whole branch dead
                    break
                # update coeff and config
                xj, cij = coupling_t_reg[xi]
                hij *= cij
                if xi != xj:
                    bj = flip_nth_bit(bj, reg)
            else:
                # didn't break out of loop
                if buf_ptr >= buf_size:
                    # need to double our storage
                    data = np.concatenate((data, np.empty_like(data)))
                    cis = np.concatenate((cis, np.empty_like(cis)))
                    cjs = np.concatenate((cjs, np.empty_like(cjs)))
                    buf_size *= 2
                data[buf_ptr] = hij
                cis[buf_ptr] = ci
                cjs[buf_ptr] = bitmap[bj]
                buf_ptr += 1

    return data[:buf_ptr], cis[:buf_ptr], cjs[:buf_ptr]


def build_coo_numba(bits, coupling_map, dtype=None, parallel=False):
    """Build an operator in COO form, using the basis ``bits`` and the
     ``coupling_map``, optionally multithreaded.

    Parameters
    ----------
    bits : array
        An array of integers, each representing a bit string.
    coupling_map : Dict[int, Dict[int, Dict[int, Tuple[int, float]]]]
        A nested numba dictionary of couplings. The outermost key is the term
        index, the next key is the register, and the innermost key is the bit
        index. The value is a tuple of (coupled bit index, coupling
        coefficient).
    parallel : bool or int, optional
        Whether to parallelize the computation. If an integer is given, it
        specifies the number of threads to use.

    Returns
    -------
    data : array
        The non-zero elements of the operator.
    cis : array
        The row indices of the non-zero elements.
    cjs : array
        The column indices of the non-zero elements.
    """
    if not parallel:
        return _build_coo_numba_core(bits, coupling_map, dtype=dtype)

    from quimb import get_thread_pool

    if parallel is True:
        n_thread_workers = None
    elif isinstance(parallel, int):
        n_thread_workers = parallel
    else:
        raise ValueError(f"Unknown parallel option {parallel}.")

    pool = get_thread_pool(n_thread_workers)
    n_thread_workers = pool._max_workers

    # need a global mapping of bits to linear indices
    kws = {
        "coupling_map": coupling_map,
        "bitmap": build_bitmap(bits),
        "dtype": dtype,
    }

    # launch the threads! note we distribtue in cyclic fashion as the sparsity
    # can be concentrated in certain ranges and we want each thread to have
    # roughly the same amount of work to do
    fs = [
        pool.submit(
            _build_coo_numba_core, bits=bits[i::n_thread_workers], **kws
        )
        for i in range(n_thread_workers)
    ]

    # gather and concatenate the results (probably some memory overhead here)
    data = []
    cis = []
    cjs = []
    for f in fs:
        d, ci, cj = f.result()
        data.append(d)
        cis.append(ci)
        cjs.append(cj)

    data = np.concatenate(data)
    cis = np.concatenate(cis)
    cjs = np.concatenate(cjs)

    return data, cis, cjs


def config_to_bit(self, config):
    """Encode a 'configuration' as a bit.

    Parameters
    ----------
    config : dict[hashable, int]
        A dictionary mapping sites to their occupation number / spin.

    Returns
    -------
    bit : int
        The bit corresponding to this configuration.
    """
    bit = 0
    for site, val in config.items():
        if val:
            bit |= 1 << self.site_to_reg(site)
    return bit


def bit_to_config(self, bit):
    """Decode a bit to a configuration.

    Parameters
    ----------
    bit : int
        The bit to decode.

    Returns
    -------
    config : dict[hashable, int]
        A dictionary mapping sites to their occupation number / spin.
    """
    config = {}
    for site, reg in self._mapping.items():
        config[site] = (bit >> reg) & 1
    return config
