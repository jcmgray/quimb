"""Tools for working with ranked 'flat configs'."""

import numpy as np
from numba import njit

nogil = True


@njit(nogil=nogil, inline="always")
def _check_next_coupled_term(
    a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
):
    """Calculate the next coupled config and coeff, or flag as zero if
    there is no coupled config. Inplace modifies the bj array.

    Parameters
    ----------
    a : int
        The index into the stacked terms.
    b : int
        The index into the stacked operators.
    n : int
        The total size of the flat configuration.
    bi : ndarray[uint8]
        The flat configuration to get the coupled configurations for.
    bj : ndarray[uint8]
        The coupled flat configuration to be modified in place.
    size_term : int
        The number of operators in the term.
    sizes_op : ndarray[int]
        Flat array of number of entries for each operator, indexed by `a`.
    regs : ndarray[int]
        Flat array of registers in each term, indexed by `a`.
    xis : ndarray[uint8]
        Flat array of input bits for each entry, indexed by `b`.
    xjs : ndarray[uint8]
        Flat array of output bits for each entry, indexed by `b`.
    cijs : ndarray[float]
        Flat array of coefficients for each entry, indexed by `b`.

    Returns
    -------
    a : int
        The updated index into the stacked terms.
    b : int
        The updated index into the stacked operators.
    valid : bool
        True if the coupled config is non-zero, False otherwise.
    hij : float
        The coefficient for the coupled config.
    """
    # reset coeff
    hij = 1.0
    valid = True
    # reset coupled config
    for q in range(n):
        bj[q] = bi[q]

    # for each operator in the term
    for da in range(size_term):
        # the number of entries in the operator
        ia = a + da
        size_op = sizes_op[ia]

        if valid:
            # check further coupling
            reg = regs[ia]
            xi = bi[reg]

            # TODO: generalize this to more than dim=2
            if size_op == 1:
                # must match single input
                valid = xi == xis[b]
                if valid:
                    # found a match
                    xj = xjs[b]
                    cij = cijs[b]
            else:
                # 0 and 1 both match
                ib = b + xi
                xj = xjs[ib]
                cij = cijs[ib]

            # valid = False
            # for each entry in the operator
            # for db in range(size_op):
            #     ib = b + db
            #     xin = xis[ib]
            #     if xi == xin:
            #         # found a match
            #         valid = True
            #         xj = xjs[ib]
            #         cij = cijs[ib]
            #         break

            if valid:
                # midstring update of coeff and config
                hij *= cij
                bj[reg] = xj

        # increment operator index
        b += size_op
    # increment the term index
    a += size_term

    return a, b, valid, hij


@njit(nogil=True)
def flatconfig_coupling_numba(flatconfig, coupling_map, dtype=np.float64):
    """Get the coupled flat configurations for a given flat configuration
    and coupling map.

    Like applying the sparse matrix to a basis vector, or retrieving a single
    row.

    Parameters
    ----------
    flatconfig : ndarray[uint8]
        The flat configuration to get the coupled configurations for.
    coupling_map : tuple[ndarray]
        The operator defined as tuple of flat arrays.
    dtype : {np.float64, np.complex128, np.float32, np.float64}, optional
        The dtype to use for the coupled coefficients. Default is np.float64.

    Returns
    -------
    coupled_flatconfigs : ndarray[uint8]
        A list of coupled flat configurations, each with the corresponding
        coefficient.
    coeffs : ndarray[dtype]
        The coefficients for each coupled flat configuration.
    """
    n = flatconfig.size
    buf_ptr = 0
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map
    num_terms = len(sizes_term)

    coupled_flatconfigs = np.empty((num_terms, n), dtype=np.uint8)
    coeffs = np.empty(num_terms, dtype=dtype)

    # indices into stacked terms and operators
    a = b = 0
    for size_term in sizes_term:
        # for each term in the hamiltonian find which, if
        # any, config it couples to, & with what coefficient
        a, b, valid, hij = _check_next_coupled_term(
            a, b, n, flatconfig, bj, size_term, sizes_op, regs, xis, xjs, cijs
        )
        if valid:
            coupled_flatconfigs[buf_ptr, :] = bj
            coeffs[buf_ptr] = hij
            buf_ptr += 1

    return coupled_flatconfigs[:buf_ptr], coeffs[:buf_ptr]


# ----------------------- unconstrained hilbert space ----------------------- #


@njit(nogil=nogil, inline="always")
def flatconfig_to_rank_nosymm(flatconfig):
    """Convert a flat config array to a rank, i.e. its position in the
    lexicographic ordering of all bitstrings of that length.

    Parameters
    ----------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8.

    Returns
    -------
    int
    """
    r = 0
    for xi in flatconfig:
        r = (r << 1) | xi
    return r


@njit(nogil=nogil, inline="always")
def rank_into_flatconfig_nosymm(flatconfig, r, n):
    """Inplace conversion of a rank to a flat config array.

    Parameters
    ----------
    flatconfig : array_like
        A input flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8, it will be overwritten.
    r : int
        The rank to convert.
    n : int
        The total length of the flat config array.
    """
    for i in range(n - 1, -1, -1):
        flatconfig[i] = r & 1
        r >>= 1


@njit(nogil=nogil, inline="always")
def rank_to_flatconfig_nosymm(r, n):
    """Convert a rank to a flat config array.

    Parameters
    ----------
    r : int
        The rank to convert.
    n : int
        The total length of the flat config array.

    Returns
    -------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array will be of dtype uint8.
    """
    flatconfig = np.empty(n, dtype=np.uint8)
    rank_into_flatconfig_nosymm(flatconfig, r, n)
    return flatconfig


@njit(nogil=nogil)
def build_coo_numba_core_nosymm(
    n,
    coupling_map,
    dtype=np.float64,
    world_size=1,
    world_rank=0,
):
    """Build sparse coo data in a unconstrained hilbert space."""
    D = 2**n

    buf_size = D
    data = np.empty(buf_size, dtype=dtype)
    rows = np.empty(buf_size, dtype=np.int64)
    cols = np.empty(buf_size, dtype=np.int64)
    buf_ptr = 0

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_nosymm(bi, ci, n)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                if buf_ptr >= buf_size:
                    # need to double our storage
                    data = np.concatenate((data, np.empty_like(data)))
                    rows = np.concatenate((rows, np.empty_like(rows)))
                    cols = np.concatenate((cols, np.empty_like(cols)))
                    buf_size *= 2

                # convert coupled config back to rank
                cj = flatconfig_to_rank_nosymm(bj)

                data[buf_ptr] = hij
                cols[buf_ptr] = ci
                rows[buf_ptr] = cj
                buf_ptr += 1

    return data[:buf_ptr], rows[:buf_ptr], cols[:buf_ptr]


@njit(nogil=nogil)
def matvec_nosymm(
    x,
    out,
    n,
    coupling_map,
    world_size=1,
    world_rank=0,
):
    D = 2**n

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_nosymm(bi, ci, n)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                cj = flatconfig_to_rank_nosymm(bj)
                out[cj] += hij * x[ci]


# --------------------- parity conserved hilbert space ---------------------- #


@njit(nogil=nogil, inline="always")
def flatconfig_to_rank_z2(flatconfig):
    """Convert a flat config array to a z2 rank, i.e. its position in the
    lexicographic ordering of all bitstrings of even or odd parity.

    Parameters
    ----------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8.

    Returns
    -------
    int
    """
    r = 0
    # we treat like the nosymm case and simply ignore the last bit
    for i in range(flatconfig.size - 1):
        r = (r << 1) | flatconfig[i]
    return r


@njit(nogil=nogil, inline="always")
def rank_into_flatconfig_z2(flatconfig, r, n, p):
    """Inplace conversion of a z2 rank to a flat config array.

    Parameters
    ----------
    flatconfig : array_like
        A input flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8, it will be overwritten.
    r : int
        The rank to convert.
    n : int
        The total length of the flat config array.
    """
    prem = 0
    m = 1 << (n - 2)
    for i in range(n - 1):
        xi = r & m != 0
        flatconfig[i] = xi
        m >>= 1
        prem ^= xi
    flatconfig[n - 1] = prem ^ p


@njit(nogil=nogil, inline="always")
def rank_to_flatconfig_z2(r, n, p):
    """Convert a z2 rank to a flat config array.

    Parameters
    ----------
    r : int
        The rank to convert.
    n : int
        The total length of the flat config array.
    p : int
        The parity of the flat config array.

    Returns
    -------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array will be of dtype uint8.
    """
    flatconfig = np.empty(n, dtype=np.uint8)
    rank_into_flatconfig_z2(flatconfig, r, n, p)
    return flatconfig


@njit(nogil=nogil)
def build_coo_numba_core_z2(
    n,
    p,
    coupling_map,
    dtype=np.float64,
    world_size=1,
    world_rank=0,
):
    """Build sparse coo data in a parity conserved hilbert space."""
    D = 2 ** (n - 1)

    buf_size = D
    data = np.empty(buf_size, dtype=dtype)
    rows = np.empty(buf_size, dtype=np.int64)
    cols = np.empty(buf_size, dtype=np.int64)
    buf_ptr = 0

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_z2(bi, ci, n, p)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                if buf_ptr >= buf_size:
                    # need to double our storage
                    data = np.concatenate((data, np.empty_like(data)))
                    rows = np.concatenate((rows, np.empty_like(rows)))
                    cols = np.concatenate((cols, np.empty_like(cols)))
                    buf_size *= 2

                # convert coupled config back to rank
                cj = flatconfig_to_rank_z2(bj)

                data[buf_ptr] = hij
                cols[buf_ptr] = ci
                rows[buf_ptr] = cj
                buf_ptr += 1

    return data[:buf_ptr], rows[:buf_ptr], cols[:buf_ptr]


@njit(nogil=nogil)
def matvec_z2(
    x,
    out,
    n,
    p,
    coupling_map,
    world_size=1,
    world_rank=0,
):
    D = 2 ** (n - 1)

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_z2(bi, ci, n, p)

        for coupling_t in coupling_map:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient

            # reset coeff
            hij = 1.0
            # reset coupled config
            for q in range(n):
                bj[q] = bi[q]

            # for each operator in the term
            for reg, coupling_t_reg in coupling_t:
                xi = bi[reg]

                for xin, xj, cij in coupling_t_reg:
                    if xi == xin:
                        # found a match
                        break
                else:
                    # not coupled to anything - break from whole term loop
                    break

                # update coeff and config
                hij *= cij
                bj[reg] = xj
            else:
                # didn't break out of loop -> valid coupled config
                cj = flatconfig_to_rank_z2(bj)
                out[cj] += hij * x[ci]


# -------------------- particle conserved hilbert space --------------------- #


@njit(nogil=nogil)
def build_pascal_table(nmax):
    """Build the Pascal triangle table for the number of ways to choose k
    elements from n elements, i.e. the binomial coefficients.
    The table is of shape (nmax+1, nmax+1) and the entry at (n, k) is
    the number of ways to choose k elements from n elements.

    Parameters
    ----------
    nmax : int
        The maximum number of particles.

    Returns
    -------
    pt : array_like
        The Pascal triangle table of shape (nmax+1, nmax+1).
    """
    d = nmax + 1
    pt = np.zeros((d, d), dtype=np.int64)
    for n in range(d):
        pt[n, 0] = 1
        for k in range(1, n + 1):
            pt[n, k] = pt[n - 1, k - 1] + pt[n - 1, k]
    return pt


@njit(nogil=nogil, inline="always")
def flatconfig_to_rank_u1_pascal(flatconfig, n, k, pt):
    """Given a flat config array, return the u1 rank of the config in the
    lexicographic ordering of all bitstrings of that length with hamming weight
    k.

    Parameters
    ----------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8.
    n : int
        The total length of the flat config array.
    k : int
        The number of particles in the flat config array.
    pt : array_like
        The Pascal triangle table of shape containing at least (n, k).

    Returns
    -------
    r : int
        The rank of the config in the lexicographic ordering.
    """
    r = 0
    krem = k
    j = n
    for i, xi in enumerate(flatconfig):
        j -= 1
        r += xi * pt[j, krem]
        krem -= xi
    return r


@njit(nogil=nogil, inline="always")
def rank_into_flatconfig_u1_pascal(flatconfig, r, n, k, pt):
    """Inplace conversion of a rank to a flat config array with hamming weight
    k.

    Parameters
    ----------
    flatconfig : array_like
        A input flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8, it will be overwritten.
    r : int
        The rank to convert.
    n : int
        The total length of the flat config array.
    k : int
        The number of particles in the flat config array.
    pt : array_like
        The Pascal triangle table of shape containing at least (n, k).
    """
    r = np.int64(r)
    krem = np.int64(k)
    j = n
    for i in range(n):
        j -= 1
        rank_if_one = pt[j, krem]
        if r >= rank_if_one:
            flatconfig[i] = 1
            r -= rank_if_one
            krem -= 1
        else:
            flatconfig[i] = 0


@njit(nogil=nogil, inline="always")
def rank_to_flatconfig_u1_pascal(r, n, k, pt):
    """Convert a rank to a flat config array with hamming weight k.

    Parameters
    ----------
    r : int
        The rank to convert.
    n : int
        The total length of the flat config array.
    k : int
        The number of particles in the flat config array.
    pt : array_like
        The Pascal triangle table of shape containing at least (n, k).

    Returns
    -------
    array_like
        A flat config array of shape (n,), corresponding to a bitstring.
    """
    flatconfig = np.empty(n, dtype=np.uint8)
    rank_into_flatconfig_u1_pascal(flatconfig, r, n, k, pt)
    return flatconfig


@njit(nogil=nogil)
def build_coo_numba_core_u1(
    n,
    k,
    coupling_map,
    dtype=np.float64,
    world_size=1,
    world_rank=0,
):
    """Build sparse coo data in a number conserved hilbert space."""
    pt = build_pascal_table(n)
    D = pt[n, k]

    buf_size = D
    data = np.empty(buf_size, dtype=dtype)
    rows = np.empty(buf_size, dtype=np.int64)
    cols = np.empty(buf_size, dtype=np.int64)
    buf_ptr = 0

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_u1_pascal(bi, ci, n, k, pt)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                if buf_ptr >= buf_size:
                    # need to double our storage
                    data = np.concatenate((data, np.empty_like(data)))
                    rows = np.concatenate((rows, np.empty_like(rows)))
                    cols = np.concatenate((cols, np.empty_like(cols)))
                    buf_size *= 2

                # convert coupled config back to rank
                cj = flatconfig_to_rank_u1_pascal(bj, n, k, pt)

                data[buf_ptr] = hij
                cols[buf_ptr] = ci
                rows[buf_ptr] = cj
                buf_ptr += 1

    return data[:buf_ptr], rows[:buf_ptr], cols[:buf_ptr]


@njit(nogil=nogil)
def matvec_u1(
    x,
    out,
    n,
    k,
    coupling_map,
    world_size=1,
    world_rank=0,
):
    pt = build_pascal_table(n)
    D = pt[n, k]

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_u1_pascal(bi, ci, n, k, pt)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                cj = flatconfig_to_rank_u1_pascal(bj, n, k, pt)
                out[cj] += hij * x[ci]


# --------------------- doubly conserved hilbert space ---------------------- #


@njit(nogil=nogil, inline="always")
def flatconfig_to_rank_u1u1_pascal(flatconfig, na, ka, nb, kb, pt):
    """Convert a flat config array to a doubly number conserved rank, i.e.
    its position in the ordering of all bitstrings of lenght `na + nb` with
    hamming weight `ka` and `kb` on each section respectively.

    Parameters
    ----------
    flatconfig : array_like
        A flat config array of shape (na + nb,), corresponding to a bitstring.
        The array should be of dtype uint8.
    na : int
        The total length of the first section of the flat config array.
    ka : int
        The hamming weight of the first section.
    nb : int
        The total length of the second section of the flat config array.
    kb : int
        The hamming weight of the second section.
    pt : array_like
        The Pascal triangle table of shape containing at least max(na, nb).

    Returns
    -------
    r : int
        The rank of the config in the lexicographic ordering.
    """
    Db = pt[nb, kb]
    return flatconfig_to_rank_u1_pascal(
        flatconfig[:na], na, ka, pt
    ) * Db + flatconfig_to_rank_u1_pascal(flatconfig[na:], nb, kb, pt)


@njit(nogil=nogil, inline="always")
def rank_into_flatconfig_u1u1_pascal(flatconfig, r, na, ka, nb, kb, pt):
    """Inplace conversion of a doubly number conserved rank to a flat config
    array.

    Parameters
    ----------
    flatconfig : array_like
        A input flat config array of shape (na + nb,), corresponding to a
        bitstring. The array should be of dtype uint8, it will be overwritten.
    r : int
        The rank to convert.
    na : int
        The total length of the first section of the flat config array.
    ka : int
        The hamming weight of the first section.
    nb : int
        The total length of the second section of the flat config array.
    kb : int
        The hamming weight of the second section.
    pt : array_like
        The Pascal triangle table of shape containing at least max(na, nb).
    """
    Db = pt[nb, kb]
    r1 = r // Db
    r2 = r % Db
    rank_into_flatconfig_u1_pascal(flatconfig[:na], r1, na, ka, pt)
    rank_into_flatconfig_u1_pascal(flatconfig[na:], r2, nb, kb, pt)


@njit(nogil=nogil, inline="always")
def rank_to_flatconfig_u1u1_pascal(r, na, ka, nb, kb, pt):
    """Convert a doubly number conserved rank to a flat config array.

    Parameters
    ----------
    r : int
        The rank to convert.
    na : int
        The total length of the first section of the flat config array.
    ka : int
        The hamming weight of the first section.
    nb : int
        The total length of the second section of the flat config array.
    kb : int
        The hamming weight of the second section.
    pt : array_like
        The Pascal triangle table of shape containing at least max(na, nb).

    Returns
    -------
    flatconfig : array_like
        A flat config array of shape (na + nb,), corresponding to a bitstring.
        The array will be of dtype uint8.
    """
    flatconfig = np.empty(na + nb, dtype=np.uint8)
    rank_into_flatconfig_u1u1_pascal(flatconfig, r, na, ka, nb, kb, pt)
    return flatconfig


@njit(nogil=nogil)
def build_coo_numba_core_u1u1(
    na,
    ka,
    nb,
    kb,
    coupling_map,
    dtype=np.float64,
    world_size=1,
    world_rank=0,
):
    pt = build_pascal_table(max(na, nb))
    D = pt[na, ka] * pt[nb, kb]
    n = na + nb

    buf_size = D
    data = np.empty(buf_size, dtype=dtype)
    rows = np.empty(buf_size, dtype=np.int64)
    cols = np.empty(buf_size, dtype=np.int64)
    buf_ptr = 0

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting configs
        rank_into_flatconfig_u1u1_pascal(bi, ci, na, ka, nb, kb, pt)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                if buf_ptr >= buf_size:
                    # need to double our storage
                    data = np.concatenate((data, np.empty_like(data)))
                    rows = np.concatenate((rows, np.empty_like(rows)))
                    cols = np.concatenate((cols, np.empty_like(cols)))
                    buf_size *= 2

                # convert coupled config back to rank
                cj = flatconfig_to_rank_u1u1_pascal(bj, na, ka, nb, kb, pt)

                data[buf_ptr] = hij
                cols[buf_ptr] = ci
                rows[buf_ptr] = cj
                buf_ptr += 1

    return data[:buf_ptr], rows[:buf_ptr], cols[:buf_ptr]


@njit(nogil=nogil)
def matvec_u1u1(
    x,
    out,
    na,
    ka,
    nb,
    kb,
    coupling_map,
    world_size=1,
    world_rank=0,
):
    pt = build_pascal_table(max(na, nb))
    D = pt[na, ka] * pt[nb, kb]
    n = na + nb

    bi = np.empty(n, dtype=np.uint8)
    bj = np.empty(n, dtype=np.uint8)

    sizes_term, regs, sizes_op, xis, xjs, cijs = coupling_map

    for ci in range(world_rank, D, world_size):
        # reset the starting config
        rank_into_flatconfig_u1u1_pascal(bi, ci, na, ka, nb, kb, pt)
        # indices into stacked terms and operators
        a = b = 0
        for size_term in sizes_term:
            # for each term in the hamiltonian find which, if
            # any, config it couples to, & with what coefficient
            a, b, valid, hij = _check_next_coupled_term(
                a, b, n, bi, bj, size_term, sizes_op, regs, xis, xjs, cijs
            )
            if valid:
                cj = flatconfig_to_rank_u1u1_pascal(bj, na, ka, nb, kb, pt)
                out[cj] += hij * x[ci]


# ------------------------------- public api -------------------------------- #


@njit(nogil=nogil)
def rank_to_flatconfig(r, sector, symmetry=None, pt=None):
    """Convert a rank to a flat config array.

    Parameters
    ----------
    r : int
        The rank to convert.
    sector : tuple[int]
        Specifies the sector to convert.

        - (n,) for unconstrained hilbert space
        - (n, parity) for Z2 symmetry
        - (n, k) for U1 symmetry
        - (na, ka, nb, kb) for U1U1 symmetry

    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        Specifies the symmetry to use.
    pt : array_like, optional
        The Pascal triangle table of shape containing at least max(na, nb).
        If not provided, it will be built internally. This is only used for
        U1 and U1U1 hilbert spaces.

    Returns
    -------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array will be of dtype uint8.
    """
    if symmetry is None:
        # unconstrained hilbert space
        (n,) = sector
        return rank_to_flatconfig_nosymm(r, n)
    elif symmetry == "Z2":
        n, p = sector
        if pt is None:
            pt = build_pascal_table(n)
        return rank_to_flatconfig_z2(r, n, p)
    elif symmetry == "U1":
        n, k = sector
        if pt is None:
            pt = build_pascal_table(n)
        return rank_to_flatconfig_u1_pascal(r, n, k, pt)
    elif symmetry == "U1U1":
        na, ka, nb, kb = sector
        if pt is None:
            pt = build_pascal_table(max(na, nb))
        return rank_to_flatconfig_u1u1_pascal(r, na, ka, nb, kb, pt)
    else:
        raise ValueError(
            r"Symmetry must be None, 'Z2', 'U1' or 'U1U1'. Got "
            f"{symmetry} instead."
        )


@njit(nogil=nogil)
def flatconfig_to_rank(flatconfig, sector, symmetry=None, pt=None):
    """Convert a flat config array to a rank, i.e. its position in the
    lexicographic ordering of all bitstrings of that length.

    Parameters
    ----------
    flatconfig : array_like
        A flat config array of shape (n,), corresponding to a bitstring.
        The array should be of dtype uint8.
    sector : tuple[int]
        Specifies the sector to convert.

        - (n,) for unconstrained hilbert space
        - (n, parity) for Z2 symmetry
        - (n, k) for U1 symmetry
        - (na, ka, nb, kb) for U1U1 symmetry

    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        Specifies the symmetry to use.
    pt : array_like, optional
        The Pascal triangle table of shape containing at least max(na, nb).
        If not provided, it will be built internally. This is only used for
        U1 and U1U1 hilbert spaces.

    Returns
    -------
    int
    """
    if symmetry is None:
        # unconstrained hilbert space
        (n,) = sector
        return flatconfig_to_rank_nosymm(flatconfig)
    elif symmetry == "Z2":
        n, p = sector
        return flatconfig_to_rank_z2(flatconfig, n, p)
    elif symmetry == "U1":
        n, k = sector
        if pt is None:
            pt = build_pascal_table(n)
        return flatconfig_to_rank_u1_pascal(flatconfig, n, k, pt)
    elif symmetry == "U1U1":
        na, ka, nb, kb = sector
        if pt is None:
            pt = build_pascal_table(max(na, nb))
        return flatconfig_to_rank_u1u1_pascal(flatconfig, na, ka, nb, kb, pt)
    else:
        raise ValueError(
            r"Symmetry must be None, 'Z2', 'U1' or 'U1U1'. Got "
            f"{symmetry} instead."
        )


@njit(nogil=nogil)
def build_coo_numba_core(
    coupling_map,
    sector,
    symmetry,
    dtype=np.float64,
    world_size=1,
    world_rank=0,
):
    """Build the data for a sparse matrix in COO format.

    Parameters
    ----------
    coupling_map : tuple[ndarray]
        The operator defined as tuple of flat arrays.
    sector : tuple[int]
        Specifies the sector to convert.

        - (n,) for unconstrained hilbert space
        - (n, parity) for Z2 symmetry
        - (n, k) for U1 symmetry
        - (na, ka, nb, kb) for U1U1 symmetry

    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        Specifies the symmetry to use.
    dtype : {np.float64, np.complex128, np.float32, np.float64}, optional
        The dtype to use for the data. Default is np.float64.
    world_size : int, optional
        The number of processes in the world. Default is 1. Only rows
        corresponding to range(world_rank, D, world_size) will be computed.
        This is used for parallelization.
    world_rank : int, optional
        The rank of the current process. Default is 0. Only rows
        corresponding to range(world_rank, D, world_size) will be computed.
        This is used for parallelization.

    Returns
    -------
    data : ndarray[float64]
        The data for the sparse matrix in COO format.
    rows : ndarray[int64]
        The row indices for the sparse matrix in COO format.
    cols : ndarray[int64]
        The column indices for the sparse matrix in COO format.
    """
    if symmetry is None:
        # unconstrained hilbert space
        (n,) = sector
        return build_coo_numba_core_nosymm(
            n, coupling_map, dtype, world_size, world_rank
        )
    elif symmetry == "Z2":
        n, p = sector
        return build_coo_numba_core_z2(
            n, p, coupling_map, dtype, world_size, world_rank
        )

    elif symmetry == "U1":
        n, k = sector
        return build_coo_numba_core_u1(
            n, k, coupling_map, dtype, world_size, world_rank
        )
    elif symmetry == "U1U1":
        na, ka, nb, kb = sector
        return build_coo_numba_core_u1u1(
            na, ka, nb, kb, coupling_map, dtype, world_size, world_rank
        )
    else:
        raise ValueError(
            "Symmetry must be None, 'Z2', 'U1' "
            f"or 'U1U1'. Got {symmetry} instead."
        )


@njit(nogil=nogil)
def matvec_numba(
    x,
    out,
    coupling_map,
    sector,
    symmetry=None,
    world_size=1,
    world_rank=0,
):
    """Apply the operator defined by the coupling map to the input vector.

    Parameters
    ----------
    x : ndarray[float64]
        The input vector to apply the operator to.
    out : ndarray[float64]
        The output vector to store the result.
    coupling_map : tuple[ndarray]
        The operator defined as tuple of flat arrays.
    sector : tuple[int]
        Specifies the sector to convert.

        - (n,) for unconstrained hilbert space
        - (n, parity) for Z2 symmetry
        - (n, k) for U1 symmetry
        - (na, ka, nb, kb) for U1U1 symmetry

    symmetry : {None, "Z2", "U1", "U1U1"}, optional
        Specifies the symmetry to use.
    world_size : int, optional
        The number of processes in the world. Default is 1. Only rows
        corresponding to range(world_rank, D, world_size) will be computed.
        This is used for parallelization.
    world_rank : int, optional
        The rank of the current process. Default is 0. Only rows
        corresponding to range(world_rank, D, world_size) will be computed.
        This is used for parallelization.
    """
    if symmetry is None:
        # unconstrained hilbert space
        (n,) = sector
        matvec_nosymm(x, out, n, coupling_map, world_size, world_rank)
    elif symmetry == "Z2":
        n, p = sector
        matvec_z2(x, out, n, p, coupling_map, world_size, world_rank)
    elif symmetry == "U1":
        n, k = sector
        matvec_u1(x, out, n, k, coupling_map, world_size, world_rank)
    elif symmetry == "U1U1":
        na, ka, nb, kb = sector
        matvec_u1u1(
            x,
            out,
            na,
            ka,
            nb,
            kb,
            coupling_map,
            world_size,
            world_rank,
        )
    else:
        raise ValueError(
            "Symmetry must be None, 'Z2', 'U1' "
            f"or 'U1U1'. Got {symmetry} instead."
        )
