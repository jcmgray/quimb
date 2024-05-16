"""Helper functions for computing marginals of classical partition functions /
SAT problems.
"""


def compute_all_marginals_via_slicing(
    tn,
    output_inds,
    optimize="auto-hq",
    **contract_kwargs,
):

    tree0 = tn.contraction_tree(output_inds=(), optimize=optimize)
    arrays = tn.arrays

    w = {}
    symbol_map = tn.get_symbol_map()

    for ix in output_inds:

        # convert quimb index to cotengra unicode symbol
        symbol = symbol_map[ix]
        symbol

        if symbol not in tree0.sliced_inds:
            tree_v = tree0.remove_ind(symbol)
        else:
            # already sliced
            tree_v = tree0

        overall_exponent = None

        # initialize each value of this index
        wv = [0.0 for i in range(tree_v.size_dict[symbol])]

        for s in range(tree_v.nslices):

            # contract the slice
            p, exponent = tree_v.contract_slice(
                arrays, s, **contract_kwargs, strip_exponent=True
            )

            if overall_exponent is None:
                # set overall exponent from first slice...
                overall_exponent = exponent

            # ... correct subsequent slices relative to first
            rel_exponent = exponent - overall_exponent

            # what index combination is this slice
            key = tree_v.slice_key(s)

            # add result to correct index position
            wv[key[symbol]] += p * 10 ** rel_exponent

        w[ix] = (wv, overall_exponent)

    return w


def compute_all_marginals_via_slicing_shared(
    tn,
    output_inds,
    optimize="auto-hq",
    **contract_kwargs,
):
    from autoray import do, lazy

    tnc = tn.copy()

    with lazy.shared_intermediates():
        tnc.apply_to_arrays(lazy.array)
        w = compute_all_marginals_via_slicing(
            tnc, output_inds=output_inds, optimize=optimize, **contract_kwargs
        )

    # stack into a single lazy array
    return do(
        'stack', tuple(
            do('stack', (w[ix][0][0], w[ix][0][1], w[ix][1]))
            for ix in output_inds
        )
    )


def compute_all_marginals_via_torch_autodiff(
    tn,
    output_inds,
    optimize="auto-hq",
    equalize_norms=1.0,
    contraction_width_error_threshold=float("inf"),
):
    import torch
    from autoray import to_backend_dtype

    tnc = tn.copy()
    if equalize_norms:
        tnc.equalize_norms_(equalize_norms)

    tnc.apply_to_arrays(torch.tensor)

    variables = {}
    for ix in output_inds:
        # create a flat input tensor for each desired output
        data = torch.ones(
            tn.ind_size(ix),
            dtype=to_backend_dtype(tnc.dtype, "torch"),
            requires_grad=True,
        )
        variables[ix] = data
        # multiply it into any connected tensor, so we don't change geometry
        t = next(iter(tnc._inds_get(ix)))
        t.multiply_index_diagonal_(ix, data)

    # find the contraction tree
    tree = tnc.contraction_tree(output_inds=(), optimize=optimize)

    if tree.contraction_width() > contraction_width_error_threshold:
        raise ValueError("Contraction width is above threshold.")

    # perform the forward contraction,
    # compute in log10, mantissa should always be 1
    _, exponent = tree.contract(tnc.arrays, strip_exponent=True)
    exponent.backward()

    # factor is just to correct for log derivative
    return {
        ix: 2.302585092994046 * data.grad.detach().cpu().numpy()
        for ix, data in variables.items()
    }
