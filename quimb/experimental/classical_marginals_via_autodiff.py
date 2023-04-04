"""Helper functions for Computing marginals of classical partition functions /
SAT problems using automatic differentiation.
"""


def compute_all_marginals_via_torch(
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
