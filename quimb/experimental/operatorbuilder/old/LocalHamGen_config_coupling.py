"""Config coupling method directly for a LocalHamGen object."""


def config_coupling(self, config, atol=1e-12):
    """Take the configuration ``config`` and return the list of other
    configurations this Hamiltonian couples it to and the corresponding
    coefficients.

    Parameters
    ----------
    config : dict[hashable, int]
        A mapping of each site to a integer value, ``[0, 1]``.
    atol : float, optional
        Absolute tolerance for the skipping couplings.

    Returns
    -------
    coupled_configs : list[dict[hashable, int]]
        List of configurations ``config`` is coupled to.
    coupling_coeffs : list[float]
        List of coupling coefficients.
    """

    coupled_coeffs = {}
    coupled_configs = []

    for coo, h in self.items():
        # get the local row
        sitevals = [config[site] for site in coo]
        hrow = h[sum(2**i * x for i, x in enumerate(reversed(sitevals)))]

        # see which local coupled_configs it couples to
        for j, hij in enumerate(hrow):
            if -atol < hij < atol:
                continue

            # update the config
            newconfig = config.copy()
            for site, siteval in zip(coo, tuple(map(int, f"{j:0>2b}"))):
                newconfig[site] = siteval

            # accumulate repeated coupled_configs
            key = tuple(newconfig.values())
            if key not in coupled_coeffs:
                coupled_configs.append(newconfig)
                coupled_coeffs[key] = hij
            else:
                coupled_coeffs[key] += hij

    return coupled_configs, list(coupled_coeffs.values())
