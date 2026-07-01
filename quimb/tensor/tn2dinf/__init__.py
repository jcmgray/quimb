"""Infinite, translation-invariant 2D tensor networks defined by a unit cell,
e.g. infinite PEPS and their imaginary-time simple update.

Some notable definitions and conventions of terms used in this subpackage:

- "site": a concrete (cell, site_type) pair defining a location site in the
  infinite lattice.
- "cell": (dx, dy) integer unit-cell translation (not a position).
- "site_type": label of a site within the unit cell, all sites with the same
  site_type share data.
- "bond_type": a bond's translation class, canonicalized with the first
  endpoint anchored at cell (0, 0) and the pair sorted.
- "bond index": the concrete index name on one specific translated bond.
- "canonical": the (0, 0) anchored representative keying a translation class.
- "fragment": the finite materialized TN patch standing in for the infinite
  network, that operations act on and are synced from.
- "shared_tensors" / "shared_indices": the set of tensors or indices in the
  fragment sharing the same site_type or bond_type, respectively.
- "position": the cartesian coordinate of a site, defined by the unit cell's
  basis and the site_type's fractional offset within the unit cell.
"""
