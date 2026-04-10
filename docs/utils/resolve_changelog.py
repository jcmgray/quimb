#!/usr/bin/env python3
"""Resolve sphinx-style cross-references in quimb changelog markdown
to full URLs suitable for GitHub release notes.

Usage:
    python resolve_changelog.py input.md [output.md]

If output.md is not given, prints to stdout.
"""

import re
import sys

BASE = "https://quimb.readthedocs.io/en/latest/autoapi"

# Known autoapi module pages (update if new modules are added)
KNOWN_MODULES = {
    "quimb",
    "quimb.calc",
    "quimb.core",
    "quimb.evo",
    "quimb.gates",
    "quimb.gen",
    "quimb.gen.operators",
    "quimb.gen.rand",
    "quimb.gen.states",
    "quimb.linalg",
    "quimb.operator",
    "quimb.operator.builder",
    "quimb.operator.models",
    "quimb.operator.pepobuilder",
    "quimb.schematic",
    "quimb.tensor",
    "quimb.tensor.array_ops",
    "quimb.tensor.belief_propagation",
    "quimb.tensor.belief_propagation.bp_common",
    "quimb.tensor.belief_propagation.d1bp",
    "quimb.tensor.belief_propagation.d2bp",
    "quimb.tensor.belief_propagation.diis",
    "quimb.tensor.belief_propagation.hd1bp",
    "quimb.tensor.belief_propagation.hv1bp",
    "quimb.tensor.belief_propagation.l1bp",
    "quimb.tensor.belief_propagation.l2bp",
    "quimb.tensor.belief_propagation.regions",
    "quimb.tensor.circuit",
    "quimb.tensor.circuit_gen",
    "quimb.tensor.contraction",
    "quimb.tensor.decomp",
    "quimb.tensor.drawing",
    "quimb.tensor.fitting",
    "quimb.tensor.gating",
    "quimb.tensor.geometry",
    "quimb.tensor.interface",
    "quimb.tensor.networking",
    "quimb.tensor.optimize",
    "quimb.tensor.tensor_1d",
    "quimb.tensor.tensor_1d_compress",
    "quimb.tensor.tensor_1d_tebd",
    "quimb.tensor.tensor_2d",
    "quimb.tensor.tensor_2d_compress",
    "quimb.tensor.tensor_2d_tebd",
    "quimb.tensor.tensor_3d",
    "quimb.tensor.tensor_3d_tebd",
    "quimb.tensor.tensor_approx_spectral",
    "quimb.tensor.tensor_arbgeom",
    "quimb.tensor.tensor_arbgeom_compress",
    "quimb.tensor.tensor_arbgeom_tebd",
    "quimb.tensor.tensor_builder",
    "quimb.tensor.tensor_core",
    "quimb.tensor.tensor_dmrg",
    "quimb.tensor.tensor_mera",
    "quimb.tensor.tn1d",
    "quimb.tensor.tn1d.compress",
    "quimb.tensor.tn1d.core",
    "quimb.tensor.tn1d.dmrg",
    "quimb.tensor.tn1d.mera",
    "quimb.tensor.tn1d.tebd",
    "quimb.tensor.tn2d",
    "quimb.tensor.tn2d.compress",
    "quimb.tensor.tn2d.core",
    "quimb.tensor.tn2d.tebd",
    "quimb.tensor.tn3d",
    "quimb.tensor.tn3d.core",
    "quimb.tensor.tn3d.tebd",
    "quimb.tensor.tnag",
    "quimb.tensor.tnag.compress",
    "quimb.tensor.tnag.core",
    "quimb.tensor.tnag.tebd",
    "quimb.utils",
    "quimb.utils_plot",
}


def fqn_to_url(fqn):
    """Convert a fully qualified Python name to its autoapi URL."""
    parts = fqn.split(".")

    # Find the longest known module prefix
    best_module = None
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        if candidate in KNOWN_MODULES:
            best_module = candidate
            break

    if best_module is None:
        # Fallback: assume everything except last component is the module
        best_module = ".".join(parts[:-1]) if len(parts) > 1 else fqn

    module_path = best_module.replace(".", "/")

    if fqn == best_module:
        return f"{BASE}/{module_path}/index.html"

    return f"{BASE}/{module_path}/index.html#{fqn}"


def resolve_links(text):
    """Resolve all sphinx-style references in markdown text."""

    # 1. Resolve [text](quimb.x.y.z) -> [text](url)
    def _resolve_fqn(m):
        link_text = m.group(1)
        target = m.group(2)
        if not target.startswith("quimb."):
            return m.group(0)
        return f"[{link_text}]({fqn_to_url(target)})"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _resolve_fqn, text)

    # 2. Resolve {issue}`NUM` and {pr}`NUM` -> #NUM
    text = re.sub(r"\{issue\}`(\d+)`", r"#\1", text)
    text = re.sub(r"\{pr\}`(\d+)`", r"#\1", text)

    return text


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    with open(sys.argv[1]) as f:
        text = f.read()

    result = resolve_links(text)

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
