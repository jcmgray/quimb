#!/usr/bin/env python3
"""Resolve sphinx-style cross-references in quimb changelog markdown
to full URLs suitable for GitHub release notes.

Usage:
    python resolve_changelog.py input.md [output.md]

If output.md is not given, prints to stdout.
"""

import re
import sys
from pathlib import Path

BASE = "https://quimb.readthedocs.io/en/latest/autoapi"


def find_package_root():
    """Find the local ``quimb`` source directory."""
    candidates = (
        Path(__file__).resolve().parents[2] / "quimb",
        Path.cwd() / "quimb",
    )
    for candidate in candidates:
        if (candidate / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError("Could not find local quimb package directory.")


def find_autoapi_modules(package_root=None):
    """Find module pages that sphinx-autoapi should generate for ``quimb``.

    This intentionally scans files rather than importing modules, since this
    script is used as a release-note helper and should have no import side
    effects.
    """
    if package_root is None:
        package_root = find_package_root()
    else:
        package_root = Path(package_root)

    modules = set()
    for path in package_root.rglob("*.py"):
        rel = path.relative_to(package_root).with_suffix("")
        parts = rel.parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.add(".".join(("quimb", *parts)))

    return frozenset(modules)


KNOWN_MODULES = find_autoapi_modules()


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
