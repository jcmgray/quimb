# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys, os
sys.path.append(os.path.abspath("./_pygments"))


# -- Project information -----------------------------------------------------

project = 'quimb'
copyright = '2022, Johnnie Gray'
author = 'Johnnie Gray'

# The full version, including alpha/beta/rc tags
try:
    from quimb import __version__
    release = __version__
except ImportError:
    try:
        from importlib.metadata import version as _version
        release = _version('quimb')
    except ImportError:
        release = '0.0.0+unknown'

version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'autoapi.extension',
]

# msyt_nb configuration
nb_execution_mode = "off"
myst_heading_anchors = 4
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]


# sphinx-autoapi
autoapi_dirs = ['../quimb']

extlinks = {
    'issue': ('https://github.com/jcmgray/quimb/issues/%s', 'GH'),
    'pull': ('https://github.com/jcmgray/quimb/pull/%s', 'PR'),
}
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        # "color-brand-primary": "hsl(45, 80%, 45%)",
        "color-brand-primary": "hsl(210, 50%, 50%)",
        "color-brand-content": "hsl(210, 50%, 50%)",
    },
    "dark_css_variables": {
        "color-brand-primary": "hsl(210, 50%, 60%)",
        "color-brand-content": "hsl(210, 50%, 60%)",
    },
    "light_logo": "quimb_logo_title.png",
    "dark_logo": "quimb_logo_title.png",
}

pygments_style = '_pygments_light.MarianaLight'
pygments_dark_style = "_pygments_dark.MarianaDark"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ["my-styles.css"]
html_favicon = "_static/quimb.ico"


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    import quimb
    import inspect

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(quimb.__file__))

    if "+" in quimb.__version__:
        return (
            f"https://github.com/jcmgray/quimb/blob/"
            f"develop/quimb/{fn}{linespec}"
        )
    else:
        return (
            f"https://github.com/pydata/quimb/blob/"
            f"v{quimb.__version__}/quimb/{fn}{linespec}"
        )
