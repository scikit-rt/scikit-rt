# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# For configuration of myst_parser extension, see:
# https://myst-parser.readthedocs.io/en/latest/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
"""
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/skrt'))
sys.path.insert(0, os.path.abspath('../../src/skrt/better_viewer'))
sys.path.insert(0, os.path.abspath('../../src/skrt/viewer'))
sys.path.insert(0, os.path.abspath('../../examples/voxtox/src'))
sys.path.insert(0, os.path.abspath('../../examples/import/src'))
"""

print(f'cwd: {os.getcwd()}')
print(f'sys.path: {sys.path}')

import import_analysis
import skrt
import voxtox

# -- Project information -----------------------------------------------------

project = 'Scikit-rt'
copyright = '2021-2024'
#author = ''

# The full version, including alpha/beta/rc tags
release = '0.8.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'myst_parser']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Heading level depth to assign HTML anchors (default: None)
myst_heading_anchors = 2

# Include documentation for special members.
# Based on suggestin at:
# https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
def skip(app, what, name, obj, would_skip, options):
    if name in ['__init__', '__repr__']:
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)
