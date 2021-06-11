"""
SPDX-FileCopyrightText: 2021 Computer Assisted Medical Interventions Group, DKFZ
SPDX-FileCopyrightText: 2021 VISION Lab, Cancer Research UK Cambridge Institute (CRUK CI)
SPDX-License-Identifier: MIT
"""

# Configuration file for the Sphinx docs builder.
#
# This file only contains a selection of the most common options. For a full
# list see the docs:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# docs root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'SIMPA'
copyright = '2020-2021, CAMI (Computer Assisted Medical Interventions), DKFZ, Heidelberg and ' \
            '2021, Cancer Research UK, Cambridge Institute (CRUK CI)'
author = 'CAMI (Computer Assisted Medical Interventions), DKFZ, Heidelberg and ' \
         'Cancer Research UK, Cambridge Institute (CRUK CI)'

# The full version
# SIMPA syntax: RELEASE.MAJOR.MINOR
with open('../../VERSION', 'r') as readme_file:
    version = readme_file.read()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['rst2pdf.pdfbuilder', 'sphinx.ext.autodoc', 'm2r']
pdf_documents = [('index', 'docs', project, author)]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['images']
