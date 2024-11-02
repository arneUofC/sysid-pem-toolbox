# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'sysid_pem_toolbox'
copyright = '2024, Arne Dankers'
author = 'Arne Dankers'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically document functions/classes
    'sphinx.ext.napoleon',  # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',  # Link to source code
]

# Use Read the Docs theme if available
html_theme = 'sphinx_rtd_theme'

# Set autodoc to follow the order of members in the source code
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
