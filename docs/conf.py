#!/usr/bin/env python
#
# xbitinfo documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import datetime
import os
import sys

import xarray

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#

sys.path.insert(0, os.path.abspath(".."))

import xbitinfo  # noqa: E402

xarray.DataArray.__module__ = "xarray"
xarray.Dataset.__module__ = "xarray"


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = True
autodoc_member_order = "groupwise"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
current_year = datetime.datetime.now().year

project = "xbitinfo"
copyright = f"{current_year}, Hauke Schulz"
author = "Hauke Schulz"
today_fmt = "%Y-%m-%d"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = xbitinfo.__version__
# The full version, including alpha/beta/rc tags.
release = xbitinfo.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"


# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "repository_url": "https://github.com/observingClouds/xbitinfo",
    "use_edit_page_button": True,
    "navbar_end": "search-field.html",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": False,
    "extra_navbar": "",
    "navbar_footer_text": "",
    "navigation_with_keys": False,
}

html_context = {
    "github_user": "observingClouds",
    "github_repo": "xbitinfo",
    "github_version": "main",
    "doc_path": "docs",
}

html_logo = "_static/xbitinfo_logo.svg"


myst_url_schemes = ["http", "https", "mailto"]

# Cupybutton configuration
# See: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

extlinks = {
    "issue": ("https://github.com/observingClouds/xbitinfo/issues/%s", "GH#%s"),
    "pr": ("https://github.com/observingClouds/xbitinfo/pull/%s", "GH#%s"),
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
}


# Should only be uncommented when testing page development while notebooks
# are breaking.
# nbsphinx_kernel_name = "bitinfo-docs"  # doesnt work
nb_execution_timeout = 120
nb_execution_mode = "auto"
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_show_tb = True

# Napoleon configurations
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {"DataArray": "~xarray.DataArray", "Dataset": "~xarray.Dataset"}

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "xbitinfodoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "xbitinfo.tex", "xbitinfo Documentation", "Hauke Schulz", "manual"),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "xbitinfo", "xbitinfo Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "xbitinfo",
        "xbitinfo Documentation",
        author,
        "xbitinfo",
        "One line description of project.",
        "Miscellaneous",
    ),
]


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def html_page_context(app, pagename, templatename, context, doctree):
    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False


def setup(app):
    app.connect("source-read", rstjinja)
    app.connect("html-page-context", html_page_context)
