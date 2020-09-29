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
import sys
import os
import nbformat
import datetime

now = datetime.datetime.now()


# -- Project information -----------------------------------------------------

project = 'QuantEcon.programming-python3'
copyright = '2020, Thomas J. Sargent and John Stachurski'
author = 'Thomas J. Sargent and John Stachurski'

# The full version, including alpha/beta/rc tags
version = '%s-%s-%s' % (now.year, now.strftime("%b"), now.day)
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax', 
	'sphinxcontrib.bibtex', 
	'IPython.sphinxext.ipython_console_highlighting',
    'myst_nb',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_static']

master_doc = "index"
master_pdf_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['source/_static']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Quantitative Economics with Python"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


#-Review-#

# Tikz HTML configuration for rendering images
tikz_latex_preamble = r"""
    \usetikzlibrary{arrows}
    \usetikzlibrary{calc}
    \usetikzlibrary{intersections}
    \usetikzlibrary{decorations}
    \usetikzlibrary{decorations.pathreplacing}
"""

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#'papersize': 'letterpaper',

# The font size ('10pt', '11pt' or '12pt').
'pointsize': '11pt',

# Additional stuff for the LaTeX preamble.
'preamble': r"""
\usepackage{amsmath, amssymb}
\usepackage{mathrsfs}

\usepackage{tikz}
\usetikzlibrary{arrows}
\usetikzlibrary{calc}
\usetikzlibrary{intersections}
\usetikzlibrary{decorations}
\usepackage{pgf}
\usepackage{pgfplots}


\usepackage{bbm}
\newcommand{\RR}{\mathbbm R}
\newcommand{\NN}{\mathbbm N}
\newcommand{\PP}{\mathbbm P}
\newcommand{\EE}{\mathbbm E \,}
\newcommand{\XX}{\mathbbm X}
\newcommand{\ZZ}{\mathbbm Z}
\newcommand{\QQ}{\mathbbm Q}

\newcommand{\fF}{\mathcal F}
\newcommand{\dD}{\mathcal D}
\newcommand{\lL}{\mathcal L}
\newcommand{\gG}{\mathcal G}
\newcommand{\hH}{\mathcal H}
\newcommand{\nN}{\mathcal N}
\newcommand{\pP}{\mathcal P}

\DeclareMathOperator{\trace}{trace}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Span}{span}
\DeclareMathOperator{\proj}{proj}
\DeclareMathOperator{\col}{col}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

\usepackage{makeidx}
\makeindex
""",

# Latex figure (float) alignment (Could use 'H' to force the placement of figures)
'figure_align': 'H',#'htbp',

#Add Frontmatter before TOC
'tableofcontents' : r"""\newpage
\thispagestyle{empty}
\chapter*{Preface}
\large
This \textbf{pdf} presents a series of lectures on quantitative economic
modeling, designed and written by \href{http://www.tomsargent.com/}{Thomas J. Sargent} and \href{http://johnstachurski.net}{John Stachurski}.  
The primary programming languages are \href{https://www.python.org}{Python} and \href{http://julialang.org/}{Julia}.
You can send feedback to the authors via contact@quantecon.org.

\vspace{5em}

\begin{leftbar}
\textbf{Note: You are currently viewing an automatically generated
pdf version of our online lectures,} which are located at

\vspace{2em}

\begin{center}
  \texttt{https://lectures.quantecon.org}
\end{center}

\vspace{2em}

Please visit the website for more information on the aims and scope of the
lectures and the two language options (Julia or Python).

\vspace{1em}

Due to automatic generation of this pdf, \textbf{presentation quality is likely
to be lower than that of the website}.

\end{leftbar}

\normalsize

\sphinxtableofcontents
"""
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_pdf_doc, 'QuantEconlectures-python3.tex', 'QuantEcon.lectures-python3 PDF',
     'Thomas J. Sargent and John Stachurski', 'manual'),
]

# ------------------
# Linkcheck Options
# ------------------

linkcheck_ignore = [r'https:\/\/github\.com\/.*?#.*'] #Anchors on Github seem to create issues with linkchecker

linkcheck_timeout = 30 

