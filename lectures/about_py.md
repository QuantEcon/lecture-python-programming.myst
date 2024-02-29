---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(about_py)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# About Python

```{contents} Contents
:depth: 2
```

```{epigraph}
"Python has gotten sufficiently weapons grade that we don’t descend into R
anymore. Sorry, R people. I used to be one of you but we no longer descend
into R." -- Chris Wiggins
```

## Overview

In this lecture we will

* outline what Python is
* compare it to some other languages
* showcase some of its abilities.

At this stage, it's **not** our intention that you try to replicate all you see.

We will work through what follows at a slow pace later in the lecture series.

Our only objective for this lecture is to give you some feel of what Python is, and what it can do.

## What's Python?

[Python](https://www.python.org) is a general-purpose programming language conceived in 1989 by Dutch programmer [Guido van Rossum](https://en.wikipedia.org/wiki/Guido_van_Rossum).

Python is free and open source, with development coordinated through the [Python Software Foundation](https://www.python.org/psf/).

Python has experienced rapid adoption in the last decade and is now one of the [most popular programming languages](https://www.tiobe.com/tiobe-index/).

### Common Uses

{index}`Python <single: Python; common uses>` is a general-purpose language used in almost all application domains such as

* AI
* communication
* web development
* CGI and graphical user interfaces
* game development
* resource planning
* multimedia, data science, security, etc., etc., etc.

Used and supported extensively by Internet services and high-tech companies including

* [Google](https://www.google.com/)
* [Netflix](https://www.netflix.com/)
* [Meta](https://opensource.fb.com/)
* [Dropbox](https://www.dropbox.com/)
* [Amazon](https://www.amazon.com/)
* [Reddit](https://www.reddit.com/)

For reasons we will discuss, Python is particularly popular within the scientific community 

Meanwhile, Python is also very beginner-friendly and is found to be suitable for
students learning programming and recommended to introduce computational methods
to students in fields other than computer science.

Python is also replacing familiar tools like Excel as an essential skill in the fields of finance and banking.


### Relative Popularity

The following chart, produced using Stack Overflow Trends, shows one measure of the relative popularity of Python

```{figure} /_static/lecture_specific/about_py/python_vs_matlab.png
```

The figure indicates not only that Python is widely used but also that adoption of Python has accelerated significantly since 2012.

This is driven at least in part by uptake in the scientific domain, particularly in rapidly growing fields like data science and AI.


### Features

Python is a [high-level language](https://en.wikipedia.org/wiki/High-level_programming_language) suitable for rapid development.

It has a relatively small core language supported by many libraries.

Other features of Python:

* multiple programming styles are supported (procedural, object-oriented, functional, etc.)
* it is interpreted rather than compiled.

### Syntax and Design

```{index} single: Python; syntax and design
```

One nice feature of Python is its elegant syntax --- we'll see many examples later on.

Elegant code might sound superfluous but in fact it's highly beneficial because it makes the syntax easy to read and easy to remember.

Closely related to elegant syntax is an elegant design.

Features like iterators, generators, decorators and list comprehensions make Python highly expressive, allowing you to get more done with less code.

[Namespaces](https://en.wikipedia.org/wiki/Namespace) improve productivity by cutting down on bugs and syntax errors.


## Scientific Programming

```{index} single: scientific programming
```

Python has become one of the core languages of scientific computing.

It's either the dominant player or a major player in

* AI, machine learning and data science
* astronomy
* chemistry
* computational biology
* meteorology
* natural language processing
* etc.

This section briefly showcases some examples of Python for scientific programming.

* All of these topics below will be covered in detail later on.


### Numerical Programming

```{index} single: scientific programming; numeric
```

Fundamental matrix and array processing capabilities are provided by the excellent [NumPy](http://www.numpy.org/) library.

NumPy provides the basic array data type plus some simple processing operations.

For example, let's build some arrays

```{code-cell} python3
import numpy as np                     # Load the library

a = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
b = np.cos(a)                          # Apply cosine to each element of a
c = np.sin(a)                          # Apply sin to each element of a
```

Now let's take the inner product

```{code-cell} python3
b @ c
```

The number you see here might vary slightly but it's essentially zero.

(For older versions of Python and NumPy you need to use the [np.dot](http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html) function)

The [SciPy](http://www.scipy.org) library is built on top of NumPy and provides additional functionality.

(tuple_unpacking_example)=
For example, let's calculate $\int_{-2}^2 \phi(z) dz$ where $\phi$ is the standard normal density.

```{code-cell} python3
from scipy.stats import norm
from scipy.integrate import quad

ϕ = norm()
value, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
value
```

SciPy includes many of the standard routines used in

* [linear algebra](http://docs.scipy.org/doc/scipy/reference/linalg.html)
* [integration](http://docs.scipy.org/doc/scipy/reference/integrate.html)
* [interpolation](http://docs.scipy.org/doc/scipy/reference/interpolate.html)
* [optimization](http://docs.scipy.org/doc/scipy/reference/optimize.html)
* [distributions and statistical techniques](http://docs.scipy.org/doc/scipy/reference/stats.html)
* [signal processing](http://docs.scipy.org/doc/scipy/reference/signal.html)

See them all [here](http://docs.scipy.org/doc/scipy/reference/index.html).

### Graphics

```{index} single: Matplotlib
```

The most popular and comprehensive Python library for creating figures and graphs is [Matplotlib](http://matplotlib.org/), with functionality including

* plots, histograms, contour images, 3D graphs, bar charts etc.
* output in many formats (PDF, PNG, EPS, etc.)
* LaTeX integration

Example 2D plot with embedded LaTeX annotations

```{figure} /_static/lecture_specific/about_py/qs.png
:scale: 75
```

Example contour plot

```{figure} /_static/lecture_specific/about_py/bn_density1.png
:scale: 70
```

Example 3D plot

```{figure} /_static/lecture_specific/about_py/career_vf.png
```

More examples can be found in the [Matplotlib thumbnail gallery](https://matplotlib.org/stable/gallery/index.html).

Other graphics libraries include

* [Plotly](https://plot.ly/python/)
* [seaborn](https://seaborn.pydata.org/) --- a high-level interface for matplotlib
* [Altair](https://altair-viz.github.io/)
* [Bokeh](http://bokeh.pydata.org/en/latest/)

You can visit the [Python Graph Gallery](https://www.python-graph-gallery.com/) for more example plots drawn using a variety of libraries.

### Networks and Graphs

Python has many libraries for studying graphs.

```{index} single: NetworkX
```

One well-known example is [NetworkX](http://networkx.github.io/).
Its features include, among many other things:

* standard graph algorithms for analyzing networks
* plotting routines

Here's some example code that generates and plots a random graph, with node color determined by the shortest path length from a central node.

```{code-cell} ipython
%matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)
np.random.seed(1234)

# Generate a random graph
p = dict((i, (np.random.uniform(0, 1), np.random.uniform(0, 1)))
         for i in range(200))
g = nx.random_geometric_graph(200, 0.12, pos=p)
pos = nx.get_node_attributes(g, 'pos')

# Find node nearest the center point (0.5, 0.5)
dists = [(x - 0.5)**2 + (y - 0.5)**2 for x, y in list(pos.values())]
ncenter = np.argmin(dists)

# Plot graph, coloring by path length from central node
p = nx.single_source_shortest_path_length(g, ncenter)
plt.figure()
nx.draw_networkx_edges(g, pos, alpha=0.4)
nx.draw_networkx_nodes(g,
                       pos,
                       nodelist=list(p.keys()),
                       node_size=120, alpha=0.5,
                       node_color=list(p.values()),
                       cmap=plt.cm.jet_r)
plt.show()
```

### Other Scientific Libraries

Here's a short list of more important scientific libraries for Python.

* [SymPy](http://www.sympy.org/) for symbolic algebra, including limits, derivatives and integrals
* [pandas](http://pandas.pydata.org/) for data maniputation
* [statsmodels](http://statsmodels.sourceforge.net/) for statistical routines
* [scikit-learn](http://scikit-learn.org/) for machine learning 
* [JAX](https://github.com/google/jax) for automatic differentiation, accelerated linear algebra and GPU computing
* [PyTorch](https://pytorch.org/) for deep learning 
* [Keras](https://keras.io/) for machine learning
* [Pyro](https://pyro.ai/) and [PyStan](https://pystan.readthedocs.org/en/latest/) for Bayesian data analysis 
* [lifelines](https://lifelines.readthedocs.io/en/latest/) for survival analysis
* [GeoPandas](https://geopandas.org/en/stable/) for spatial data analysis
* [Dask](https://docs.dask.org/en/stable/) for parallelization
* [Numba](http://numba.pydata.org/) for making Python run at the same speed as native machine code
* [CVXPY](https://www.cvxpy.org/) for convex optimization 
* [PyTables](http://www.pytables.org) for managing large data sets
* [scikit-image](https://scikit-image.org/) and [OpenCV](https://opencv.org/) for processing and analysing image data
* [FLAML](https://mlflow.org/docs/latest/index.html) for automated machine learning and hyperparameter tuning
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for extracting data from HTML and XML files


In this lecture series we will learn how to use many of these libraries for
scientific computing tasks in economics and finance.
