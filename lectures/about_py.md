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

Python has experienced rapid adoption in the last decade and is now one of the most popular programming languages.

### Common Uses

{index}`Python <single: Python; common uses>` is a general-purpose language used in almost all application domains such as

* communications
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

For reasons we will discuss, Python is particularly popular within the scientific community and behind many scientific achievements in 
* [Space Science](https://code.nasa.gov/?q=python)
* [Particle Physics](https://home.cern/news/news/physics/speeding-machine-learning-particle-physics)
* [Genetics](https://github.com/deepmind/alphafold)

and practically all branches of academia.

Meanwhile, Python is also very beginner-friendly and is found to be suitable for students learning programming and recommended to introduce computational methods to students in [fields other than computer science](https://www.sciencedirect.com/science/article/pii/S1477388021000177).

Python is also [replacing familiar tools like Excel as an essential skill](https://www.efinancialcareers.com.au/news/2021/08/python-for-banking-jobs) in the fields of finance and banking.

### Relative Popularity

The following chart, produced using Stack Overflow Trends, shows one measure of the relative popularity of Python

```{figure} /_static/lecture_specific/about_py/python_vs_matlab.png
```

The figure indicates not only that Python is widely used but also that adoption of Python has accelerated significantly since 2012.

We suspect this is driven at least in part by uptake in the scientific
domain, particularly in rapidly growing fields like data science.

For example, the popularity of [pandas](http://pandas.pydata.org/), a library for data analysis with Python has exploded, as seen here.

(The corresponding time path for MATLAB is shown for comparison)

```{figure} /_static/lecture_specific/about_py/pandas_vs_matlab.png
```

Note that pandas takes off in 2012, which is the same year that we see
Python's popularity begin to spike in the first figure.

Overall, it's clear that

* Python is [one of the most popular programming languages worldwide](https://spectrum.ieee.org/top-programming-languages-2021).
* Python is a major tool for scientific computing, accounting for a rapidly rising share of scientific work around the globe.

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

Remembering how to read from files, sort dictionaries and other such routine tasks means that you don't need to break your flow in order to hunt down correct syntax.

Closely related to elegant syntax is an elegant design.

Features like iterators, generators, decorators and list comprehensions make Python highly expressive, allowing you to get more done with less code.

[Namespaces](https://en.wikipedia.org/wiki/Namespace) improve productivity by cutting down on bugs and syntax errors.

## Scientific Programming

```{index} single: scientific programming
```

Python has become one of the core languages of scientific computing.

It's either the dominant player or a major player in

* [machine learning and data science](http://scikit-learn.org/stable/)
* [astronomy](http://www.astropy.org/)
* [artificial intelligence](https://wiki.python.org/moin/PythonForArtificialIntelligence)
* [chemistry](http://chemlab.github.io/chemlab/)
* [computational biology](http://biopython.org/wiki/Main_Page)
* [meteorology](https://pypi.org/project/meteorology/)
* [natural language processing](https://www.nltk.org/)

Its popularity in economics is also beginning to rise.

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
* [Bokeh](http://bokeh.pydata.org/en/latest/)

You can visit the [Python Graph Gallery](https://www.python-graph-gallery.com/) for more example plots drawn using a variety of libraries.

### Symbolic Algebra

It's useful to be able to manipulate symbolic expressions, as in Mathematica or Maple.

```{index} single: SymPy
```

The [SymPy](http://www.sympy.org/) library provides this functionality from within the Python shell.

```{code-cell} python3
from sympy import Symbol

x, y = Symbol('x'), Symbol('y')  # Treat 'x' and 'y' as algebraic symbols
x + x + x + y
```

We can manipulate expressions

```{code-cell} python3
expression = (x + y)**2
expression.expand()
```

solve polynomials

```{code-cell} python3
from sympy import solve

solve(x**2 + x + 2)
```

and calculate limits, derivatives and integrals

```{code-cell} python3
from sympy import limit, sin, diff, integrate

limit(1 / x, x, 0)
```

```{code-cell} python3
limit(sin(x) / x, x, 0)
```

```{code-cell} python3
diff(sin(x), x)
```

```{code-cell} python3
integrate(sin(x) * x, x)
```

The beauty of importing this functionality into Python is that we are working within a fully fledged programming language.

We can easily create tables of derivatives, generate LaTeX output, add that output to figures and so on.

### Statistics

Python's data manipulation and statistics libraries have improved rapidly over
the last few years to tackle
[specific problems in data science](https://ieeexplore.ieee.org/document/8757088).

#### Pandas

```{index} single: Pandas
```

One of the most popular libraries for working with data is [pandas](http://pandas.pydata.org/).

Pandas is fast, efficient, flexible and well designed.

Here's a simple example, using some dummy data generated with Numpy's excellent
`random` functionality.

```{code-cell} python3
import pandas as pd
np.random.seed(1234)

data = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
dates = pd.date_range('2010-12-28', periods=5)

df = pd.DataFrame(data, columns=('price', 'weight'), index=dates)
print(df)
```

```{code-cell} python3
df.mean()
```


#### Other Useful Statistics and Data Science Libraries

```{index} single: statsmodels
```

* [statsmodels](http://statsmodels.sourceforge.net/) --- various statistical routines

```{index} single: scikit-learn
```

* [scikit-learn](http://scikit-learn.org/) --- Machine Learning in Python

```{index} single: PyTorch
```

* [PyTorch](https://pytorch.org/) --- Deep learning framework in Python and other major competitors in the field including [TensorFlow](https://www.tensorflow.org/overview) and [Keras](https://keras.io/)

```{index} single: Pyro
```

* [Pyro](https://pyro.ai/) and [PyStan](https://pystan.readthedocs.org/en/latest/) --- for Bayesian data analysis building on [Pytorch](https://pytorch.org/) and [stan](http://mc-stan.org/) respectively

```{index} single: lifelines
```

* [lifelines](https://lifelines.readthedocs.io/en/latest/) --- for survival analysis

```{index} single: GeoPandas
```

* [GeoPandas](https://geopandas.org/en/stable/) --- for spatial data analysis


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

### Cloud Computing

```{index} single: cloud computing
```

Running your Python code on massive servers in the cloud is becoming easier and easier.

```{index} single: cloud computing; google colab
```

An excellent example of the portability of python in a cloud computing environment is [Google Colab](https://colab.research.google.com/). It hosts the Jupyter notebook on cloud servers with no pre-configuration necessary to run Python code using cloud servers.


There are also commercial applications of cloud computing using Python:

```{index} single: cloud computing; anaconda enterprise
```
* [Anaconda Enterprise](https://www.anaconda.com/enterprise/)

```{index} single: cloud computing; AWS
```

* [Amazon Web Services](https://aws.amazon.com/developer/language/python/?nc1=f_dr)

```{index} single: cloud computing; Google Cloud
```

* [Google Cloud](https://cloud.google.com/)

```{index} single: cloud computing; digital ocean
```

* [Digital Ocean](https://www.digitalocean.com/)


### Parallel Processing

```{index} single: parallel computing
```

Apart from the cloud computing options listed above, you might like to consider

```{index} single: parallel computing; ipython
```

* [Parallel computing through IPython clusters](https://ipyparallel.readthedocs.io/en/latest/).


```{index} single: parallel computing; Dask
```

* [Dask](https://www.dask.org/) parallelises PyData and Machine Learning in Python.

```{index} single: parallel computing; pycuda
```

* GPU programming through [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), [PyCuda](https://wiki.tiker.net/PyCuda), [PyOpenCL](https://documen.tician.de/pyopencl/), [Rapids](https://rapids.ai/), etc.


Here is more about [recent developments](https://pasc22.pasc-conference.org/program/papers/) in high-performance computing (HPC) in scientific computing and [how HPC helps researchers in different fields](https://pasc22.pasc-conference.org/program/keynote-presentations/). 

(intfc)=
### Other Developments

There are many other interesting developments with scientific programming in Python.

Some representative examples include

```{index} single: scientific programming; Jupyter
```

* [Jupyter](http://jupyter.org/) --- Python in your browser with interactive code cells,  embedded images and other useful features.

```{index} single: scientific programming; Numba
```

* [Numba](http://numba.pydata.org/) --- make Python run at the same speed as native machine code!

```{index} single: scientific programming; CVXPY
```

* [CVXPY](https://www.cvxpy.org/) --- convex optimization in Python.


```{index} single: scientific programming; PyTables
```

* [PyTables](http://www.pytables.org) --- manage large data sets.


```{index} single: scientific programming; scikit-image
```

* [scikit-image](https://scikit-image.org/) and [OpenCV](https://opencv.org/) --- process and analyse scientific image data.


```{index} single: scientific programming; mlflow
```

* [FLAML](https://mlflow.org/docs/latest/index.html) --- automate machine learning and hyperparameter tuning.


```{index} single: scientific programming; BeautifulSoup
```

* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) --- extract data from HTML and XML files.

```{index} single: scientific programming; PyInstaller
```

* [PyInstaller](https://pyinstaller.org/en/stable/) --- create packaged app from python script.

## Learn More

* Browse some Python projects on [GitHub](https://github.com/trending?l=python).
* Read more about [Python's history and rise in popularity](https://www.welcometothejungle.com/en/articles/btc-python-popular) and [version history](https://www.python.org/doc/versions/).
* Have a look at [some of the Jupyter notebooks](http://nbviewer.jupyter.org/) people have shared on various scientific topics.

```{index} single: Python; PyPI
```

* Visit the [Python Package Index](https://pypi.org/).
* View some of the questions people are asking about Python on [Stackoverflow](http://stackoverflow.com/questions/tagged/python).
* Keep up to date on what's happening in the Python community with the [Python subreddit](https://www.reddit.com:443/r/Python/).

