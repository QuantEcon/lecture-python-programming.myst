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

(sp)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`SciPy <single: SciPy>`

```{index} single: Python; SciPy
```

In addition to what’s in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
```

We use the following imports.

```{code-cell} ipython3
import numpy as np
import quantecon as qe
```

## Overview

[SciPy](https://scipy.org/) builds on top of NumPy to provide common tools for scientific programming such as

* [linear algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)
* [numerical integration](https://docs.scipy.org/doc/scipy/reference/integrate.html)
* [interpolation](https://docs.scipy.org/doc/scipy/reference/interpolate.html)
* [optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
* [distributions and random number generation](https://docs.scipy.org/doc/scipy/reference/stats.html)
* [signal processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
* etc., etc

Like NumPy, SciPy is stable, mature and widely used.

Many SciPy routines are thin wrappers around industry-standard Fortran libraries such as [LAPACK](https://en.wikipedia.org/wiki/LAPACK), [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms), etc.

It's not really necessary to "learn" SciPy as a whole.

A more common approach is to get some idea of what's in the library and then look up [documentation](https://docs.scipy.org/doc/scipy/reference/index.html) as required.

In this lecture, we aim only to highlight some useful parts of the package.

## {index}`SciPy <single: SciPy>` versus {index}`NumPy <single: NumPy>`

SciPy is a package that contains various tools that are built on top of NumPy, using its array data type and related functionality.

````{note} 
In older versions of SciPy (`scipy < 0.15.1`), importing the package would also import NumPy symbols into the global namespace, as can be seen from this excerpt the SciPy initialization file:

```python
from numpy import *
from numpy.random import rand, randn
from numpy.fft import fft, ifft
from numpy.lib.scimath import *
```

However, it is better practice to use NumPy functionality explicitly.

```python
import numpy as np

a = np.identity(3)
```

More recent versions of SciPy (1.15+) no longer automatically import NumPy symbols.
````

What is useful in SciPy is the functionality in its sub-packages

* `scipy.optimize`, `scipy.integrate`, `scipy.stats`, etc.

Let's explore some of the major sub-packages.

## Statistics

```{index} single: SciPy; Statistics
```

The `scipy.stats` subpackage supplies

* numerous random variable objects (densities, cumulative distributions, random sampling, etc.)
* some estimation procedures
* some statistical tests

### Random Variables and Distributions

Recall that `numpy.random` provides functions for generating random variables

```{code-cell} python3
np.random.beta(5, 5, size=3)
```

This generates a draw from the distribution with the density function below when `a, b = 5, 5`

```{math}
:label: betadist2

f(x; a, b) = \frac{x^{(a - 1)} (1 - x)^{(b - 1)}}
    {\int_0^1 u^{(a - 1)} (1 - u)^{(b - 1)} du}
    \qquad (0 \leq x \leq 1)
```

Sometimes we need access to the density itself, or the cdf, the quantiles, etc.

For this, we can use `scipy.stats`, which provides all of this functionality as well as random number generation in a single consistent interface.

Here's an example of usage

```{code-cell} ipython
from scipy.stats import beta
import matplotlib.pyplot as plt

q = beta(5, 5)      # Beta(a, b), with a = b = 5
obs = q.rvs(2000)   # 2000 observations
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, q.pdf(grid), 'k-', linewidth=2)
plt.show()
```

The object `q` that represents the distribution has additional useful methods, including

```{code-cell} python3
q.cdf(0.4)      # Cumulative distribution function
```

```{code-cell} python3
q.ppf(0.8)      # Quantile (inverse cdf) function
```

```{code-cell} python3
q.mean()
```

The general syntax for creating these objects that represent distributions (of type `rv_frozen`) is

> `name = scipy.stats.distribution_name(shape_parameters, loc=c, scale=d)`

Here `distribution_name` is one of the distribution names in [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html).

The `loc` and `scale` parameters transform the original random variable
$X$ into $Y = c + d X$.

### Alternative Syntax

There is an alternative way of calling the methods described above.

For example, the code that generates the figure above can be replaced by

```{code-cell} python3
obs = beta.rvs(5, 5, size=2000)
grid = np.linspace(0.01, 0.99, 100)

fig, ax = plt.subplots()
ax.hist(obs, bins=40, density=True)
ax.plot(grid, beta.pdf(grid, 5, 5), 'k-', linewidth=2)
plt.show()
```

### Other Goodies in scipy.stats

There are a variety of statistical functions in `scipy.stats`.

For example, `scipy.stats.linregress` implements simple linear regression

```{code-cell} python3
from scipy.stats import linregress

x = np.random.randn(200)
y = 2 * x + 0.1 * np.random.randn(200)
gradient, intercept, r_value, p_value, std_err = linregress(x, y)
gradient, intercept
```

To see the full list, consult the [documentation](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-functions-scipy-stats).

## Roots and Fixed Points

A **root** or **zero** of a real function $f$ on $[a,b]$ is an $x \in [a, b]$ such that $f(x)=0$.

For example, if we plot the function

```{math}
:label: root_f

f(x) = \sin(4 (x - 1/4)) + x + x^{20} - 1
```

with $x \in [0,1]$ we get

```{code-cell} python3
f = lambda x: np.sin(4 * (x - 1/4)) + x + x**20 - 1
x = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f(x)$')
ax.axhline(ls='--', c='k')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)
ax.legend(fontsize=12)
plt.show()
```

The unique root is approximately 0.408.

Let's consider some numerical techniques for finding roots.

### {index}`Bisection <single: Bisection>`

```{index} single: SciPy; Bisection
```

One of the most common algorithms for numerical root-finding is *bisection*.

To understand the idea, recall the well-known game where

* Player A thinks of a secret number between 1 and 100
* Player B asks if it's less than 50
    * If yes, B asks if it's less than 25
    * If no, B asks if it's less than 75

And so on.

This is bisection.

Here's a simplistic implementation of the algorithm in Python.

It works for all sufficiently well behaved increasing continuous functions with $f(a) < 0 < f(b)$

(bisect_func)=
```{code-cell} python3
def bisect(f, a, b, tol=10e-5):
    """
    Implements the bisection root finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b

    while upper - lower > tol:
        middle = 0.5 * (upper + lower)
        if f(middle) > 0:   # root is between lower and middle
            lower, upper = lower, middle
        else:               # root is between middle and upper
            lower, upper = middle, upper

    return 0.5 * (upper + lower)
```

Let's test it using the function $f$ defined in {eq}`root_f`

```{code-cell} python3
bisect(f, 0, 1)
```

Not surprisingly, SciPy provides its own bisection function.

Let's test it using the same function $f$ defined in {eq}`root_f`

```{code-cell} python3
from scipy.optimize import bisect

bisect(f, 0, 1)
```

### The {index}`Newton-Raphson Method <single: Newton-Raphson Method>`

```{index} single: SciPy; Newton-Raphson Method
```

Another very common root-finding algorithm is the [Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method).

In SciPy this algorithm is implemented by `scipy.optimize.newton`.

Unlike bisection, the Newton-Raphson method uses local slope information in an attempt to increase the speed of convergence.

Let's investigate this using the same function $f$ defined above.

With a suitable initial condition for the search we get convergence:

```{code-cell} python3
from scipy.optimize import newton

newton(f, 0.2)   # Start the search at initial condition x = 0.2
```

But other initial conditions lead to failure of convergence:

```{code-cell} python3
newton(f, 0.7)   # Start the search at x = 0.7 instead
```

### Hybrid Methods

A general principle of numerical methods is as follows:

* If you have specific knowledge about a given problem, you might be able to exploit it to generate efficiency.
* If not, then the choice of algorithm involves a trade-off between speed and robustness.

In practice, most default algorithms for root-finding, optimization and fixed points use *hybrid* methods.

These methods typically combine a fast method with a robust method in the following manner:

1. Attempt to use a fast method
1. Check diagnostics
1. If diagnostics are bad, then switch to a more robust algorithm

In `scipy.optimize`, the function `brentq` is such a hybrid method and a good default

```{code-cell} python3
from scipy.optimize import brentq

brentq(f, 0, 1)
```

Here the correct solution is found and the speed is better than bisection:

```{code-cell} ipython
with qe.Timer(unit="milliseconds"):
    brentq(f, 0, 1)
```

```{code-cell} ipython
with qe.Timer(unit="milliseconds"):
    bisect(f, 0, 1)
```

### Multivariate Root-Finding

```{index} single: SciPy; Multivariate Root-Finding
```

Use `scipy.optimize.fsolve`, a wrapper for a hybrid method in MINPACK.

See the [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html) for details.

### Fixed Points

A **fixed point** of a real function $f$ on $[a,b]$ is an $x \in [a, b]$ such that $f(x)=x$.

```{index} single: SciPy; Fixed Points
```

SciPy has a function for finding (scalar) fixed points too

```{code-cell} python3
from scipy.optimize import fixed_point

fixed_point(lambda x: x**2, 10.0)  # 10.0 is an initial guess
```

If you don't get good results, you can always switch back to the `brentq` root finder, since
the fixed point of a function $f$ is the root of $g(x) := x - f(x)$.

## {index}`Optimization <single: Optimization>`

```{index} single: SciPy; Optimization
```

Most numerical packages provide only functions for *minimization*.

Maximization can be performed by recalling that the maximizer of a function $f$ on domain $D$ is
the minimizer of $-f$ on $D$.

Minimization is closely related to root-finding: For smooth functions, interior optima correspond to roots of the first derivative.

The speed/robustness trade-off described above is present with numerical optimization too.

Unless you have some prior information you can exploit, it's usually best to use hybrid methods.

For constrained, univariate (i.e., scalar) minimization, a good hybrid option is `fminbound`

```{code-cell} python3
from scipy.optimize import fminbound

fminbound(lambda x: x**2, -1, 2)  # Search in [-1, 2]
```

### Multivariate Optimization

```{index} single: Optimization; Multivariate
```

Multivariate local optimizers include `minimize`, `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, and `fmin_ncg`.

Constrained multivariate local optimizers include `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`.

See the [documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html) for details.

## {index}`Integration <single: Integration>`

```{index} single: SciPy; Integration
```

Most numerical integration methods work by computing the integral of an approximating polynomial.

The resulting error depends on how well the polynomial fits the integrand, which in turn depends on how "regular" the integrand is.

In SciPy, the relevant module for numerical integration is `scipy.integrate`.

A good default for univariate integration is `quad`

```{code-cell} python3
from scipy.integrate import quad

integral, error = quad(lambda x: x**2, 0, 1)
integral
```

In fact, `quad` is an interface to a very standard numerical integration routine in the Fortran library QUADPACK.

It uses [Clenshaw-Curtis quadrature](https://en.wikipedia.org/wiki/Clenshaw-Curtis_quadrature),  based on expansion in terms of Chebychev polynomials.

There are other options for univariate integration---a useful one is `fixed_quad`, which is fast and hence works well inside `for` loops.

There are also functions for multivariate integration.

See the [documentation](https://docs.scipy.org/doc/scipy/reference/integrate.html) for more details.

## {index}`Linear Algebra <single: Linear Algebra>`

```{index} single: SciPy; Linear Algebra
```

We saw that NumPy provides a module for linear algebra called `linalg`.

SciPy also provides a module for linear algebra with the same name.

The latter is not an exact superset of the former, but overall it has more functionality.

We leave you to investigate the [set of available routines](https://docs.scipy.org/doc/scipy/reference/linalg.html).

## Exercises

The first few exercises concern pricing a European call option under the
assumption of risk neutrality.  The price satisfies

$$
P = \beta^n \mathbb E \max\{ S_n - K, 0 \}
$$

where

1. $\beta$ is a discount factor,
2. $n$ is the expiry date,
2. $K$ is the strike price and
3. $\{S_t\}$ is the price of the underlying asset at each time $t$.

For example, if the call option is to buy stock in Amazon at strike price $K$, the owner has the right (but not the obligation) to buy 1 share in Amazon at price $K$ after $n$ days.  

The payoff is therefore $\max\{S_n - K, 0\}$

The price is the expectation of the payoff, discounted to current value.


```{exercise-start}
:label: sp_ex01
```

Suppose that $S_n$ has the [log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution) distribution with parameters $\mu$ and $\sigma$.  Let $f$ denote the density of this distribution.  Then

$$
P = \beta^n \int_0^\infty \max\{x - K, 0\} f(x) dx
$$

Plot the function 

$$
g(x) = \beta^n  \max\{x - K, 0\} f(x)
$$ 

over the interval $[0, 400]$ when `μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40`.

```{hint}
:class: dropdown

From `scipy.stats` you can import `lognorm` and then use `lognorm.pdf(x, σ, scale=np.exp(μ))` to get the density $f$.
```

```{exercise-end}
```

```{solution-start} sp_ex01
:class: dropdown
```

Here's one possible solution

```{code-cell} ipython3
from scipy.integrate import quad
from scipy.stats import lognorm

μ, σ, β, n, K = 4, 0.25, 0.99, 10, 40

def g(x):
    return β**n * np.maximum(x - K, 0) * lognorm.pdf(x, σ, scale=np.exp(μ))

x_grid = np.linspace(0, 400, 1000)
y_grid = g(x_grid) 

fig, ax = plt.subplots()
ax.plot(x_grid, y_grid, label="$g$")
ax.legend()
plt.show()
```

```{solution-end}
```

```{exercise}
:label: sp_ex02

In order to get the option price, compute the integral of this function numerically using `quad` from `scipy.integrate`.

```

```{solution-start} sp_ex02
:class: dropdown
```

```{code-cell} ipython3
P, error = quad(g, 0, 1_000)
print(f"The numerical integration based option price is {P:.3f}")
```

```{solution-end}
```

```{exercise}
:label: sp_ex03

Try to get a similar result using Monte Carlo to compute the expectation term in the option price, rather than `quad`.

In particular, use the fact that if $S_n^1, \ldots, S_n^M$ are independent
draws from the lognormal distribution specified above, then, by the law of
large numbers,

$$ \mathbb E \max\{ S_n - K, 0 \} 
    \approx
    \frac{1}{M} \sum_{m=1}^M \max \{S_n^m - K, 0 \}
    $$
    
Set `M = 10_000_000`

```

```{solution-start} sp_ex03
:class: dropdown
```

Here is one solution:

```{code-cell} ipython3
M = 10_000_000
S = np.exp(μ + σ * np.random.randn(M))
return_draws = np.maximum(S - K, 0)
P = β**n * np.mean(return_draws) 
print(f"The Monte Carlo option price is {P:3f}")
```


```{solution-end}
```



```{exercise}
:label: sp_ex1

In {ref}`this lecture <functions>`, we discussed the concept of {ref}`recursive function calls <recursive_functions>`.

Try to write a recursive implementation of the homemade bisection function {ref}`described above <bisect_func>`.

Test it on the function {eq}`root_f`.
```

```{solution-start} sp_ex1
:class: dropdown
```

Here's a reasonable solution:


```{code-cell} python3
def bisect(f, a, b, tol=10e-5):
    """
    Implements the bisection root-finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b
    if upper - lower < tol:
        return 0.5 * (upper + lower)
    else:
        middle = 0.5 * (upper + lower)
        print(f'Current mid point = {middle}')
        if f(middle) > 0:   # Implies root is between lower and middle
            return bisect(f, lower, middle)
        else:               # Implies root is between middle and upper
            return bisect(f, middle, upper)
```

We can test it as follows

```{code-cell} python3
f = lambda x: np.sin(4 * (x - 0.25)) + x + x**20 - 1
bisect(f, 0, 1)
```

```{solution-end}
```
