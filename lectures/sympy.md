---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# {index}`SymPy <single: SymPy>`

```{index} single: Python; SymPy
```

## Overview

Unlike numerical libraries that deal with values, [SymPy](https://www.sympy.org/en/index.html) focuses on manipulating mathematical symbols and expressions directly.

SymPy provides [a wide range of features](https://www.sympy.org/en/features.html) including 

- symbolic expression
- equation solving
- simplification
- calculus
- matrices
- discrete math, etc.

These functions make SymPy a popular open-source alternative to other proprietary symbolic computational software such as Mathematica.

In this lecture, we will explore some of the functionality of SymPy and demonstrate how to use basic SymPy functions to solve economic models.

## Getting Started

Let’s first import the library and initialize the printer for symbolic output

```{code-cell} ipython3
from sympy import *
from sympy.plotting import plot, plot3d_parametric_line, plot3d
from sympy.solvers.inequalities import reduce_rational_inequalities
from sympy.stats import Poisson, Exponential, Binomial, density, moment, E, cdf

import numpy as np
import matplotlib.pyplot as plt

# Enable the mathjax printer
init_printing(use_latex='mathjax')
```

## Symbolic algebra

### Symbols

First we initialize some symbols to work with

```{code-cell} ipython3
x, y, z = symbols('x y z')
```

Symbols are the basic units for symbolic computation in SymPy.

### Expressions

We can now use symbols `x`, `y`, and `z` to build expressions and equations.

Here we build a simple expression first

```{code-cell} ipython3
expr = (x+y) ** 2
expr
```

We can expand this expression with the `expand` function

```{code-cell} ipython3
expand_expr = expand(expr)
expand_expr
```

and factorize it back to the factored form with the `factor` function

```{code-cell} ipython3
factor(expand_expr)
```

We can solve this expression

```{code-cell} ipython3
solve(expr)
```

Note this is equivalent to solving the following equation for `x`

$$
(x + y)^2 = 0 
$$

```{note}
[Solvers](https://docs.sympy.org/latest/modules/solvers/index.html) is an important module with tools to solve different types of equations. 

There are a variety of solvers available in SymPy depending on the nature of the problem.
```

### Equations

SymPy provides several functions to manipulate equations.

Let's develop an equation with the expression we defined before

```{code-cell} ipython3
eq = Eq(expr, 0)
eq
```

Solving this equation with respect to $x$ gives the same output as solving the expression directly

```{code-cell} ipython3
solve(eq, x)
```

SymPy can handle equations with multiple solutions

```{code-cell} ipython3
eq = Eq(expr, 1)
solve(eq, x)
```

`solve` function can also combine multiple equations together and solve a system of equations

```{code-cell} ipython3
eq2 = Eq(x, y)
eq2
```

```{code-cell} ipython3
solve([eq, eq2], [x, y])
```

We can also solve for the value of $y$ by simply substituting $x$ with $y$

```{code-cell} ipython3
expr_sub = expr.subs(x, y)
expr_sub
```

```{code-cell} ipython3
solve(Eq(expr_sub, 1))
```

Below is another example equation with the symbol `x` and functions `sin`, `cos`, and `tan` using the `Eq` function

```{code-cell} ipython3
# Create an equation
eq = Eq(cos(x) / (tan(x)/sin(x)), 0)
eq
```

Now we simplify this equation using the `simplify` function

```{code-cell} ipython3
# Simplify an expression
simplified_expr = simplify(eq)
simplified_expr
```

Again, we use the `solve` function to solve this equation

```{code-cell} ipython3
# Solve the equation
sol = solve(eq, x)
sol
```

SymPy can also handle more complex equations involving trigonometry and complex numbers.

We demonstrate this using [Euler's formula](https://en.wikipedia.org/wiki/Euler%27s_formula)

```{code-cell} ipython3
# 'I' represents the imaginary number i 
euler = cos(x) + I*sin(x)
euler
```

```{code-cell} ipython3
simplify(euler)
```

If you are interested, we encourage you to read the lecture on [trigonometry and complex numbers](https://python.quantecon.org/complex_and_trig.html).

#### Example: fixed point computation

Fixed point computation is frequently used in economics and finance.

Here we solve the fixed point of the Solow-Swan growth dynamics:

$$
k_{t+1}=s f\left(k_t\right)+(1-\delta) k_t, \quad t=0,1, \ldots
$$

where $k_t$ is the capital stock, $f$ is a production function, $\delta$ is a rate of depreciation.

We are interested in calculating the fixed point of this dynamics, i.e., the value of $k$ such that $k_{t+1} = k_t$.

With $f(k) = Ak^\alpha$, we can show the unique fixed point of the dynamics $k^*$ using pen and paper:

$$
k^*:=\left(\frac{s A}{\delta}\right)^{1 /(1-\alpha)}
$$ 

This can be easily computed in SymPy

```{code-cell} ipython3
A, s, k, α, δ = symbols('A s k^* α δ')
```

Now we solve for the fixed point $k^*$

$$
k^* = sA(k^*)^{\alpha}+(1-\delta) k^*
$$

```{code-cell} ipython3
# Define Solow-Swan growth dynamics
solow = Eq(s*A*k**α + (1-δ)*k, k)
solow
```

```{code-cell} ipython3
solve(solow, k)
```

### Inequalities and logic

SymPy also allows users to define inequalities and set operators and provides a wide range of [operations](https://docs.sympy.org/latest/modules/solvers/inequalities.html).

```{code-cell} ipython3
reduce_inequalities([2*x + 5*y <= 30, 4*x + 2*y <= 20], [x])
```

```{code-cell} ipython3
And(2*x + 5*y <= 30, x > 0)
```

### Series

Series are widely used in economics and statistics, from asset pricing to the expectation of discrete random variables.

We can construct a simple series of summations using `Sum` function and `Indexed` symbols

```{code-cell} ipython3
x, y, i, j = symbols("x y i j")
sum_xy = Sum(Indexed('x', i)*Indexed('y', j), 
            (i, 0, 3),
            (j, 0, 3))
sum_xy
```

To evaluate the sum, we can [`lambdify`](https://docs.sympy.org/latest/modules/utilities/lambdify.html#sympy.utilities.lambdify.lambdify) the formula.

The lambdified expression can take numeric values as input for $x$ and $y$ and compute the result

```{code-cell} ipython3
sum_xy = lambdify([x, y], sum_xy)
grid = np.arange(0, 4, 1)
sum_xy(grid, grid)
```

#### Example: bank deposits

Imagine a bank with $D_0$ as the deposit at time $t$.

It loans $(1-r)$ of its deposits and keeps a fraction $r$ as cash reserves.

Its deposits over an infinite time horizon can be written as

$$
\sum_{i=0}^\infty (1-r)^i D_0
$$

Let's compute the deposits at time $t$

```{code-cell} ipython3
D = symbols('D_0')
r = Symbol('r', positive=True)
Dt = Sum('(1 - r)^i * D_0', (i, 0, oo))
Dt
```

We can call the `doit` method to evaluate the series

```{code-cell} ipython3
Dt.doit()
```

Simplifying the expression above gives

```{code-cell} ipython3
simplify(Dt.doit())
```

This is consistent with the solution in the lecture on [geometric series](https://intro.quantecon.org/geom_series.html#example-the-money-multiplier-in-fractional-reserve-banking).


#### Example: discrete random variable

In the following example, we compute the expectation of a discrete random variable.

Let's define a discrete random variable $X$ following a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution):

$$
f(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots
$$

```{code-cell} ipython3
λ = symbols('lambda')

# We refine the symbol x to positive integers
x = Symbol('x', integer=True, positive=True)
pmf = λ**x * exp(-λ) / factorial(x)
pmf
```

We can verify if the sum of probabilities for all possible values equals $1$:

$$
\sum_{x=0}^{\infty} f(x) = 1
$$

```{code-cell} ipython3
sum_pmf = Sum(pmf, (x, 0, oo))
sum_pmf.doit()
```

The expectation of the distribution is:

$$
E(X) = \sum_{x=0}^{\infty} x f(x) 
$$

```{code-cell} ipython3
fx = Sum(x*pmf, (x, 0, oo))
fx.doit()
```

SymPy includes a statistics submodule called [`Stats`](https://docs.sympy.org/latest/modules/stats.html).

`Stats` offers built-in distributions and functions on probability distributions.

The computation above can also be condensed into one line using the expectation function `E` in the `Stats` module

```{code-cell} ipython3
λ = Symbol("λ", positive = True)

# Using sympy.stats.Poisson() method
X = Poisson("x", λ)
E(X)
```

## Symbolic Calculus

SymPy allows us to perform various calculus operations, such as limits, differentiation, and integration.


### Limits

We can compute limits for a given expression using the `limit` function

```{code-cell} ipython3
# Define an expression
f = x**2 / (x-1)

# Compute the limit
lim = limit(f, x, 0)
lim
```

### Derivatives

We can differentiate any SymPy expression using the `diff` function

```{code-cell} ipython3
# Differentiate a function with respect to x
df = diff(f, x)
df
```

### Integrals

We can compute definite and indefinite integrals using the `integrate` function

```{code-cell} ipython3
# Calculate the indefinite integral
indef_int = integrate(df, x)
indef_int
```

Let's use this function to compute the moment-generating function of [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) with the probability density function:

$$
f(x) = \lambda e^{-\lambda x}, \quad x \ge 0
$$

```{code-cell} ipython3
λ = Symbol('lambda', positive=True)
x = Symbol('x', positive=True)
pdf = λ * exp(-λ*x)
pdf
```

```{code-cell} ipython3
t = Symbol('t', positive=True)
moment_t = integrate(exp(t*x) * pdf, (x, 0, oo))
simplify(moment_t)
```

Note that we can also use `Stats` module to compute the moment

```{code-cell} ipython3
X = Exponential(x, λ)
```

```{code-cell} ipython3
moment(X, 1)
```

```{code-cell} ipython3
E(X**t)
```

Using the `integrate` function, we can derive the cumulative density function of the exponential distribution with $\lambda = 0.5$

```{code-cell} ipython3
λ_pdf = pdf.subs(λ, 1/2)
λ_pdf
```

```{code-cell} ipython3
integrate(λ_pdf, (x, 0, 4))
```

Using `cdf` in `Stats` module gives the same solution

```{code-cell} ipython3
cdf(X, 1/2)
```

```{code-cell} ipython3
# Plug in a value for z 
λ_cdf = cdf(X, 1/2)(4)
λ_cdf
```

```{code-cell} ipython3
# Substitute λ
λ_cdf.subs({λ: 1/2})
```

## Plotting

SymPy provides a powerful plotting feature. 

First we plot a simple function using the `plot` function

```{code-cell} ipython3
f = sin(2 * sin(2 * sin(2 * sin(x))))
p = plot(f, (x, -10, 10), show=False)
p.title = 'A Simple Plot'
p.show()
```

Similar to Matplotlib, SymPy provides an interface to customize the graph

```{code-cell} ipython3
plot_f = plot(f, (x, -10, 10), 
              xlabel='', ylabel='', 
              legend = True, show = False)
plot_f[0].label = 'f(x)'
df = diff(f)
plot_df = plot(df, (x, -10, 10), 
            legend = True, show = False)
plot_df[0].label = 'f\'(x)'
plot_f.append(plot_df[0])
plot_f.show()
```

It also supports plotting implicit functions and visualizing inequalities

```{code-cell} ipython3
p = plot_implicit(Eq((1/x + 1/y)**2, 1))
```

```{code-cell} ipython3
p = plot_implicit(And(2*x + 5*y <= 30, 4*x + 2*y >= 20),
                     (x, -1, 10), (y, -10, 10))
```

and visualizations in three-dimensional space

```{code-cell} ipython3
p = plot3d(cos(2*x + y), zlabel='')
```

## Application: Two-person Exchange Economy

Imagine a pure exchange economy with two people ($a$ and $b$) and two goods recorded as proportions ($x$ and $y$).

They can trade goods with each other according to their preferences.

Assume that the utility functions of the consumers are given by

$$
u_a(x, y) = x^{\alpha} y^{1-\alpha}
$$

$$
u_b(x, y) = (1 - x)^{\beta} (1 - y)^{1-\beta}
$$

where $\alpha, \beta \in (0, 1)$.

First we define the symbols and utility functions

```{code-cell} ipython3
# Define symbols and utility functions
x, y, α, β = symbols('x, y, α, β')
u_a = x**α * y**(1-α)
u_b = (1 - x)**β * (1 - y)**(1 - β)
```

```{code-cell} ipython3
u_a
```

```{code-cell} ipython3
u_b
```

We are interested in the Pareto optimal allocation of goods $x$ and $y$.

Note that a point is Pareto efficient when the allocation is optimal for one person given the allocation for the other person.

In terms of marginal utility:

$$
\frac{\frac{\partial u_a}{\partial x}}{\frac{\partial u_a}{\partial y}} = \frac{\frac{\partial u_b}{\partial x}}{\frac{\partial u_b}{\partial y}}
$$

```{code-cell} ipython3
# A point is Pareto efficient when the allocation is optimal 
# for one person given the allocation for the other person

pareto = Eq(diff(u_a, x)/diff(u_a, y), 
            diff(u_b, x)/diff(u_b, y))
pareto
```

```{code-cell} ipython3
# Solve the equation
sol = solve(pareto, y)[0]
sol
```

Let's compute the Pareto optimal allocations of the economy (contract curves) with $\alpha = \beta = 0.5$ using SymPy

```{code-cell} ipython3
# Substitute α = 0.5 and β = 0.5
sol.subs({α: 0.5, β: 0.5})
```

We can use this result to visualize more contract curves under different parameters

```{code-cell} ipython3
# Plot a range of αs and βs
params = [{α: 0.5, β: 0.5}, 
          {α: 0.1, β: 0.9},
          {α: 0.1, β: 0.8},
          {α: 0.8, β: 0.9},
          {α: 0.4, β: 0.8}, 
          {α: 0.8, β: 0.1},
          {α: 0.9, β: 0.8},
          {α: 0.8, β: 0.4},
          {α: 0.9, β: 0.1}]

p = plot(xlabel='x', ylabel='y', show=False)

for param in params:
    p_add = plot(sol.subs(param), (x, 0, 1), 
                 show=False)
    p.append(p_add[0])
p.show()
```

We invite you to play with the parameters and see how the contract curves change and think about the following two questions:

- Can you think of a way to draw the same graph using `numpy`? 
- How difficult will it be to write a `numpy` implementation?

## Exercises

```{exercise}
:label: sympy_ex1

L'Hôpital's rule states that for two functions $f(x)$ and $g(x)$, if $\lim_{x \to a} f(x) = \lim_{x \to a} g(x) = 0$ or $\pm \infty$, then

$$
\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}
$$

Use SymPy to verify L'Hôpital's rule for the following functions

$$
f(x) = \frac{y^x - 1}{x}
$$

as $x$ approaches to $0$
```

```{solution-start} sympy_ex1
:class: dropdown
```

Let's define the function first

```{code-cell} ipython3
f_upper = y**x - 1
f_lower = x
f = f_upper/f_lower
f
```

Sympy is smart enough to solve this limit

```{code-cell} ipython3
lim = limit(f, x, 0)
lim
```

We compare the result suggested by L'Hôpital's rule

```{code-cell} ipython3
lim = limit(diff(f_upper, x)/
            diff(f_lower, x), x, 0)
lim
```

```{solution-end}
```

```{exercise}
:label: sympy_ex2

[Maximum likelihood estimation (MLE)](https://python.quantecon.org/mle.html) is a method to estimate the parameters of a statistical model. 

It usually involves maximizing a log-likelihood function and solving the first-order derivative.

The binomial distribution is given by

$$
f(x; n, θ) = \frac{n!}{x!(n-x)!}θ^x(1-θ)^{n-x}
$$

where $n$ is the number of trials and $x$ is the number of successes.

Assume we observed a series of binary outcomes with $x$ successes out of $n$ trials.

Compute the MLE of $θ$ using SymPy
```

```{solution-start} sympy_ex2
:class: dropdown
```

First, we define the binomial distribution

```{code-cell} ipython3
n, x, θ = symbols('n x θ')

binomial_factor = (factorial(n)) / (factorial(x)*factorial(n-r))
binomial_factor
```

```{code-cell} ipython3
bino_dist = binomial_factor * ((θ**x)*(1-θ)**(n-x))
bino_dist
```

Now we compute the log-likelihood function and solve for the result

```{code-cell} ipython3
log_bino_dist = log(bino_dist)
```

```{code-cell} ipython3
log_bino_diff = simplify(diff(log_bino_dist, θ))
log_bino_diff
```

```{code-cell} ipython3
solve(Eq(log_bino_diff, 0), θ)[0]
```

```{solution-end}
```