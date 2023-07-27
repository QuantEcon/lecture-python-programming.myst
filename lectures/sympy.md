---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"user_expressions": []}

# SymPy

## Overview

Unlike numerical libraries that deal with values, [SymPy](https://www.sympy.org/en/index.html) focuses on manipulating mathematical symbols and expressions directly.

SymPy provides [a wide range of functionalities](https://www.sympy.org/en/features.html) including 

- Symbolic Expression
- Equation Solving
- Simplification
- Calculus
- Matrices
- Discrete Math, etc.

These functions make SymPy a popular open-source alternative to other proprietary symbolic computation software such as Mathematica.

In this lecture, we will explore some of the functionalities of SymPy and demonstrate how to use basic SymPy functions to solve economic models.

## Getting Started

Let’s first import the library and initialize the printer for symbolic output

```{code-cell} ipython3
from sympy import *
from sympy.plotting import plot, plot3d_parametric_line, plot3d
from sympy.solvers.inequalities import reduce_rational_inequalities

import numpy as np

# Enable the best printer available
init_printing()
```

+++ {"user_expressions": []}

## Symbolic algebra

### Symbols

+++ {"user_expressions": []}

Before we start manipulating the symbols, we initialize some symbols to work with

```{code-cell} ipython3
x, y, z = symbols('x y z')
```

Symbols are the basic units for symbolic computation in SymPy.

+++ {"user_expressions": []}

### Expressions

We can now use symbols `x`, `y`, and `z` to build expressions and equations.

Here we build a simple expression first

```{code-cell} ipython3
---
mystnb:
  image:
    width: 10%
---
expr = (x + y) ** 2
expr
```

+++ {"user_expressions": []}

We can expand this expression with the `expand` function

```{code-cell} ipython3
expand_expr = expand(expr)
expand_expr
```

+++ {"user_expressions": []}

and factorize it back to the factored form with the `factor` function

```{code-cell} ipython3
factor(expand_expr)
```

+++ {"user_expressions": []}

We can solve this expression

```{code-cell} ipython3
solve(expr)
```

+++ {"user_expressions": []}

Note this is equivalent to solving the equation

$$
(x + y)^2 = 0 
$$

Another way to solve this equation is to use `solveset`.

`solveset` allows users to define the domain of `x` in the evaluation and output a set as a solution

```{code-cell} ipython3
solveset(expr, x, domain=S.Naturals)
```

```{note}
[Solvers](https://docs.sympy.org/latest/modules/solvers/index.html) is an important module with tools to solve different types of equations. 

There are a variety of solvers available in SymPy depending on the nature of the problem.
```

+++ {"user_expressions": []}

### Equations

SymPy provides several functions to manipulate equations.

Let's develop an equation with the expression we defined before

```{code-cell} ipython3
eq = Eq(expr, 0)
eq
```

+++ {"user_expressions": []}

Solving this equation with respect to x gives the same output as solving the expression directly

```{code-cell} ipython3
solve(eq, x)
```

+++ {"user_expressions": []}

SymPy can easily handle equations with multiple solutions

```{code-cell} ipython3
eq = Eq(expr, 1)
solve(eq, x)
```

+++ {"user_expressions": []}

`solve` function can also combine multiple equations together and solve a system of equations

```{code-cell} ipython3
eq2 = Eq(x, y)
eq2
```

```{code-cell} ipython3
solve([eq, eq2], [x, y])
```

+++ {"user_expressions": []}

We can also solve for the value of $y$ by simply substituting $x$ with $y$

```{code-cell} ipython3
expr_sub = expr.subs(x, y)
expr_sub
```

```{code-cell} ipython3
solve(Eq(expr_sub, 1))
```

+++ {"user_expressions": []}

Below is another example equation with symbol `x` and functions `sin`, `cos`, and `tan` using the `Eq` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
# Create an equation
eq = Eq(cos(x) / (tan(x)/sin(x)), 0)
eq
```

+++ {"user_expressions": []}

Now we simplify this equation using the `simplify` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 10%
---
# Simplify an expression
simplified_expr = simplify(eq)
simplified_expr
```

+++ {"user_expressions": []}

We can solve equations using the `solve` function in SymPy

```{code-cell} ipython3
---
mystnb:
  image:
    width: 5.5%
---
# Solve the equation
sol = solve(eq, x)
sol
```

+++ {"user_expressions": []}

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

+++ {"user_expressions": []}

#### Example: fixed point computation

Fixed point computation is a common problem in economics and finance.

Here we solve the fixed point of the Solow-Swan growth dynamics:

$$
k_{t+1}=s f\left(k_t\right)+(1-\delta) k_t, \quad t=0,1, \ldots
$$

where $k_t$ is the capital stock, $f$ is a production function, $\delta$ is a rate of depreciation.

With $f(k) = Ak^a$, we can show the unique fixed point of the dynamics using pen and paper:

$$
k^*:=\left(\frac{s A}{\delta}\right)^{1 /(1-\alpha)}
$$ 

This can be easily computed in SymPy

```{code-cell} ipython3
A, s, k, α, δ = symbols('A s k α δ')
```

```{code-cell} ipython3
# Define Solow-Swan growth dynamics
solow = Eq(s*A*k**α + (1 - δ) * k, k)
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

+++ {"user_expressions": []}

### Series

Series are widely used in economics and statistics, from asset pricing to the expectation of discrete random variables.

We can construct a simple series of summations using `Sum` function

```{code-cell} ipython3
x, y, i, j = symbols("x y i j")
sum_xy = Sum(Indexed('x',i) * Indexed('y', j), 
            (i, 0, 3),
            (j, 0, 3))
sum_xy
```

+++ {"user_expressions": []}

To evaluate the sum, we can `lamdify` the formula.

The lamdified expression can take vectors as input for $x$ and $y$ and compute the result

```{code-cell} ipython3
sum_xy = lambdify([x, y], sum_xy)
grid = np.arange(0, 4, 1)
sum_xy(grid, grid)
```

+++ {"user_expressions": []}

#### Example: bank deposits

Imagine a bank with $D_0$ as the deposit at time $t$, it loans $(1-r)$ of its deposits and keeps a fraction $r$ as cash reserves.

Its deposits at time is

$$
\sum_{i=0}^\infty (1-r)^i D_0
$$

```{code-cell} ipython3
r, D = symbols('r D')

Dt = Sum('(1 - r)^i * D', (i, 0, oo))
Dt
```

+++ {"user_expressions": []}

We can call `doit` method to evaluate the series

```{code-cell} ipython3
Dt.doit()
```

+++ {"user_expressions": []}

Simplifying the expression above gives

```{code-cell} ipython3
simplify(Dt.doit())
```

+++ {"user_expressions": []}

This is consistent with the solution we provided in our lecture on [geometric series](https://python.quantecon.org/geom_series.html#example-the-money-multiplier-in-fractional-reserve-banking).


#### Example: discrete random variable

We can also use SymPy to compute the expectation of a discrete random variable.

Let's define a discrete random variable $X$ with [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution):

$$
f(x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots
$$

```{code-cell} ipython3
λ = symbols('lambda')

# We refine the symbol x to positive integers
x = Symbol('x', integer=True, positive=True)
pmf = λ**x * exp(-λ)/factorial(x)
pmf
```

We can verify if the distribution follows the fundamental property of probability distribution:

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

```{note}

SymPy offers a statistics module called [`Stats`](https://docs.sympy.org/latest/modules/stats.html).

`Stats` module offers built-in distributions and functions on probability distributions.

The operation above can also be computed using `Stats` module.
```

+++ {"user_expressions": []}

## Symbolic Calculus

SymPy allows us to perform various calculus operations, such as differentiation and integration.


### Limits

We can compute limits for a given function using the `limit` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 3%
---
# Define a function
f = x ** 2 / (x - 1)

# Compute the limit
lim = limit(f, x, 0)
lim
```

+++ {"user_expressions": []}

### Derivatives

We can differentiate any SymPy expression using `diff(func, var)`

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
---
# Differentiate a function
df = diff(f, x)
df
```

+++ {"user_expressions": []}

### Integrals

We can compute definite and indefinite integrals using `integrate` function

```{code-cell} ipython3
---
mystnb:
  image:
    width: 12%
---
# Calculate the indefinite integral
indef_int = integrate(df, x)
indef_int
```

Let's use this function to compute the moment-generating function of [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution) with the probability density function:

$$
f(x) = \lambda e^{-\lambda x}, \quad x \ge 0
$$

```{code-cell} ipython3
λ, x = symbols('lambda x')
pdf = λ * exp(-λ*x)
pdf
```

```{code-cell} ipython3
t = symbols('t')
moment_n = integrate(exp(t * x) * pdf, (x, 0, oo))
simplify(moment_n)
```

Using `integrate` function, we can derive the cumulative density function of the exponential distribution with $\lambda = 0.5$

```{code-cell} ipython3
lamda_pdf = pdf.subs(λ, 1/2)
lamda_pdf
```

```{code-cell} ipython3
integrate(lamda_pdf, (x, 0, 4))
```

+++ {"user_expressions": []}

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
              xlabel='', ylabel='', legend = True, show = False)
plot_f[0].label = 'f(x)'
df = diff(f)
plot_df = plot(df, (x, -10, 10), legend = True, show = False)
plot_df[0].label = 'f\'(x)'
plot_f.append(plot_df[0])
plot_f.show()
```

It also supports plotting implicit functions and visualizing inequalities

```{code-cell} ipython3
p = plot_implicit(Eq((1/x + 1/y)**2, 1))
```

```{code-cell} ipython3
p = plot_implicit(And(2*x + 5*y <= 30, 4*x + 2*y >= 20), (x, -1, 10), (y, -10, 10))
```

It also supports visualizations in three-dimensional space

```{code-cell} ipython3
p = plot3d(cos(2*x + y))
```

+++ {"user_expressions": []}

## Applications

In this section, we apply SymPy to an economic model that explores the wage gap between college and high school graduates. 

### Defining the Symbols

The first step in symbolic computation is to define the symbols that represent the variables in our equations. In our model, we have the following variables:

 * $R$ The gross rate of return on a one-period bond

 * $t = 0, 1, 2, \ldots T$ Denote the years that a person either works or attends college
 
 * $T$ denotes the last period that a person  works
 
 * $w$ The wage of a high school graduate
 
 * $g$ The growth rate of wages

 * $D$ The upfront monetary costs of going to college
 
 * $\phi$ The wage gap, defined as the ratio of the wage of a college graduate to the wage of a high school graduate
 
Let's define these symbols using sympy

```{code-cell} ipython3
# Define the symbols
w, R, g, D, phi = symbols('w R g D phi')
```

+++ {"user_expressions": []}

### Defining the Present Value Equations
We calculate the present value of the earnings of a high school graduate and a college graduate. For a high school graduate, we sum the discounted wages from year 1 to $T = 4$. For a college graduate, we sum the discounted wages from year 5 to $T = 4$, considering the cost of college $D$ paid upfront

+++ {"user_expressions": []}

$$
PV_{\text{{highschool}}} = \frac{w}{R} + \frac{w*(1 + g)}{R^2} + \frac{w*(1 + g)^2}{R^3} + \frac{w*(1 + g)^3}{R^4}
$$

$$
PV_{\text{{college}}} = D + \frac{\phi * w * (1 + g)^4}{R} + \frac{\phi * w * (1 + g)^5}{R^2} + \frac{\phi * w * (1 + g)^6}{R^3} + \frac{\phi * w * (1 + g)^7}{R^4}
$$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
# Define the present value equations
PV_highschool = w/R + w*(1 + g)/R**2 + w*(1 + g)**2/R**3 + w*(1 + g)**3/R**4
PV_college = D + phi*w*(1 + g)**4/R + phi*w*(1 + g)**5 / \
    R**2 + phi*w*(1 + g)**6/R**3 + phi*w*(1 + g)**7/R**4
```

+++ {"user_expressions": []}

### Defining the Indifference Equation
The indifference equation represents the condition that a worker is indifferent between going to college or not. This is given by the equation $PV_h$ = $PV_c$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 17.5%
---
# Define the indifference equation
indifference_eq = Eq(PV_highschool, PV_college)
```

+++ {"user_expressions": []}

We can now solve the indifference equation for the wage gap $\phi$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 65%
---
# Solve for phi
solution = solve(indifference_eq, phi)

# Simplify the solution
solution = simplify(solution[0])
solution
```

+++ {"user_expressions": []}

To compute a numerical value for $\phi$, we replace symbols $w$, $R$, $g$, and $D$ with specific numbers. 

Here we use $w = 1$, $R = 1.05$, $g = 0.02$, and $D = 0.5$. 

Substituting them into the equation, we find a specific value for $\phi$

```{code-cell} ipython3
---
mystnb:
  image:
    width: 20%
---
# Substitute specific values
solution_num = solution.subs({w: 1, R: 1.05, g: 0.02, D: 0.5})
solution_num
```

+++ {"user_expressions": []}

The wage of a college graduate is approximately 0.797 times the wage of a high school graduate.

+++ {"user_expressions": []}

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

+++

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

Maximum likelihood estimation (MLE) is a method to estimate the parameters of a statistical model. 

It usually involves maximizing a log-likelihood function and solving the first-order derivative.

In this exercise, we use SymPy to compute the MLE of a binomial distribution.
```

```{solution-start} sympy_ex2
:class: dropdown
```

First, we define the binomial distribution

```{code-cell} ipython3
n, x, θ = symbols('n x θ')

binomial_factor = (factorial(n))/ (factorial(x) *factorial(n-r))
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

```{exercise}
:label: sympy_ex3

Imagine a pure exchange economy with two consumers ($a$ and $b$)and two goods ($x$ and $y$).

Assume that the utility functions of the consumers are given by

$$
u_a(x, y) = x^{\alpha} y^{1-\alpha}
$$

$$
u_b(x, y) = (1 - x)^{\beta} (1 - y)^{1-\beta}
$$

where $\alpha, \beta \in (0, 1)$.

In this exercise, we are interested in the Pareto optimal allocation of goods $x$ and $y$.

Note that A point is Pareto efficient when the allocation is optimal for one person given the allocation for the other person.

In terms of marginal utility:

$$
\frac{\frac{\partial u_a}{\partial x}}{\frac{\partial u_a}{\partial y}}=\frac{\frac{\partial u_b}{\partial x}}{\frac{\partial u_b}{\partial y}}
$$

Compute the Pareto optimal allocation of the economy with $\alpha = \beta = 0.5$ using SymPy
```

```{solution-start} sympy_ex3
:class: dropdown
```

+++

Here is one solution

```{code-cell} ipython3
# Define symbols and utility functions
x, y, α, β = symbols('x, y, α, β')
a = x**α * y**(1-α)
b = (1 - x)**β * (1 - y)**(1 - β)
```

```{code-cell} ipython3
a
```

```{code-cell} ipython3
b
```

```{code-cell} ipython3
# A point is Pareto efficient when the allocation is optimal 
# for one person given the allocation for the other person

pareto = Eq(diff(a, x)/diff(a, y), diff(b, x)/diff(b, y))
pareto
```

```{code-cell} ipython3
# Solve the equation
sol = solve(pareto, y)[0]
sol
```

```{code-cell} ipython3
# Substitute α = 0.5 and β = 0.5
sol.subs({α: 0.5, β: 0.5})
```

We can visualize contract curves with Cobb-Douglas utility functions

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
          {α: 0.9, β: 0.1},]

p = plot(xlabel='x', ylabel='y', show=False)

for param in params:
    p_add = plot(sol.subs(param), (x, 0, 1), show=False)
    p.append(p_add[0])
p.show()
```

Extra tasks: 

- Can you think of a way to draw the same graph using `numpy`? 
- How difficult will it be to write a `numpy` implementation?

+++

```{solution-end}
```
