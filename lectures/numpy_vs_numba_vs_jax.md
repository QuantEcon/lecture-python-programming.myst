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

(parallel)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# NumPy vs Numba vs JAX

In the preceding lectures, we've discussed three core libraries for scientific
and numerical computing:

* [NumPy](numpy)
* [Numba](numba)
* [JAX](jax_intro)

Which one should we use in any given situation?

This lecture addresses that question, at least partially, by discussing some use cases.

Before getting started, we note that the first two are a natural pair: NumPy and
Numba play well together.

JAX, on the other hand, stands alone.

When considering each approach, we will consider not just efficiency and memory
footprint but also clarity and ease of use.

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon jax
```

We will use the following imports.

```{code-cell} ipython3
import random
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import jax
import jax.numpy as jnp
```

## Vectorized operations

Some operations can be perfectly vectorized --- all loops are easily eliminated
and numerical operations are reduced to calculations on arrays.

In this case, which approach is best?

### Problem Statement

Consider the problem of maximizing a function $f$ of two variables $(x,y)$ over
the square $[-a, a] \times [-a, a]$.

For $f$ and $a$ let's choose

$$
f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
\quad \text{and} \quad
a = 3
$$

Here's a plot of $f$

```{code-cell} ipython3

def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

xgrid = np.linspace(-3, 3, 50)
ygrid = xgrid
x, y = np.meshgrid(xgrid, ygrid)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                f(x, y),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.7,
                linewidth=0.25)
ax.set_zlim(-0.5, 1.0)
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
plt.show()
```

For the sake of this exercise, we're going to use brute force for the
maximization.

1. Evaluate $f$ for all $(x,y)$ in a grid on the square.
1. Return the maximum of observed values.

Just to illustrate the idea, here's a non-vectorized version that uses Python loops.

```{code-cell} ipython3
grid = np.linspace(-3, 3, 50)
m = -np.inf
for x in grid:
    for y in grid:
        z = f(x, y)
        if z > m:
            m = z
```


### NumPy vectorization

If we switch to NumPy-style vectorization we can use a much larger grid and the
code executes relatively quickly.

Here we use `np.meshgrid` to create two-dimensional input grids `x` and `y` such
that `f(x, y)` generates all evaluations on the product grid.

(This strategy dates back to Matlab.)

```{code-cell} ipython3
grid = np.linspace(-3, 3, 3_000)
x, y = np.meshgrid(grid, grid)

with qe.Timer(precision=8):
    np.max(f(x, y))
```

In the vectorized version, all the looping takes place in compiled code.

Moreover, NumPy uses implicit multithreading, so that at least some parallelization occurs.

```{note}
If you have a system monitor such as htop (Linux/Mac) or perfmon
(Windows), then try running this and then observing the load on your CPUs.

(You will probably need to bump up the grid size to see large effects.)

The output typically shows that the operation is successfully distributed across multiple threads.
```

(The parallelization cannot be highly efficient because the binary is compiled
before it sees the size of the arrays `x` and `y`.)


### A Comparison with Numba

Now let's see if we can achieve better performance using Numba with a simple loop.

```{code-cell} ipython3
import numba

@numba.jit
def compute_max_numba(grid):
    m = -np.inf
    for x in grid:
        for y in grid:
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2)
            if z > m:
                m = z
    return m

grid = np.linspace(-3, 3, 3_000)

with qe.Timer(precision=8):
    compute_max_numba(grid)
```

```{code-cell} ipython3
with qe.Timer(precision=8):
    compute_max_numba(grid)
```


Depending on your machine, the Numba version can be a bit slower or a bit faster than NumPy.

On one hand, NumPy combines efficient arithmetic (like Numba) with some multithreading (unlike this Numba code), which provides an advantage.

On the other hand, the Numba routine uses much less memory, since we are only
working with a single one-dimensional grid.


### Parallelized Numba

Now let's try parallelization with Numba using `prange`:

First we parallelize just the outer loop.

```{code-cell} ipython3
@numba.jit(parallel=True)
def compute_max_numba_parallel(grid):
    n = len(grid)
    m = -np.inf
    for i in numba.prange(n):
        for j in range(n):
            x = grid[i]
            y = grid[j]
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2)
            if z > m:
                m = z
    return m

with qe.Timer(precision=8):
    compute_max_numba_parallel(grid)
```


```{code-cell} ipython3
with qe.Timer(precision=8):
    compute_max_numba_parallel(grid)
```

Next we parallelize both loops.

```{code-cell} ipython3
@numba.jit(parallel=True)
def compute_max_numba_parallel_nested(grid):
    n = len(grid)
    m = -np.inf
    for i in numba.prange(n):
        for j in numba.prange(n):
            x = grid[i]
            y = grid[j]
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2)
            if z > m:
                m = z
    return m

with qe.Timer(precision=8):
    compute_max_numba_parallel_nested(grid)
```

```{code-cell} ipython3
with qe.Timer(precision=8):
    compute_max_numba_parallel_nested(grid)
```


Depending on your machine, you might or might not see large benefits from parallelization here.

If you have a small number of cores, the overhead of thread management and synchronization can
overwhelm the benefits of parallel execution. 

For more powerful machines and larger grid sizes, parallelization can generate
large speed gains.



### Vectorized code with JAX

In most ways, vectorization is the same in JAX as it is in NumPy.

But there are also some differences, which we highlight here.

Let's start with the function.


```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)

```

As with NumPy, to get the right shape and the correct nested `for` loop
calculation, we can use a `meshgrid` operation designed for this purpose:

```{code-cell} ipython3
grid = jnp.linspace(-3, 3, 3_000)
x_mesh, y_mesh = np.meshgrid(grid, grid)

with qe.Timer(precision=8):
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
with qe.Timer(precision=8):
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Once compiled, JAX will be significantly faster than NumPy, especially if you are using a GPU.

The compilation overhead is a one-time cost that pays off when the function is called repeatedly.


### JAX plus vmap

There is one problem with both the NumPy code and the JAX code: 

While the flat arrays are low-memory

```{code-cell} ipython3
grid.nbytes 
```

the mesh grids are memory intensive

```{code-cell} ipython3
x_mesh.nbytes + y_mesh.nbytes
```

This extra memory usage can be a big problem in actual research calculations.

Fortunately, JAX admits a different approach using [jax.vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)

#### Version 1

Here's one way we can do this

```{code-cell} ipython3
# Set up f to compute f(x, y) at every x for any given y
f_vec_x = lambda y: f(grid, y)
# Vectorize this operation over all y
f_vec = jax.vmap(f_vec_x)
# Compute result at all y
z_vmap = f_vec(grid)
```

Let's see the timing:

```{code-cell} ipython3
with qe.Timer(precision=8):
    z_vmap_1 = f_vec(grid)
    z_vmap_1.block_until_ready()
```

Let's check we got the right result:


```{code-cell} ipython3
jnp.allclose(z_mesh, z_vmap)
```

The execution time is similar to as the mesh operation but we are using much
less memory.

In addition, `vmap` allows us to break vectorization up into stages, which is
often easier to comprehend than the traditional approach.

This will become more obvious when we tackle larger problems.


#### Version 2

Here's a more generic approach to using `vmap` that we often use in the lectures.

First we vectorize in `y`.

```{code-cell} ipython3
f_vec_y = jax.vmap(f, in_axes=(None, 0))
```

In the line above, `(None, 0)` indicates that we are vectorizing in the second argument, which is `y`.

Next, we vectorize in the first argument, which is `x`.

```{code-cell} ipython3
f_vec = jax.vmap(f_vec_y, in_axes=(0, None))
```

With this construction, we can now call $f$ directly on flat (low memory) arrays.

```{code-cell} ipython3
x, y = grid, grid
with qe.Timer(precision=8):
    z_vmap = f_vec(x, y).block_until_ready()
```

Let's run it again to eliminate compilation time:

```{code-cell} ipython3
with qe.Timer(precision=8):
    z_vmap = f_vec(x, y).block_until_ready()
```



### Summary

In our view, JAX is the winner for vectorized operations.

It dominates NumPy both in terms of speed (via JIT-compilation and parallelization) and memory efficiency (via vmap).

Moreover, the `vmap` approach can sometimes lead to significantly clearer code.

While Numba is impressive, the beauty of JAX is that, with fully vectorized
operations, we can run exactly the
same code on machines with hardware accelerators and reap all the benefits
without paying extra cost.


## Sequential operations

Some operations are inherently sequential -- and hence difficult or impossible
to vectorize.

In this case NumPy is a poor option and we are left with the choice of Numba or
JAX.

To compare these choices, we will revisit the problem of iterating on the
quadratic map that we saw in our {doc}`Numba lecture <numba>`.


### Numba Version

Here's the Numba version.

```{code-cell} ipython3
@numba.jit
def qm(x0, n, α=4.0):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
      x[t+1] = α * x[t] * (1 - x[t])
    return x
```

Let's generate a time series of length 10,000,000 and time the execution:

```{code-cell} ipython3
n = 10_000_000

with qe.Timer(precision=8):
    x = qm(0.1, n)
```

Let's run it again to eliminate compilation time:

```{code-cell} ipython3
with qe.Timer(precision=8):
    x = qm(0.1, n)
```

Numba handles this sequential operation very efficiently.

Notice that the second run is significantly faster after JIT compilation completes.

Numba's compilation is typically quite fast, and the resulting code performance is excellent for sequential operations like this one.

### JAX Version

Now let's create a JAX version using `lax.scan`:

```{code-cell} ipython3
from jax import lax
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def qm_jax(x0, n, α=4.0):
    def update(x, t):
        x_new = α * x * (1 - x)
        return x_new, x_new

    _, x = lax.scan(update, x0, jnp.arange(n))
    return jnp.concatenate([jnp.array([x0]), x])
```

Let's time it with the same parameters:

```{code-cell} ipython3
with qe.Timer(precision=8):
    x_jax = qm_jax(0.1, n).block_until_ready()
```

Let's run it again to eliminate compilation overhead:

```{code-cell} ipython3
with qe.Timer(precision=8):
    x_jax = qm_jax(0.1, n).block_until_ready()
```

JAX is also very efficient for this sequential operation.

Both JAX and Numba deliver strong performance after compilation.

While the raw speed is similar for this type of operation, there are notable differences in code complexity and ease of understanding, which we discuss in the next section.

### Summary

While both Numba and JAX deliver excellent performance for sequential operations, there are significant differences in code readability and ease of use.

The Numba version is straightforward and natural to read: we simply allocate an array and fill it element by element using a standard Python loop.

This is exactly how most programmers think about the algorithm.

The JAX version, on the other hand, requires using `lax.scan`, which is less intuitive and has a steeper learning curve.

Additionally, JAX's immutable arrays mean we cannot simply update array elements in place.

Instead, we must use functional programming patterns with `lax.scan`, where we define an `update` function that returns both the new state and the value to accumulate.

For this type of sequential operation, Numba is the clear winner in terms of code clarity and ease of implementation, while maintaining competitive performance.
