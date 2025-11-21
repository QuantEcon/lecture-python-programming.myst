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
footprint but also elegance and ease of use.

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon jax
```

We will use the following imports.

```{code-cell} ipython
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

```{code-cell} ipython

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

```{code-cell} python3
grid = np.linspace(-3, 3, 50)
m = -np.inf
for x in grid:
    for y in grid:
        z = f(x, y)
        if z > m:
            m = z
```


### NumPy vectorization

If we switch to NumPy style vectorization we can use a much larger grid and the
code executes relatively quickly:

```{code-cell} python3
grid = np.linspace(-3, 3, 1000)
x, y = np.meshgrid(grid, grid)

with qe.Timer():
    np.max(f(x, y))
```

In the vectorized version, all the looping takes place in compiled code.

One of the reasons this is quick is that NumPy uses implicit multithreading,
so that at least some parallelization occurs.

```{note}
If you have a system monitor such as htop (Linux/Mac) or perfmon
(Windows), then try running this and then observing the load on your CPUs.

(You will probably need to bump up the grid size to see large effects.)

The output typically shows that the operation is successfully distributed across multiple threads.
```


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

grid = np.linspace(-3, 3, 1000)

with qe.Timer():
    compute_max_numba(grid)
```

This Numba version is competitive with the vectorized NumPy code, but uses less memory since we don't create mesh grids.

Note that this timing includes JIT compilation overhead.

Once the function is compiled, subsequent calls will be faster.

### Parallelized Numba

Now let's try parallelization with Numba using `prange`:

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

with qe.Timer():
    compute_max_numba_parallel(grid)
```

The parallelized version's performance depends on several factors.

Parallelization introduces overhead for thread management and synchronization.

For some problem sizes and configurations, this overhead can outweigh the benefits of parallel execution.

Generally, parallel Numba shows its strength with larger computational workloads where the work per thread justifies the parallelization overhead.

What if we try to parallelize both loops?

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

with qe.Timer():
    compute_max_numba_parallel_nested(grid)
```

While Numba does support nested `prange` loops, the performance may not be what you expect.

In this case, the nested parallel version performs slightly better than parallelizing only the outer loop, but still worse than the simple non-parallel version.

This illustrates an important principle: more parallelization is not always better.

The overhead of thread management and synchronization can easily overwhelm the benefits of parallel execution, especially for moderate-sized problems like this one.

Another issue with parallelizing this particular example is that all threads need to update the shared variable `m` to track the current maximum.

This creates contention, as threads must synchronize their access to `m` to avoid race conditions.

For reduction operations like this (finding a maximum, sum, etc.), a more efficient approach is to have each thread maintain a local maximum.

These local maxima can then be combined in a final reduction step.



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
grid = jnp.linspace(-3, 3, 1000)
x_mesh, y_mesh = np.meshgrid(grid, grid)

with qe.Timer():
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
with qe.Timer():
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Notice the dramatic speedup on the second run after JIT compilation completes.

JAX's JIT compiler analyzes the computation and generates highly optimized machine code.

Once compiled, JAX can be significantly faster than NumPy, especially on problems that can leverage GPU acceleration or other specialized hardware.

The compilation overhead is a one-time cost that pays off when the function is called repeatedly.


### JAX plus vmap

There is one problem with both the NumPy code and the JAX code: the mesh grids use a lot of memory.

```{code-cell} ipython3
x_mesh.nbytes + y_mesh.nbytes
```

By comparison, the flat arrays are much smaller:

```{code-cell} ipython3
x = grid
y = grid
x.nbytes + y.nbytes  # y is just a pointer to x, so minimal extra memory
```

This extra memory usage can be a big problem in actual research calculations.

So let's try a different approach using [jax.vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)

First we vectorize `f` in `y`.

```{code-cell} ipython3
f_vec_y = jax.vmap(f, in_axes=(None, 0))
```

In the line above, `(None, 0)` indicates that we are vectorizing in the second argument, which is `y`.

Next, we vectorize in the first argument, which is `x`.

```{code-cell} ipython3
f_vec = jax.vmap(f_vec_y, in_axes=(0, None))
```

With this construction, we can now call the function $f$ on flat (low memory) arrays.

```{code-cell} ipython3
with qe.Timer():
    z_vmap = f_vec(x, y).block_until_ready()
```

Let's run it again to eliminate compilation time:

```{code-cell} ipython3
with qe.Timer():
    z_vmap = f_vec(x, y).block_until_ready()
```

The execution time is essentially the same as the mesh operation but we are using much less memory.

This is an important advantage: `vmap` achieves the same performance as explicit mesh grids while using significantly less memory.

For larger-scale problems, this memory efficiency can be the difference between a computation that fits in memory and one that doesn't.

And we produce the correct answer:

```{code-cell} ipython3
jnp.allclose(z_vmap, z_mesh)
```


## Sequential operations

Some operations are inherently sequential -- and hence difficult or impossible
to vectorize.

In this case NumPy is a poor option and we are left with the choice of Numba or
JAX.

### Numba Version


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

with qe.Timer():
    x = qm(0.1, n)
```

Let's run it again to eliminate compilation time:

```{code-cell} ipython3
with qe.Timer():
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
with qe.Timer():
    x_jax = qm_jax(0.1, n).block_until_ready()
```

Let's run it again to eliminate compilation overhead:

```{code-cell} ipython3
with qe.Timer():
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
