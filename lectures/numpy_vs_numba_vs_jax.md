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

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon jax
```

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

### Speed

### Speed Comparisons

```{index} single: Vectorization; Operations on Arrays
```

We mentioned in an {doc}`previous lecture <need_for_speed>` that NumPy-based vectorization can
accelerate scientific applications.

In this section we try some speed comparisons to illustrate this fact.

### Vectorization vs Loops

Let's begin with some non-vectorized code, which uses a native Python loop to generate,
square and then sum a large number of random variables:

```{code-cell} python3
n = 1_000_000
```

```{code-cell} python3
with qe.Timer():
    y = 0      # Will accumulate and store sum
    for i in range(n):
        x = random.uniform(0, 1)
        y += x**2
```

The following vectorized code achieves the same thing.

```{code-cell} ipython
with qe.Timer():
    x = np.random.uniform(0, 1, n)
    y = np.sum(x**2)
```

As you can see, the second code block runs much faster.  Why?

The second code block breaks the loop down into three basic operations

1. draw `n` uniforms
1. square them
1. sum them

These are sent as batch operators to optimized machine code.

Apart from minor overheads associated with sending data back and forth, the result is C or Fortran-like speed.

When we run batch operations on arrays like this, we say that the code is *vectorized*.

The next section illustrates this point.

(ufuncs)=
### Universal Functions

```{index} single: NumPy; Universal Functions
```

As discussed above, many functions provided by NumPy are universal functions (ufuncs).

By exploiting ufuncs, many operations can be vectorized, leading to faster
execution.

For example, consider the problem of maximizing a function $f$ of two
variables $(x,y)$ over the square $[-a, a] \times [-a, a]$.

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

To maximize it, we're going to use a naive grid search:

1. Evaluate $f$ for all $(x,y)$ in a grid on the square.
1. Return the maximum of observed values.

The grid will be

```{code-cell} python3
grid = np.linspace(-3, 3, 1000)
```

Here's a non-vectorized version that uses Python loops.

```{code-cell} python3
with qe.Timer():
    m = -np.inf

    for x in grid:
        for y in grid:
            z = f(x, y)
            if z > m:
                m = z
```

And here's a vectorized version

```{code-cell} python3
with qe.Timer():
    x, y = np.meshgrid(grid, grid)
    np.max(f(x, y))
```

In the vectorized version, all the looping takes place in compiled code.

As you can see, the second version is *much* faster.


### Implicit Multithreading in NumPy

Actually, you have already been using multithreading in your Python code,
although you might not have realized it.

(We are, as usual, assuming that you are running the latest version of
Anaconda Python.)

This is because NumPy cleverly implements multithreading in a lot of its
compiled code.

Let's look at an example to see this in action.

The next piece of code computes the eigenvalues of a large number of randomly
generated matrices.

It takes a few seconds to run.

```{code-cell} python3
n = 20
m = 1000
for i in range(n):
    X = np.random.randn(m, m)
    λ = np.linalg.eigvals(X)
```

Now, let's look at the output of the htop system monitor on our machine while
this code is running:

```{figure} /_static/lecture_specific/parallelization/htop_parallel_npmat.png
:scale: 80
```

We can see that 4 of the 8 CPUs are running at full speed.

This is because NumPy's `eigvals` routine neatly splits up the tasks and
distributes them to different threads.


#### A Multithreaded Ufunc

Over the last few years, NumPy has managed to push this kind of multithreading
out to more and more operations.

For example, let's return to a maximization problem {ref}`discussed previously <ufuncs>`:

```{code-cell} python3
def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

grid = np.linspace(-3, 3, 5000)
x, y = np.meshgrid(grid, grid)
```

```{code-cell} ipython3
with qe.Timer():
    np.max(f(x, y))
```

If you have a system monitor such as htop (Linux/Mac) or perfmon
(Windows), then try running this and then observing the load on your CPUs.

(You will probably need to bump up the grid size to see large effects.)

At least on our machine, the output shows that the operation is successfully
distributed across multiple threads.

This is one of the reasons why the vectorized code above is fast.

#### A Comparison with Numba

To get some basis for comparison for the last example, let's try the same
thing with Numba.

In fact there is an easy way to do this, since Numba can also be used to
create custom {ref}`ufuncs <ufuncs>` with the [@vectorize](https://numba.pydata.org/numba-doc/dev/user/vectorize.html) decorator.

```{code-cell} python3
from numba import vectorize

@vectorize
def f_vec(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

np.max(f_vec(x, y))  # Run once to compile
```

```{code-cell} ipython3
with qe.Timer():
    np.max(f_vec(x, y))
```

At least on our machine, the difference in the speed between the
Numba version and the vectorized NumPy version shown above is not large.

But there's quite a bit going on here so let's try to break down what is
happening.

Both Numba and NumPy use efficient machine code that's specialized to these
floating point operations.

However, the code NumPy uses is, in some ways, less efficient.

The reason is that, in NumPy, the operation `np.cos(x**2 + y**2) / (1 +
x**2 + y**2)` generates several intermediate arrays.

For example, a new array is created when `x**2` is calculated.

The same is true when `y**2` is calculated, and then `x**2 + y**2` and so on.

Numba avoids creating all these intermediate arrays by compiling one
function that is specialized to the entire operation.

But if this is true, then why isn't the Numba code faster?

The reason is that NumPy makes up for its disadvantages with implicit
multithreading, as we've just discussed.

#### Multithreading a Numba Ufunc

Can we get both of these advantages at once?

In other words, can we pair

* the efficiency of Numba's highly specialized JIT compiled function and
* the speed gains from parallelization obtained by NumPy's implicit
  multithreading?

It turns out that we can, by adding some type information plus `target='parallel'`.

```{code-cell} python3
@vectorize('float64(float64, float64)', target='parallel')
def f_vec(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

np.max(f_vec(x, y))  # Run once to compile
```

```{code-cell} ipython3
with qe.Timer():
    np.max(f_vec(x, y))
```

Now our code runs significantly faster than the NumPy version.

#### Writing vectorized code

Writing fast JAX code requires shifting repetitive tasks from loops to array processing operations, so that the JAX compiler can easily understand the whole operation and generate more efficient machine code.

This procedure is called **vectorization** or **array programming**, and will be
familiar to anyone who has used NumPy or MATLAB.

In most ways, vectorization is the same in JAX as it is in NumPy.

But there are also some differences, which we highlight here.

As a running example, consider the function

$$
    f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
$$

Suppose that we want to evaluate this function on a square grid of $x$ and $y$ points and then plot it.

To clarify, here is the slow `for` loop version.

```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)

n = 80
x = jnp.linspace(-2, 2, n)
y = x

z_loops = np.empty((n, n))
```

```{code-cell} ipython3
with qe.Timer():
    for i in range(n):
        for j in range(n):
            z_loops[i, j] = f(x[i], y[j])
```

Even for this very small grid, the run time is extremely slow.

(Notice that we used a NumPy array for `z_loops` because we wanted to write to it.)

+++

OK, so how can we do the same operation in vectorized form?

If you are new to vectorization, you might guess that we can simply write

```{code-cell} ipython3
z_bad = f(x, y)
```

But this gives us the wrong result because JAX doesn't understand the nested for loop.

```{code-cell} ipython3
z_bad.shape
```

Here is what we actually wanted:

```{code-cell} ipython3
z_loops.shape
```

To get the right shape and the correct nested for loop calculation, we can use a `meshgrid` operation designed for this purpose:

```{code-cell} ipython3
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

Now we get what we want and the execution time is very fast.

```{code-cell} ipython3
with qe.Timer():
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
with qe.Timer():
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

Let's confirm that we got the right answer.

```{code-cell} ipython3
jnp.allclose(z_mesh, z_loops)
```

Now we can set up a serious grid and run the same calculation (on the larger grid) in a short amount of time.

```{code-cell} ipython3
n = 6000
x = jnp.linspace(-2, 2, n)
y = x
x_mesh, y_mesh = jnp.meshgrid(x, y)
```

```{code-cell} ipython3
with qe.Timer():
    z_mesh = f(x_mesh, y_mesh).block_until_ready()
```

But there is one problem here: the mesh grids use a lot of memory.

```{code-cell} ipython3
x_mesh.nbytes + y_mesh.nbytes
```

By comparison, the flat array `x` is just

```{code-cell} ipython3
x.nbytes  # and y is just a pointer to x
```

This extra memory usage can be a big problem in actual research calculations.

So let's try a different approach using [jax.vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)

+++

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

The execution time is essentially the same as the mesh operation but we are using much less memory.

And we produce the correct answer:

```{code-cell} ipython3
jnp.allclose(z_vmap, z_mesh)
```

### Syntax and Semantics

## Sequential operations

### Speed

### Syntax and Semantics
