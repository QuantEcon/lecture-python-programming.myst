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

# Parallelization

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

```{code-cell} ipython
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
```

## Vectorized operations

### Speed

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
