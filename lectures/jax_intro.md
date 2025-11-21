---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# An Introduction to JAX

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install jax quantecon
```

This lecture provides a short introduction to [Google JAX](https://github.com/jax-ml/jax).

Here we are focused on using JAX on the CPU, rather than on accelerators such as
GPUs or TPUs.

This means we will only see a small amount of the possible benefits from using
JAX.

At the same time, JAX computing on the CPU is a good place to start, since the
JAX just-in-time compiler seamlessly handles transitions across different
hardware platforms.

(In other words, if you do want to shift to using GPUs, you will almost never
need to modify your code.)

For a discussion of JAX on GPUs, see [our JAX lecture series](https://jax.quantecon.org/intro.html).


## JAX as a NumPy Replacement

One way to use JAX is as a plug-in NumPy replacement. Let's look at the
similarities and differences.

### Similarities


The following import is standard, replacing `import numpy as np`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import quantecon as qe
```

Now we can use `jnp` in place of `np` for the usual array operations:

```{code-cell} ipython3
a = jnp.asarray((1.0, 3.2, -1.5))
```

```{code-cell} ipython3
print(a)
```

```{code-cell} ipython3
print(jnp.sum(a))
```

```{code-cell} ipython3
print(jnp.mean(a))
```

```{code-cell} ipython3
print(jnp.dot(a, a))
```

However, the array object `a` is not a NumPy array:

```{code-cell} ipython3
a
```

```{code-cell} ipython3
type(a)
```

Even scalar-valued maps on arrays return JAX arrays.

```{code-cell} ipython3
jnp.sum(a)
```

Operations on higher dimensional arrays are also similar to NumPy:

```{code-cell} ipython3
A = jnp.ones((2, 2))
B = jnp.identity(2)
A @ B
```

```{code-cell} ipython3
from jax.numpy import linalg
```

```{code-cell} ipython3
linalg.inv(B)   # Inverse of identity is identity
```

```{code-cell} ipython3
linalg.eigh(B)  # Computes eigenvalues and eigenvectors
```

### Differences


One difference between NumPy and JAX is that JAX uses 32 bit floats by default.

This is because JAX is often used for GPU computing, and most GPU computations use 32 bit floats.

Using 32 bit floats can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.

In these cases 64 bit floats can be enforced via the command

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

Let's check this works:

```{code-cell} ipython3
jnp.ones(3)
```

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.

For example, with NumPy we can write

```{code-cell} ipython3
import numpy as np
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```{code-cell} ipython3
a[0] = 1
a
```

In JAX this fails:

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
a
```

```{code-cell} ipython3
:tags: [raises-exception]

a[0] = 1
```

In line with immutability, JAX does not support inplace operations:

```{code-cell} ipython3
a = np.array((2, 1))
a.sort()
a
```

```{code-cell} ipython3
a = jnp.array((2, 1))
a_new = a.sort()
a, a_new
```

The designers of JAX chose to make arrays immutable because JAX uses a
*functional programming style*.

This design choice has important implications, which we explore next!

We should note, however, that, JAX does provide a version of in-place array modification
using the [`at` method](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html).

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
```

Applying `at[0].set(1)` returns a new copy of `a` with the first element set to 1

```{code-cell} ipython3
a = a.at[0].set(1)
a
```

Obviously, there are downsides to using `at`.

The syntax is not very pretty and we want to avoid creating fresh arrays in memory every time we change a single value.

Hence, for the most part, we try to avoid this syntax.

(Although it can in fact be efficient inside JIT-compiled functions -- but let's
put this aside for now.)


## Functional Programming

From JAX's documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has "una anima di pura programmazione funzionale".*

In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure.


Pure functions have the following characteristics:

1. *Deterministic*
2. *No side effects*

Deterministic means

*  Same input $\implies$ same output
*  Outputs do not depend on global state

In particular, pure functions will always return the same result if invoked with the same inputs.

No side effects means that the function

* Won't change global state
* Won't modify data passed to the function (immutable data)

### Examples

Here's an example of a non-pure function

```{code-cell} ipython3
tax_rate = 0.1
prices = [10.0, 20.0]

def add_tax(prices):
    for i, price in enumerate(prices):
        prices[i] = price * (1 + tax_rate)
    print('Modified prices: ', prices)
    return prices
```

This function fails to be pure because

* side effects --- it modifies the global variable `prices`
* non-deterministic --- a change to the global variable `tax_rate` will modify
  function outputs, even with the same inputs.

Here's a pure version

```{code-cell} ipython3
tax_rate = 0.1
prices = (10.0, 20.0)

def add_tax_pure(prices, tax_rate):
    return [price * (1 + tax_rate) for price in prices]
```

This pure version makes all dependencies explicit through function arguments, and doesn't modify any external state.

Now that we understand what pure functions are, let's explore how JAX's approach to random numbers maintains this purity.


## Random Numbers

Random numbers are rather different in JAX, compared to what you find in NumPy
or Matlab.

At first you might find the syntax rather verbose.

But actually it makes a lot of sense:

* maintains the functional programming style we just discussed, and
* makes the control of random state explicit and convenient for running over
  multiple threads --- essential for parallelization.

### Random number generation

In JAX, the state of the random number generator needs to be controlled explicitly.

First we produce a key, which seeds the random number generator.

```{code-cell} ipython3
key = jax.random.PRNGKey(1)
```

Now we can use the key to generate some random numbers:

```{code-cell} ipython3
x = jax.random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```{code-cell} ipython3
jax.random.normal(key, (3, 3))
```

To produce a (quasi-) independent draw, one option is to "split" the existing key:

```{code-cell} ipython3
key, subkey = jax.random.split(key)
```

```{code-cell} ipython3
jax.random.normal(key, (3, 3))
```

```{code-cell} ipython3
jax.random.normal(subkey, (3, 3))
```

This syntax will seem unusual for a NumPy or Matlab user --- but will make a lot
of sense when we progress to parallel programming.

The function below produces `k` (quasi-) independent random `n x n` matrices using `split`.

```{code-cell} ipython3
def gen_random_matrices(key, n=2, k=3):
    matrices = []
    for _ in range(k):
        key, subkey = jax.random.split(key)
        A = jax.random.uniform(subkey, (n, n))
        matrices.append(A)
        print(A)
    return matrices
```

```{code-cell} ipython3
key = jax.random.PRNGKey(1)
matrices = gen_random_matrices(key)
```

We can also use `fold_in` when iterating in a loop:

```{code-cell} ipython3
def gen_random_matrices(key, n=2, k=3):
    matrices = []
    for i in range(k):
        step_key = jax.random.fold_in(key, i)
        A = jax.random.uniform(step_key, (n, n))
        matrices.append(A)
        print(A)
    return matrices
```

```{code-cell} ipython3
key = jax.random.PRNGKey(1)
matrices = gen_random_matrices(key)
```

### Why explicit random state?

Why does JAX require this somewhat verbose approach to random number generation.

The reason is to maintain pure functions.

Let's see how random number generation relates to pure functions by comparing NumPy and JAX.

#### NumPy's approach

In NumPy, random number generation works by maintaining hidden global state.

Each time we call a random function, this state is updated:

```{code-cell} ipython3
np.random.seed(42)
print(np.random.randn())
print(np.random.randn())
print(np.random.randn())
```

Notice that each call returns a different value, even though we're calling the same function with the same inputs (no arguments).

This function is *not pure* because:

* It's non-deterministic: same inputs (none, in this case) give different outputs
* It has side effects: it modifies the global random number generator state


#### JAX's approach

As we saw above, JAX takes a different approach, making randomness explicit through keys.

For example,

```{code-cell} ipython3
def random_sum_jax(key):
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1)
    y = jax.random.normal(key2)
    return x + y
```

With the same key, we always get the same result:

```{code-cell} ipython3
key = jax.random.PRNGKey(42)
random_sum_jax(key)
```

```{code-cell} ipython3
random_sum_jax(key)
```

Different keys give different results:

```{code-cell} ipython3
key1 = jax.random.PRNGKey(1)
key2 = jax.random.PRNGKey(2)
print(random_sum_jax(key1))
print(random_sum_jax(key2))
```

The  function `random_sum_jax` is pure because:

* It's deterministic: same key always produces same output
* No side effects: no hidden state is modified

The explicitness of JAX brings significant benefits:

* Reproducibility: Easy to reproduce results by reusing keys
* Parallelization: Each thread can have its own key without conflicts
* Debugging: No hidden state makes code easier to reason about
* JIT compatibility: The compiler can optimize pure functions more aggressively

The last point about JIT compatibility is explained in the next section.


## JIT compilation

The JAX just-in-time (JIT) compiler accelerates execution by generating
efficient machine code that varies with both task size and hardware.

### A simple example

Let's say we want to evaluate the cosine function at many points.

```{code-cell}
n = 50_000_000
x = np.linspace(0, 10, n)
```

#### With NumPy

Let's try with NumPy

```{code-cell}
with qe.Timer():
    y = np.cos(x)
```

And one more time.

```{code-cell}
with qe.Timer():
    y = np.cos(x)
```

Here NumPy uses a pre-built binary file, compiled from carefully written
low-level code, for applying cosine to an array of floats.

This binary file ships with NumPy.

#### With JAX

Now let's try with JAX.

```{code-cell}
x = jnp.linspace(0, 10, n)
```

Let's time the same procedure.

```{code-cell}
with qe.Timer():
    y = jnp.cos(x)
    jax.block_until_ready(y);
```

```{note}
Here, in order to measure actual speed, we use the `block_until_ready` method
to hold the interpreter until the results of the computation are returned.

This is necessary because JAX uses asynchronous dispatch, which
allows the Python interpreter to run ahead of numerical computations.

For non-timed code, you can drop the line containing `block_until_ready`.
```


And let's time it again.


```{code-cell}
with qe.Timer():
    y = jnp.cos(x)
    jax.block_until_ready(y);
```

If you are running this on a GPU the code will run much faster than its NumPy
equivalent, which ran on the CPU.

Even if you are running on a machine with many CPUs, the second JAX run should
be substantially faster with JAX.

But notice also that the second time is shorter than the first.

This is because even built in functions like `jnp.cos` are JIT-compiled --- and the
first run includes compile time.

Why would JAX want to JIT-compile built in functions like `jnp.cos` instead of
just providing pre-compiled versions, like NumPy?

The reason is that the JIT compiler wants to specialize on the *size* of the array
being used (as well as the data type).

The size matters for generating optimized code because efficient parallelization
requires matching the size of the task to the available hardware.

That's why JAX waits to see the size of the array before compiling --- which
requires a JIT-compiled approach instead of supplying precompiled binaries.

#### Changing array sizes

Here we change the input size and see the run time increase and then fall again.

```{code-cell}
x = jnp.linspace(0, 10, n + 1)
```

```{code-cell}
with qe.Timer():
    y = jnp.cos(x)
    jax.block_until_ready(y);
```


```{code-cell}
with qe.Timer():
    y = jnp.cos(x)
    jax.block_until_ready(y);
```

This is because the JIT compiler specializes on array size to exploit
parallelization --- and hence generates fresh compiled code when the array size
changes.

### Evaluating a more complicated function

Let's try the same thing with a more complex function.

```{code-cell}
def f(x):
    y = np.cos(2 * x**2) + np.sqrt(np.abs(x)) + 2 * np.sin(x**4) - 0.1 * x**2
    return y
```

#### With NumPy

We'll try first with NumPy

```{code-cell}
n = 50_000_000
x = np.linspace(0, 10, n)
```

```{code-cell}
with qe.Timer():
    y = f(x)
```

```{code-cell}
with qe.Timer():
    y = f(x)
```


#### With JAX

Now let's try again with JAX.

As a first pass, we replace `np` with `jnp` throughout:

```{code-cell}
def f(x):
    y = jnp.cos(2 * x**2) + jnp.sqrt(jnp.abs(x)) + 2 * jnp.sin(x**4) - x**2
    return y
```

Now let's time it.

```{code-cell}
x = jnp.linspace(0, 10, n)
```

```{code-cell}
with qe.Timer():
    y = f(x)
    jax.block_until_ready(y);
```

```{code-cell}
with qe.Timer():
    y = f(x)
    jax.block_until_ready(y);
```

The outcome is similar to the `cos` example --- JAX is faster, especially if you
use a GPU and especially on the second run.

Moreover, with JAX, we have another trick up our sleeve:


### Compiling the Whole Function

The JAX just-in-time (JIT) compiler can accelerate execution within functions by fusing linear
algebra operations into a single optimized kernel.

Let's try this with the function `f`:

```{code-cell}
f_jax = jax.jit(f)
```

```{code-cell}
with qe.Timer():
    y = f_jax(x)
    jax.block_until_ready(y);
```

```{code-cell}
with qe.Timer():
    y = f_jax(x)
    jax.block_until_ready(y);
```

The runtime has improved again --- now because we fused all the operations,
allowing the compiler to optimize more aggressively.

For example, the compiler can eliminate multiple calls to the hardware
accelerator and the creation of a number of intermediate arrays.


Incidentally, a more common syntax when targeting a function for the JIT
compiler is

```{code-cell} ipython3
@jax.jit
def f(x):
    pass # put function body here
```

### Compiling non-pure functions

Now that we've seen how powerful JIT compilation can be, it's important to understand its relationship with pure functions.

JAX will not usually throw errors when compiling impure functions but execution becomes unpredictable.

Here's an illustration of this fact, using global variables:

```{code-cell} ipython3
a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```{code-cell} ipython3
x = jnp.ones(2)
```

```{code-cell} ipython3
f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected --- as long as the same compiled version is called.

```{code-cell} ipython3
a = 42
```

```{code-cell} ipython3
f(x)
```

Changing the dimension of the input triggers a fresh compilation of the function, at which time the change in the value of `a` takes effect:

```{code-cell} ipython3
x = jnp.ones(3)
```

```{code-cell} ipython3
f(x)
```

Moral of the story: write pure functions when using JAX!


### Summary

Now we can see why both developers and compilers benefit from pure functions.

We love pure functions because they

* Help testing: each function can operate in isolation
* Promote deterministic behavior and hence reproducibility
* Prevent bugs that arise from mutating shared state

The compiler loves pure functions and functional programming because

* Data dependencies are explicit, which helps with optimizing complex computations
* Pure functions are easier to differentiate (autodiff)
* Pure functions are easier to parallelize and optimize (don't depend on shared mutable state)


## Gradients

JAX can use automatic differentiation to compute gradients.

This can be extremely useful for optimization and solving nonlinear systems.

We will see significant applications later in this lecture series.

For now, here's a very simple illustration involving the function

```{code-cell} ipython3
def f(x):
    return (x**2) / 2
```

Let's take the derivative:

```{code-cell} ipython3
f_prime = jax.grad(f)
```

```{code-cell} ipython3
f_prime(10.0)
```

Let's plot the function and derivative, noting that $f'(x) = x$.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x_grid = jnp.linspace(-4, 4, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, [f_prime(x) for x in x_grid], label="$f'$")
ax.legend(loc='upper center')
plt.show()
```

We defer further exploration of automatic differentiation with JAX until {doc}`jax:autodiff`.


## Exercises


```{exercise-start}
:label: jax_intro_ex2
```

In the Exercise section of {doc}`a lecture on Numba <numba>`, we used Monte
Carlo to price a European call option.

The code was accelerated by Numba-based multithreading.

Try writing a version of this operation for JAX, using all the same
parameters.



```{exercise-end}
```


```{solution-start} jax_intro_ex2
:class: dropdown
```
Here is one solution:

```{code-cell} ipython3
M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           key=jax.random.PRNGKey(1)):

    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)
    for t in range(n):
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))

    return β**n * expectation
```

Let's run it once to compile it:

```{code-cell} ipython3
with qe.Timer():
    compute_call_price_jax().block_until_ready()
```

And now let's time it:

```{code-cell} ipython3
with qe.Timer():
    compute_call_price_jax().block_until_ready()
```

```{solution-end}
```
