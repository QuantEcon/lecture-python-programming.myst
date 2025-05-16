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

# Python for Scientific Computing

```{epigraph}
"We should forget about small efficiencies, say about 97% of the time:
premature optimization is the root of all evil." -- Donald Knuth
```

## Overview

Python is extremely popular for scientific computing, due to such factors as

* the accessible and expressive nature of the language itself,
* its vast range of high quality scientific libraries,
* the fact that the language and libraries are open source,
* the popular [Anaconda Python distribution](https://www.anaconda.com/download), which simplifies installation and management of scientific libraries, and
* the key role that Python plays in data science, machine learning and artificial intelligence.

In previous lectures, we looked at some scientific Python libaries such as NumPy and Matplotlib.

However, our main focus was the core Python language, rather than the libraries.

Now we turn to the scientific libraries and give them our full attention.

We'll also discuss the following topics:

* What are the relative strengths and weaknesses of Python for scientific work?
* What are the main elements of the scientific Python ecosystem?
* How is the situation changing over time?

In addition to what's in Anaconda, this lecture will need

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```



## Scientific Libraries

Let's briefly review Python's scientific libraries, starting with why we need them.

### The Role of Scientific Libraries

One reason we use scientific libraries is because they implement routines we want to use.

* numerical integration, interpolation, linear algebra, root finding, etc.

For example, it's almost always better to use an existing routine for root finding than to write a new one from scratch.

(For standard algorithms, efficiency is maximized if the community can coordinate on a common set of implementations, written by experts and tuned by users to be as fast and robust as possible.)

But this is not the only reason that we use Python's scientific libraries.

Another is that pure Python, while flexible and elegant, is not fast.

So we need libraries that are designed to accelerate execution of Python code.

They do this using two strategies:

1. using compilers that convert Python-like statements into fast machine code for individual threads of logic and
2. parallelizing tasks across multiple "workers" (e.g., CPUs, individual threads inside GPUs).

There are several Python libraries that can do this extremely well.


### Python's Scientific Ecosystem

At QuantEcon, the scientific libraries we use most often are

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/) 
* [Pandas](https://pandas.pydata.org/)
* [Numba](https://numba.pydata.org/) and
* [JAX](https://github.com/jax-ml/jax)

Here's how they fit together:

* NumPy forms foundations by providing a basic array data type (think of
  vectors and matrices) and functions for acting on these arrays (e.g., matrix
  multiplication).
* SciPy builds on NumPy by adding numerical methods routinely used in science (interpolation, optimization, root finding, etc.).
* Matplotlib is used to generate figures, with a focus on plotting data stored in NumPy arrays.
* Pandas provides types and functions for manipulating data.
* Numba provides a just-in-time compiler that integrates well with NumPy and
  helps accelerate Python code.
* JAX includes array processing operations similar to NumPy, automatic
  differentiation, a parallelization-centric just-in-time compiler, and automated integration with hardware accelerators such as
  GPUs.




## The Need for Speed

Let's discuss execution speed and how scientific libraries can help us accelerate code.

Higher-level languages like Python  are optimized for humans.

This means that the programmer can leave many details to the runtime environment

* specifying variable types
* memory allocation/deallocation, etc.

On one hand, compared to low-level languages, high-level languages are typically faster to write, less error-prone and  easier to debug.

On the other hand, high-level languages are harder to optimize --- that is, to turn into fast machine code --- than languages like C or Fortran.

Indeed, the standard implementation of Python (called CPython) cannot match the speed of compiled languages such as C or Fortran.

Does that mean that we should just switch to C or Fortran for everything?

The answer is: No, no, and one hundred times no!

(This is what you should say to your professor when they insist that your model needs to be rewritten in Fortran or C++.)

There are two reasons why:

First, for any given program, relatively few lines are ever going to be time-critical.

Hence it is far more efficient to write most of our code in a high productivity language like Python.

Second, even for those lines of code that *are* time-critical, we can now achieve the same speed as C or Fortran using Python's scientific libraries.

In fact we can often do better, because some scientific libraries are so
effective at accelerating and parallelizing our code.


### Where are the Bottlenecks?

Before we learn how to do this, let's try to understand why plain vanilla Python is slower than C or Fortran.

This will, in turn, help us figure out how to speed things up.

In reading the following, remember that the Python interpreter executes code line-by-line.

#### Dynamic Typing

```{index} single: Dynamic Typing
```

Consider this Python operation

```{code-cell} python3
a, b = 10, 10
a + b
```

Even for this simple operation, the Python interpreter has a fair bit of work to do.

For example, in the statement `a + b`, the interpreter has to know which
operation to invoke.

If `a` and `b` are strings, then `a + b` requires string concatenation

```{code-cell} python3
a, b = 'foo', 'bar'
a + b
```

If `a` and `b` are lists, then `a + b` requires list concatenation

```{code-cell} python3
a, b = ['foo'], ['bar']
a + b
```

(We say that the operator `+` is *overloaded* --- its action depends on the
type of the objects on which it acts)

As a result, when executing `a + b`, Python must first check the type of the objects and then call the correct operation.

This involves substantial overheads.


#### Static Types

```{index} single: Static Types
```

Compiled languages avoid these overheads with explicit, static types.

For example, consider the following C code, which sums the integers from 1 to 10

```{code-block} c
:class: no-execute

#include <stdio.h>

int main(void) {
    int i;
    int sum = 0;
    for (i = 1; i <= 10; i++) {
        sum = sum + i;
    }
    printf("sum = %d\n", sum);
    return 0;
}
```

The variables `i` and `sum` are explicitly declared to be integers.

Hence, the meaning of addition here is completely unambiguous.

### Data Access

Another drag on speed for high-level languages is data access.

To illustrate, let's consider the problem of summing some data --- say, a collection of integers.

#### Summing with Compiled Code

In C or Fortran, these integers would typically be stored in an array, which
is a simple data structure for storing homogeneous data.

Such an array is stored in a single contiguous block of memory

* In modern computers, memory addresses are allocated to each byte (one byte = 8 bits).
* For example, a 64 bit integer is stored in 8 bytes of memory.
* An array of $n$ such integers occupies $8n$ **consecutive** memory slots.

Moreover, the compiler is made aware of the data type by the programmer.

* In this case 64 bit integers

Hence, each successive data point can be accessed by shifting forward in memory
space by a known and fixed amount.

* In this case 8 bytes

#### Summing in Pure Python

Python tries to replicate these ideas to some degree.

For example, in the standard Python implementation (CPython), list elements are placed in memory locations that are in a sense contiguous.

However, these list elements are more like pointers to data rather than actual data.

Hence, there is still overhead involved in accessing the data values themselves.

This is a considerable drag on speed.

In fact, it's generally true that memory traffic is a major culprit when it comes to slow execution.

Let's look at some ways around these problems.




## {index}`Vectorization <single: Vectorization>`

```{index} single: Python; Vectorization
```

There is a clever method called **vectorization** that can be
used to speed up high level languages in numerical applications.

The key idea is to send array processing operations in batch to pre-compiled
and efficient native machine code.

The machine code itself is typically compiled from carefully optimized C or Fortran.

For example, when working in a high level language, the operation of inverting a large matrix can be subcontracted to efficient machine code that is pre-compiled for this purpose and supplied to users as part of a package.

This clever idea dates back to MATLAB, which uses vectorization extensively.


```{figure} /_static/lecture_specific/need_for_speed/matlab.png
```

Vectorization can greatly accelerate many numerical computations, as we will see
in later lectures.

(numba-p_c_vectorization)=
## Beyond Vectorization

At its best, vectorization yields fast, simple code.

However, it's not without disadvantages.

One issue is that it can be highly memory-intensive.

For example, the vectorized maximization routine above is far more memory
intensive than the non-vectorized version that preceded it.

This is because vectorization tends to create many intermediate arrays before
producing the final calculation.

Another issue is that not all algorithms can be vectorized.

In these kinds of settings, we need to go back to loops.

Fortunately, there are alternative ways to speed up Python loops that work in
almost any setting.

For example, [Numba](http://numba.pydata.org/) solves the main problems with
vectorization listed above.

It does so through something called **just in time (JIT) compilation**,
which can generate extremely fast and efficient code.

[Later](numba.md) we'll learn how to use Numba to accelerate Python code.

