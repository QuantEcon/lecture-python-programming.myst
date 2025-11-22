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

(speed)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Python for Scientific Computing

```{epigraph}
"We should forget about small efficiencies, say about 97% of the time:
premature optimization is the root of all evil." -- Donald Knuth
```

## Overview

It's probably safe to say that Python is the most popular language for scientific computing.

This is due to 

* the accessible and expressive nature of the language itself,
* the huge range of high quality scientific libraries,
* the fact that the language and libraries are open source,
* the central role that Python plays in data science, machine learning and AI. 

In previous lectures, we used some scientific Python libraries, including NumPy and Matplotlib.

However, our main focus was the core Python language, rather than the libraries.

Now we turn to the scientific libraries and give them our full attention.

In this introductory lecture, we'll discuss the following topics:

1. What are the main elements of the scientific Python ecosystem?
1. How do they fit together?
1. How is the situation changing over time?

In addition to what's in Anaconda, this lecture will need

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install quantecon
```

Let's start with some imports:

```{code-cell} ipython
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
import random
```


## Major Scientific Libraries

Let's briefly review Python's scientific libraries.


### Why do we need them?

One reason we use scientific libraries is because they implement routines we want to use.

* numerical integration, interpolation, linear algebra, root finding, etc.

For example, it's usually better to use an existing routine for root finding than to write a new one from scratch.

(For standard algorithms, efficiency is maximized if the community can
coordinate on a common set of implementations, written by experts and tuned by
users to be as fast and robust as possible!)

But this is not the only reason that we use Python's scientific libraries.

Another is that pure Python is not fast.

So we need libraries that are designed to accelerate execution of Python code.

They do this using two strategies:

1. using compilers that convert Python-like statements into fast machine code for individual threads of logic and
2. parallelizing tasks across multiple "workers" (e.g., CPUs, individual threads inside GPUs).

We will discuss these ideas extensively in this and the remaining lectures from
this series.


### Python's Scientific Ecosystem

At QuantEcon, the scientific libraries we use most often are

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/) 
* [JAX](https://github.com/jax-ml/jax)
* [Pandas](https://pandas.pydata.org/)
* [Numba](https://numba.pydata.org/) 

Here's how they fit together:

* NumPy forms foundations by providing a basic array data type (think of
  vectors and matrices) and functions for acting on these arrays (e.g., matrix
  multiplication).
* SciPy builds on NumPy by adding numerical methods routinely used in science (interpolation, optimization, root finding, etc.).
* Matplotlib is used to generate figures, with a focus on plotting data stored in NumPy arrays.
* JAX includes array processing operations similar to NumPy, automatic
  differentiation, a parallelization-centric just-in-time compiler, and automated integration with hardware accelerators such as
  GPUs.
* Pandas provides types and functions for manipulating data.
* Numba provides a just-in-time compiler that plays well with NumPy and helps accelerate Python code.

We will discuss all of these libraries extensively in this lecture series.


## Pure Python is slow

As mentioned above, one major attraction of the scientific libraries is greater execution speeds.

We will discuss how scientific libraries can help us accelerate code.

For this topic, it will be helpful if we understand what's driving slow execution speeds.


### High vs low level code

Higher-level languages like Python are optimized for humans.

This means that the programmer can leave many details to the runtime environment

* specifying variable types
* memory allocation/deallocation
* etc.

In addition, pure Python is run by an [interpreter](https://en.wikipedia.org/wiki/Interpreter_(computing)), which executes code statement-by-statement.

This makes Python flexible, interactive, easy to write, easy to read, and relatively easy to debug.

On the other hand, the standard implementation of Python (called CPython) cannot
match the speed of compiled languages such as C or Fortran.


### Where are the bottlenecks?

Why is this the case?


#### Dynamic typing

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

This involves a nontrivial overhead.

If we repeatedly execute this expression in a tight loop, the nontrivial
overhead becomes a large overhead.


#### Static types

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

Moreover, when we make a statement such as `int i`, we are making a promise to the compiler
that `i` will *always* be an integer, throughout execution of the program.

As such, the meaning of addition in the expression `sum + i` is completely unambiguous.

There is no need for type-checking and hence no overhead.


### Data Access

Another drag on speed for high-level languages is data access.

To illustrate, let's consider the problem of summing some data --- say, a collection of integers.

#### Summing with Compiled Code

In C or Fortran, these integers would typically be stored in an array, which
is a simple data structure for storing homogeneous data.

Such an array is stored in a single contiguous block of memory

* In modern computers, memory addresses are allocated to each byte (one byte = 8 bits).
* For example, a 64 bit integer is stored in 8 bytes of memory.
* An array of $n$ such integers occupies $8n$ *consecutive* memory slots.

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



### Summary

Does the discussion above mean that we should just switch to C or Fortran for everything?

The answer is: Definitely not!

For any given program, relatively few lines are ever going to be time-critical.

Hence it is far more efficient to write most of our code in a high productivity language like Python.

Moreover, even for those lines of code that *are* time-critical, we can now
equal or outpace binaries compiled from C or Fortran by using Python's scientific libraries.

On that note, we emphasize that, in the last few years, accelerating code has become essentially
synonymous with parallelization.

This task is best left to specialized compilers!

Certain Python libraries have outstanding capabilities for parallelizing scientific code -- we'll discuss this more as we go along.




## Accelerating Python

In this section we look at three related techniques for accelerating Python
code.

Here we'll focus on the fundamental ideas.

Later we'll look at specific libraries and how they implement these ideas.



### {index}`Vectorization <single: Vectorization>`

```{index} single: Python; Vectorization
```

One method for avoiding memory traffic and type checking is [array
programming](https://en.wikipedia.org/wiki/Array_programming). 

Many economists usually refer to array programming as "vectorization."

```{note}
In computer science, this term has [a slightly different meaning](https://en.wikipedia.org/wiki/Automatic_vectorization).
```

The key idea is to send array processing operations in batch to pre-compiled
and efficient native machine code.

The machine code itself is typically compiled from carefully optimized C or Fortran.

For example, when working in a high level language, the operation of inverting a
large matrix can be subcontracted to efficient machine code that is pre-compiled
for this purpose and supplied to users as part of a package.

The core benefits are

1. type-checking is paid *per array*, rather than per element, and
1. arrays containing elements with the same data type are efficient in terms of
   memory access.

The idea of vectorization dates back to MATLAB, which uses vectorization extensively.


```{figure} /_static/lecture_specific/need_for_speed/matlab.png
```



### Vectorization vs for pure Python loops

Let's try a quick speed comparison to illustrate how vectorization can
accelerate code.

Here's some non-vectorized code, which uses a native Python loop to generate,
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

The following vectorized code uses NumPy, which we'll soon investigate in depth,
to achieve the same thing.

```{code-cell} ipython
with qe.Timer():
    x = np.random.uniform(0, 1, n)
    y = np.sum(x**2)
```

As you can see, the second code block runs much faster. 

It breaks the loop down into three basic operations

1. draw `n` uniforms
1. square them
1. sum them

These are sent as batch operators to optimized machine code.




(numba-p_c_vectorization)=
### JIT compilers

At best, vectorization yields fast, simple code.

However, it's not without disadvantages.

One issue is that it can be highly memory-intensive.

This is because vectorization tends to create many intermediate arrays before
producing the final calculation.

Another issue is that not all algorithms can be vectorized.

Because of these issues, most high performance computing is moving away from
traditional vectorization and towards the use of [just-in-time
compilers](https://en.wikipedia.org/wiki/Just-in-time_compilation).

In later lectures in this series, we will learn about how modern Python
libraries exploit just-in-time compilers to generate fast, efficient,
parallelized machine code.




## Parallelization

The growth of CPU clock speed (i.e., the speed at which a single chain of logic
can be run) has slowed dramatically in recent years.

Chip designers and computer programmers have responded to the slowdown by
seeking a different path to fast execution: parallelization.

Hardware makers have increased the number of cores (physical CPUs) embedded in each machine.

For programmers, the challenge has been to exploit these multiple CPUs by
running many processes in parallel (i.e., simultaneously).

This is particularly important in scientific programming, which requires handling

* large amounts of data and
* CPU intensive simulations and other calculations.

Below we discuss parallelization for scientific computing, with a focus on

1. the best tools for parallelization in Python and
1. how these tools can be applied to quantitative economic problems.


### Parallelization on CPUs

Let's review the two main kinds of CPU-based parallelization commonly used in
scientific computing and discuss their pros and cons.


#### Multiprocessing

Multiprocessing means concurrent execution of multiple processes using more than one processor.

In this context, a **process** is a chain of instructions (i.e., a program).

Multiprocessing can be carried out on one machine with multiple CPUs or on a
collection of machines connected by a network.

In the latter case, the collection of machines is usually called a
**cluster**.

With multiprocessing, each process has its own memory space, although the
physical memory chip might be shared.

#### Multithreading

Multithreading is similar to multiprocessing, except that, during execution, the threads all share the same memory space.

Native Python struggles to implement multithreading due to some [legacy design
features](https://wiki.python.org/moin/GlobalInterpreterLock).

But this is not a restriction for scientific libraries like NumPy and Numba.

Functions imported from these libraries and JIT-compiled code run in low level
execution environments where Python's legacy restrictions don't apply.

#### Advantages and Disadvantages

Multithreading is more lightweight because most system and memory resources
are shared by the threads.

In addition, the fact that multiple threads all access a shared pool of memory
is extremely convenient for numerical programming.

On the other hand, multiprocessing is more flexible and can be distributed
across clusters.

For the great majority of what we do in these lectures, multithreading will
suffice.


### Hardware Accelerators

While CPUs with multiple cores have become standard for parallel computing, a
more dramatic shift has occurred with the rise of specialized hardware
accelerators.

These accelerators are designed specifically for the kinds of highly parallel
computations that arise in scientific computing, machine learning, and data
science.

#### GPUs and TPUs

The two most important types of hardware accelerators are

* **GPUs** (Graphics Processing Units) and
* **TPUs** (Tensor Processing Units).

GPUs were originally designed for rendering graphics, which requires performing
the same operation on many pixels simultaneously.

```{figure} /_static/lecture_specific/need_for_speed/geforce.png
:scale: 40
```

Scientists and engineers realized that this same architecture --- many simple
processors working in parallel --- is ideal for scientific computing tasks

TPUs are a more recent development, designed by Google specifically for machine learning workloads.

Like GPUs, TPUs excel at performing massive numbers of matrix operations in parallel.


#### Why TPUs/GPUs Matter 

The performance gains from using hardware accelerators can be dramatic.

For example, a modern GPU can contain thousands of small processing cores,
compared to the 8-64 cores typically found in CPUs.

When a problem can be expressed as many independent operations on arrays of
data, GPUs can be orders of magnitude faster than CPUs.

This is particularly relevant for scientific computing because many algorithms 
naturally map onto the parallel architecture of GPUs.


### Single GPUs vs GPU Servers

There are two common ways to access GPU resources:

#### Single GPU Systems

Many workstations and laptops now come with capable GPUs, or can be equipped with them.

A single modern GPU can dramatically accelerate many scientific computing tasks.

For individual researchers and small projects, a single GPU is often sufficient.

Modern Python libraries like JAX, discussed extensively in this lecture series,
automatically detect and use available GPUs with minimal code changes.


#### Multi-GPU Servers

For larger-scale problems, servers containing multiple GPUs (often 4-8 GPUs per server) are increasingly common.

```{figure} /_static/lecture_specific/need_for_speed/dgx.png
:scale: 40
```


With appropriate software, computations can be distributed across multiple GPUs, either within a single server or across multiple servers.

This enables researchers to tackle problems that would be infeasible on a single GPU or CPU.


### Summary

GPU computing is becoming far more accessible, particularly from within Python.

Some Python scientific libraries, like JAX, now support GPU acceleration with minimal changes to existing code.

We will explore GPU computing in more detail in later lectures, applying it to a
range of economic applications.

