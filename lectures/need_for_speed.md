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

Python is popular for scientific computing due to factors such as

* the accessible and expressive nature of the language itself,
* the huge range of high quality scientific libraries,
* the fact that the language and libraries are open source,
* the popular [Anaconda Python distribution](https://www.anaconda.com/download), which simplifies installation and management of scientific libraries, and
* the key role that Python plays in data science, machine learning and artificial intelligence.

In previous lectures, we looked at some scientific Python libraries such as NumPy and Matplotlib.

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

(For standard algorithms, efficiency is maximized if the community can
coordinate on a common set of implementations, written by experts and tuned by
users to be as fast and robust as possible.)

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
* [JAX](https://github.com/jax-ml/jax)
* [Pandas](https://pandas.pydata.org/)
* [Numba](https://numba.pydata.org/) and

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

The answer is: No!

There are three reasons why:

First, for any given program, relatively few lines are ever going to be time-critical.

Hence it is far more efficient to write most of our code in a high productivity language like Python.

Second, even for those lines of code that *are* time-critical, we can now achieve the same speed as C or Fortran using Python's scientific libraries.

Third, in the last few years, accelerating code has become essentially
synonymous with parallelizing execution, and this task is best left to
specialized compilers.

Certain Python libraries have outstanding capabilities for parallelizing
scientific code -- we'll discuss this more as we go along.


### Where are the Bottlenecks?

Before we do so, let's try to understand why plain vanilla Python is slower than C or Fortran.

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

One method for avoiding memory traffic and type checking is [array programming](https://en.wikipedia.org/wiki/Array_programming). 

Economists usually refer to array programming as ``vectorization.''

(In computer science, this term has [a slightly different meaning](https://en.wikipedia.org/wiki/Automatic_vectorization).)

The key idea is to send array processing operations in batch to pre-compiled
and efficient native machine code.

The machine code itself is typically compiled from carefully optimized C or Fortran.

For example, when working in a high level language, the operation of inverting a
large matrix can be subcontracted to efficient machine code that is pre-compiled
for this purpose and supplied to users as part of a package.

This idea dates back to MATLAB, which uses vectorization extensively.


```{figure} /_static/lecture_specific/need_for_speed/matlab.png
```

Vectorization can greatly accelerate many numerical computations, as we will see
in later lectures.

(numba-p_c_vectorization)=
## Beyond Vectorization

At best, vectorization yields fast, simple code.

However, it's not without disadvantages.

One issue is that it can be highly memory-intensive.

This is because vectorization tends to create many intermediate arrays before
producing the final calculation.

Another issue is that not all algorithms can be vectorized.

Because of these issues, most high performance computing is moving away from
traditional vectorization and towards the use of [just-in-time compilers](https://en.wikipedia.org/wiki/Just-in-time_compilation).

In later lectures in this series, we will learn about how modern Python libraries exploit
just-in-time compilers to generate fast, efficient, parallelized machine code.

## Parallelization

The growth of CPU clock speed (i.e., the speed at which a single chain of logic can
be run) has slowed dramatically in recent years.

This is unlikely to change in the near future, due to inherent physical
limitations on the construction of chips and circuit boards.

Chip designers and computer programmers have responded to the slowdown by
seeking a different path to fast execution: parallelization.

Hardware makers have increased the number of cores (physical CPUs) embedded in each machine.

For programmers, the challenge has been to exploit these multiple CPUs by running many processes in parallel (i.e., simultaneously).

This is particularly important in scientific programming, which requires handling

* large amounts of data and
* CPU intensive simulations and other calculations.

In this lecture we discuss parallelization for scientific computing, with a focus on

1. the best tools for parallelization in Python and
1. how these tools can be applied to quantitative economic problems.

Let's start with some imports:

```{code-cell} ipython
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
```

### Parallelization on CPUs

Large textbooks have been written on different approaches to parallelization but we will keep a tight focus on what's most useful to us.

We will briefly review the two main kinds of CPU-based parallelization commonly used in
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

While CPUs with multiple cores have become standard for parallel computing, a more dramatic shift has occurred with the rise of specialized hardware accelerators.

These accelerators are designed specifically for the kinds of highly parallel computations that arise in scientific computing, machine learning, and data science.

#### GPUs and TPUs

The two most important types of hardware accelerators are

* **GPUs** (Graphics Processing Units) and
* **TPUs** (Tensor Processing Units).

GPUs were originally designed for rendering graphics, which requires performing the same operation on many pixels simultaneously.

Scientists and engineers realized that this same architecture --- many simple processors working in parallel --- is ideal for scientific computing tasks such as

* matrix operations,
* numerical simulation,
* solving partial differential equations and
* training machine learning models.

TPUs are a more recent development, designed by Google specifically for machine learning workloads.

Like GPUs, TPUs excel at performing massive numbers of matrix operations in parallel.

#### Why GPUs Matter for Scientific Computing

The performance gains from using GPUs can be dramatic.

A modern GPU can contain thousands of small processing cores, compared to the 8-64 cores typically found in CPUs.

When a problem can be expressed as many independent operations on arrays of data, GPUs can be orders of magnitude faster than CPUs.

This is particularly relevant for scientific computing because many algorithms in

* linear algebra,
* optimization,
* Monte Carlo simulation and
* numerical methods for differential equations

naturally map onto the parallel architecture of GPUs.

#### Single GPUs vs GPU Servers

There are two common ways to access GPU resources:

**Single GPU Systems**

Many workstations and laptops now come with capable GPUs, or can be equipped with them.

```{figure} /_static/lecture_specific/need_for_speed/geforce.png
:scale: 40
```

A single modern GPU can dramatically accelerate many scientific computing tasks.

For individual researchers and small projects, a single GPU is often sufficient.

Python libraries like JAX, PyTorch, and TensorFlow can automatically detect and use available GPUs with minimal code changes.

**Multi-GPU Servers**

For larger-scale problems, servers containing multiple GPUs (often 4-8 GPUs per server) are increasingly common.

```{figure} /_static/lecture_specific/need_for_speed/dgx.png
:scale: 23
```

These can be located

* in local compute clusters,
* in university or national lab computing facilities, or
* in cloud computing platforms (AWS, Google Cloud, Azure, etc.).

With appropriate software, computations can be distributed across multiple GPUs, either within a single server or across multiple servers.

This enables researchers to tackle problems that would be infeasible on a single GPU or CPU.

#### GPU Programming in Python

The good news for Python users is that many scientific libraries now support GPU acceleration with minimal changes to existing code.

For example, JAX code that runs on CPUs can often run on GPUs simply by ensuring the data is placed on the GPU device.

We will explore GPU computing in more detail in later lectures, particularly when we discuss JAX.

