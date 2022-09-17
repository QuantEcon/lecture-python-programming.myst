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

(oop_intro)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

<style>
  .auto {
    width: 100%;
    height: auto;
    } 
</style>

# OOP I: Objects and Names

```{contents} Contents
:depth: 2
```

## Overview

[Object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming) (OOP) is one of the major paradigms in programming.

The traditional programming paradigm (think Fortran, C, MATLAB, etc.) is called *procedural*.

It works as follows

* The program has a state corresponding to the values of its variables.
* Functions are called to act on these data.
* Data are passed back and forth via function calls.

In contrast, in the OOP paradigm

* data and functions are "bundled together" into "objects"

(Functions in this context are referred to as **methods**)

### Python and OOP

Python is a pragmatic language that blends object-oriented and procedural styles, rather than taking a purist approach.

However, at a foundational level, Python *is* object-oriented.

In particular, in Python, *everything is an object*.

In this lecture, we explain what that statement means and why it matters.

## Objects

```{index} single: Python; Objects
```

In Python, an *object* is a collection of data and instructions held in computer memory that consists of

1. a type
1. a unique identity
1. data (i.e., content)
1. methods

These concepts are defined and discussed sequentially below.

(type)=
### Type

```{index} single: Python; Type
```

Python provides for different types of objects, to accommodate different categories of data.

For example

```{code-cell} python3
s = 'This is a string'
type(s)
```

```{code-cell} python3
x = 42   # Now let's create an integer
type(x)
```

The type of an object matters for many expressions.

For example, the addition operator between two strings means concatenation

```{code-cell} python3
'300' + 'cc'
```

On the other hand, between two numbers it means ordinary addition

```{code-cell} python3
300 + 400
```

Consider the following expression

```{code-cell} python3
---
tags: [raises-exception]
---
'300' + 400
```

Here we are mixing types, and it's unclear to Python whether the user wants to

* convert `'300'` to an integer and then add it to `400`, or
* convert `400` to string and then concatenate it with `'300'`

Some languages might try to guess but Python is *strongly typed*

* Type is important, and implicit type conversion is rare.
* Python will respond instead by raising a `TypeError`.

To avoid the error, you need to clarify by changing the relevant type.

For example,

```{code-cell} python3
int('300') + 400   # To add as numbers, change the string to an integer
```

(identity)=
### Identity

```{index} single: Python; Identity
```

In Python, each object has a unique identifier, which helps Python (and us) keep track of the object.

The identity of an object can be obtained via the `id()` function

```{code-cell} python3
y = 2.5
z = 2.5
id(y)
```

```{code-cell} python3
id(z)
```

In this example, `y` and `z` happen to have the same value (i.e., `2.5`), but they are not the same object.

The identity of an object is in fact just the address of the object in memory.

### Object Content: Data and Attributes

```{index} single: Python; Content
```

If we set `x = 42` then we create an object of type `int` that contains
the data `42`.

In fact, it contains more, as the following example shows

```{code-cell} python3
x = 42
x
```

```{code-cell} python3
x.imag
```

```{code-cell} python3
x.__class__
```

When Python creates this integer object, it stores with it various auxiliary information, such as the imaginary part, and the type.

Any name following a dot is called an *attribute* of the object to the left of the dot.

* e.g.,``imag`` and `__class__` are attributes of `x`.

We see from this example that objects have attributes that contain auxiliary information.

They also have attributes that act like functions, called *methods*.

These attributes are important, so let's discuss them in-depth.

(methods)=
### Methods

```{index} single: Python; Methods
```

Methods are *functions that are bundled with objects*.

Formally, methods are attributes of objects that are callable (i.e., can be called as functions)

```{code-cell} python3
x = ['foo', 'bar']
callable(x.append)
```

```{code-cell} python3
callable(x.__doc__)
```

Methods typically act on the data contained in the object they belong to, or combine that data with other data

```{code-cell} python3
x = ['a', 'b']
x.append('c')
s = 'This is a string'
s.upper()
```

```{code-cell} python3
s.lower()
```

```{code-cell} python3
s.replace('This', 'That')
```

A great deal of Python functionality is organized around method calls.

For example, consider the following piece of code

```{code-cell} python3
x = ['a', 'b']
x[0] = 'aa'  # Item assignment using square bracket notation
x
```

It doesn't look like there are any methods used here, but in fact the square bracket assignment notation is just a convenient interface to a method call.

What actually happens is that Python calls the `__setitem__` method, as follows

```{code-cell} python3
x = ['a', 'b']
x.__setitem__(0, 'aa')  # Equivalent to x[0] = 'aa'
x
```

(If you wanted to you could modify the `__setitem__` method, so that square bracket assignment does something totally different)


(name_res)=
## Names and Name Resolution

### Variable Names in Python

```{index} single: Python; Variable Names
```

Consider the Python statement

```{code-cell} python3
x = 42
```

We now know that when this statement is executed, Python creates an object of
type `int` in your computer's memory, containing

* the value `42`
* some associated attributes

But what is `x` itself?

In Python, `x` is called a *name*, and the statement `x = 42` *binds* the name `x` to the integer object we have just discussed.

Under the hood, this process of binding names to objects is implemented as a dictionary---more about this in a moment.

There is no problem binding two or more names to the one object, regardless of what that object is

```{code-cell} python3
def f(string):      # Create a function called f
    print(string)   # that prints any string it's passed

g = f
id(g) == id(f)
```

```{code-cell} python3
g('test')
```

In the first step, a function object is created, and the name `f` is bound to it.

After binding the name `g` to the same object, we can use it anywhere we would use `f`.

What happens when the number of names bound to an object goes to zero?

Here's an example of this situation, where the name `x` is first bound to one object and then rebound to another

```{code-cell} python3
x = 'foo'
id(x)
```

```{code-cell} python3
x = 'bar'  # No names bound to the first object
```

What happens here is that the first object is garbage collected.

In other words, the memory slot that stores that object is deallocated, and returned to the operating system.

Garbage collection is actually an active research area in computer science.

You can [read more on garbage collection](https://rushter.com/blog/python-garbage-collector/) if you are interested.

### Namespaces

```{index} single: Python; Namespaces
```

Recall from the preceding discussion that the statement

```{code-cell} python3
x = 42
```

binds the name `x` to the integer object on the right-hand side.

We also mentioned that this process of binding `x` to the correct object is implemented as a dictionary.

This dictionary is called a *namespace*.

**Definition:** A namespace is a symbol table that maps names to objects in memory.

Python uses multiple namespaces, creating them on the fly as necessary .

For example, every time we import a module, Python creates a namespace for that module.

To see this in action, suppose we write a script `math2.py` with a single line

```{code-cell} python3
%%file mathfoo.py
pi = 'foobar'
```

Now we start the Python interpreter and import it

```{code-cell} python3
import mathfoo
```

Next let's import the `math` module from the standard library

```{code-cell} python3
import math
```

Both of these modules have an attribute called `pi`

```{code-cell} python3
math.pi
```

```{code-cell} python3
mathfoo.pi
```

These two different bindings of `pi` exist in different namespaces, each one implemented as a dictionary.

We can look at the dictionary directly, using `module_name.__dict__`

```{code-cell} python3
import math

math.__dict__.items()
```

```{code-cell} python3
import mathfoo

mathfoo.__dict__.items()
```

As you know, we access elements of the namespace using the dotted attribute notation

```{code-cell} python3
math.pi
```

In fact this is entirely equivalent to `math.__dict__['pi']`

```{code-cell} python3
math.__dict__['pi'] == math.pi
```

### Viewing Namespaces

As we saw above, the `math` namespace can be printed by typing `math.__dict__`.

Another way to see its contents is to type `vars(math)`

```{code-cell} python3
vars(math).items()
```

If you just want to see the names, you can type

```{code-cell} python3
dir(math)[0:10]
```

Notice the special names `__doc__` and `__name__`.

These are initialized in the namespace when any module is imported

* `__doc__` is the doc string of the module
* `__name__` is the name of the module

```{code-cell} python3
print(math.__doc__)
```

```{code-cell} python3
math.__name__
```

### Interactive Sessions

```{index} single: Python; Interpreter
```

In Python, **all** code executed by the interpreter runs in some module.

What about commands typed at the prompt?

These are also regarded as being executed within a module --- in this case, a module called `__main__`.

To check this, we can look at the current module name via the value of `__name__` given at the prompt

```{code-cell} python3
print(__name__)
```

When we run a script using IPython's `run` command, the contents of the file are executed as part of `__main__` too.

To see this, let's create a file `mod.py` that prints its own `__name__` attribute

```{code-cell} ipython
%%file mod.py
print(__name__)
```

Now let's look at two different ways of running it in IPython

```{code-cell} python3
import mod  # Standard import
```

```{code-cell} ipython
%run mod.py  # Run interactively
```

In the second case, the code is executed as part of `__main__`, so `__name__` is equal to `__main__`.

To see the contents of the namespace of `__main__` we use `vars()` rather than `vars(__main__)` .

If you do this in IPython, you will see a whole lot of variables that IPython
needs, and has initialized when you started up your session.

If you prefer to see only the variables you have initialized, use `whos`

```{code-cell} ipython
x = 2
y = 3

import numpy as np

%whos
```

### The Global Namespace

```{index} single: Python; Namespace (Global)
```

Python documentation often makes reference to the "global namespace".

The global namespace is *the namespace of the module currently being executed*.

For example, suppose that we start the interpreter and begin making assignments .

We are now working in the module `__main__`, and hence the namespace for `__main__` is the global namespace.

Next, we import a module called `amodule`

```{code-block} python3
:class: no-execute

import amodule
```

At this point, the interpreter creates a namespace for the module `amodule` and starts executing commands in the module.

While this occurs, the namespace `amodule.__dict__` is the global namespace.

Once execution of the module finishes, the interpreter returns to the module from where the import statement was made.

In this case it's `__main__`, so the namespace of `__main__` again becomes the global namespace.

### Local Namespaces

```{index} single: Python; Namespace (Local)
```

Important fact: When we call a function, the interpreter creates a *local namespace* for that function, and registers the variables in that namespace.

The reason for this will be explained in just a moment.

Variables in the local namespace are called *local variables*.

After the function returns, the namespace is deallocated and lost.

While the function is executing, we can view the contents of the local namespace with `locals()`.

For example, consider

```{code-cell} python3
def f(x):
    a = 2
    print(locals())
    return a * x
```

Now let's call the function

```{code-cell} python3
f(1)
```

You can see the local namespace of `f` before it is destroyed.

### The `__builtins__` Namespace

```{index} single: Python; Namespace (__builtins__)
```

We have been using various built-in functions, such as `max(), dir(), str(), list(), len(), range(), type()`, etc.

How does access to these names work?

* These definitions are stored in a module called `__builtin__`.
* They have their own namespace called `__builtins__`.

```{code-cell} python3
dir()[0:10]
```

```{code-cell} python3
dir(__builtins__)[0:10]
```

We can access elements of the namespace as follows

```{code-cell} python3
__builtins__.max
```

But `__builtins__` is special, because we can always access them directly as well

```{code-cell} python3
max
```

```{code-cell} python3
__builtins__.max == max
```

The next section explains how this works ...

### Name Resolution

```{index} single: Python; Namespace (Resolution)
```

Namespaces are great because they help us organize variable names.

(Type `import this` at the prompt and look at the last item that's printed)

However, we do need to understand how the Python interpreter works with multiple namespaces .

At any point of execution, there are in fact at least two namespaces that can be accessed directly.

("Accessed directly" means without using a dot, as in  `pi` rather than `math.pi`)

These namespaces are

* The global namespace (of the module being executed)
* The builtin namespace

If the interpreter is executing a function, then the directly accessible namespaces are

* The local namespace of the function
* The global namespace (of the module being executed)
* The builtin namespace

Sometimes functions are defined within other functions, like so

```{code-cell} python3
def f():
    a = 2
    def g():
        b = 4
        print(a * b)
    g()
```

Here `f` is the *enclosing function* for `g`, and each function gets its
own namespaces.

Now we can give the rule for how namespace resolution works:

The order in which the interpreter searches for names is

1. the local namespace (if it exists)
1. the hierarchy of enclosing namespaces (if they exist)
1. the global namespace
1. the builtin namespace

If the name is not in any of these namespaces, the interpreter raises a `NameError`.

This is called the **LEGB rule** (local, enclosing, global, builtin).

Here's an example that helps to illustrate.

Visualizations here are created by [nbtutor](https://github.com/lgpage/nbtutor) in a Jupyter notebook.

They can help you better understand your program when you are learning a new language.

Consider a script `test.py` that looks as follows

```{code-cell} python3
%%file test.py
def g(x):
    a = 1
    x = x + a
    return x

a = 0
y = g(10)
print("a = ", a, "y = ", y)
```

What happens when we run this script?

```{code-cell} ipython
%run test.py
```

First,

* The global namespace `{}` is created.

```{figure} /_static/lecture_specific/oop_intro/global.png
:figclass: auto
```

* The function object is created, and `g` is bound to it within the global namespace.
* The name `a` is bound to `0`, again in the global namespace.

```{figure} /_static/lecture_specific/oop_intro/global2.png
:figclass: auto
```

Next `g` is called via `y = g(10)`, leading to the following sequence of actions

* The local namespace for the function is created.
* Local names `x` and `a` are bound, so that the local namespace becomes `{'x': 10, 'a': 1}`.

```{figure} /_static/lecture_specific/oop_intro/local1.png
:figclass: auto
```

> Note that the global `a` was not affected by the local `a`.


* Statement `x = x + a` uses the local `a` and local `x` to compute `x + a`, and binds local name `x` to the result. 


* This value is returned, and `y` is bound to it in the global namespace.
* Local `x` and `a` are discarded (and the local namespace is deallocated).

```{figure} /_static/lecture_specific/oop_intro/local1.png
:figclass: auto
```


(mutable_vs_immutable)=
### {index}`Mutable <single: Mutable>` Versus {index}`Immutable <single: Immutable>` Parameters

This is a good time to say a little more about mutable vs immutable objects.

Consider the code segment

```{code-cell} python3
def f(x):
    x = x + 1
    return x

x = 1
print(f(x), x)
```

We now understand what will happen here: The code prints `2` as the value of `f(x)` and `1` as the value of `x`.

First `f` and `x` are registered in the global namespace.

The call `f(x)` creates a local namespace and adds `x` to it, bound to `1`.

Next, this local `x` is rebound to the new integer object `2`, and this value is returned.

None of this affects the global `x`.

However, it's a different story when we use a **mutable** data type such as a list

```{code-cell} python3
def f(x):
    x[0] = x[0] + 1
    return x

x = [1]
print(f(x), x)
```

This prints `[2]` as the value of `f(x)` and *same* for `x`.

Here's what happens

* `f` is registered as a function in the global namespace
* `x` bound to `[1]` in the global namespace
* The call `f(x)`
    * Creates a local namespace
    * Adds `x` to local namespace, bound to `[1]`
    * The list `[1]` is modified to `[2]`
    * Returns the list `[2]`
    * The local namespace is deallocated, and local `x` is lost
* Global `x` has been modified



## Summary

Messages in this lecture are clear:

  * In Python, *everything in memory is treated as an object*.
  * Zero, one or many names can be bound to a given object.
  * Every name resides within a scope defined by its namespace.

This includes not just lists, strings, etc., but also less obvious things, such as

* functions (once they have been read into memory)
* modules  (ditto)
* files opened for reading or writing
* integers, etc.

Consider, for example, functions.

When Python reads a function definition, it creates a **function object** and stores it in memory.

The following code illustrates further this idea

```{code-cell} python3
#reset the current namespace
%reset
```

```{code-cell} python3
def f(x): return x**2
f
```

```{code-cell} python3
type(f)
```

```{code-cell} python3
id(f)
```

```{code-cell} python3
f.__name__
```

We can see that `f` has type, identity, attributes and so on---just like any other object.

It also has methods.

One example is the `__call__` method, which just evaluates the function

```{code-cell} python3
f.__call__(3)
```

Another is the `__dir__` method, which returns a list of attributes.

We can also find `f` our current namespace

```{code-cell} python3
'f' in dir()
```

Modules loaded into memory are also treated as objects

```{code-cell} python3
import math

id(math)
```

We can find `math` in our global namespace after the import

```{code-cell} python3
print(dir()[-1::-1])
```

We can also find all objects associated with the `math` module in the private namespace of `math`

```{code-cell} python3
print(dir(math))
```

We can also directly import objects to our current namespace using `from ... import ...`

```{code-cell} python3
from math import log, pi, sqrt

print(dir()[-1::-1])
```

We can find these names appear in the current namespace now.

*This uniform treatment of data in Python (everything is an object) helps keep the language simple and consistent.*

## Exercises

```{exercise-start}
:label: oop_intro_ex1
```

We have met the {any}`boolean data type <boolean>` previously. 
Using what we have learnt in this lecture, print a list of methods of boolean objects.

(hint: you can use `callable()` to test whether an attribute of an object can be called as a function)

```{exercise-end}
```

```{solution-start} oop_intro_ex1
:class: dropdown
```

Firstly, we need to find all attributes of a boolean object.

You can use one of the following ways:

you can call the `.__dir__()` method

```{code-cell} python3
print(sorted(True.__dir__()))
```

you can use the built-in function `dir()`

```{code-cell} python3
print(sorted(dir(True)))
```

or, since the boolean data type is a primitive type, you can also find it in the built-in namespace

```{code-cell} python3
print(dir(__builtins__.bool))
```

Next, we can use a `for` loop to filter out attributes that are callable

```{code-cell} python3
attrls = dir(__builtins__.bool)
callablels = list()

for i in attrls:
  # we use eval() to transform a string into a statement
  if callable(eval('True.'+i)):
    callablels.append(i)
print(callablels)
```

here is a one-line solution

```{code-cell} python3
print([i for i in attrls if callable(eval("True." + i))])
```

You can explore these methods and see what they are used for.

```{solution-end}
```
