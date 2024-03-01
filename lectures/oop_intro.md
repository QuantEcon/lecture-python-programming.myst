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

# OOP I: Objects and Methods

## Overview

The traditional programming paradigm (think Fortran, C, MATLAB, etc.) is called [procedural](https://en.wikipedia.org/wiki/Procedural_programming).

It works as follows

* The program has a state corresponding to the values of its variables.
* Functions are called to act on these data.
* Data are passed back and forth via function calls.

Two other important paradigms are [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming) (OOP) and [functional programming](https://en.wikipedia.org/wiki/Functional_programming).


In the OOP paradigm data and functions are "bundled together" into "objects" (and functions in this context are referred to as **methods**).

* think of a Python list that contains data and exposes methods such as `append()` and `count()`

Functional programming languages are built on the idea of composing functions.

* Influential examples include [Lisp](https://en.wikipedia.org/wiki/Common_Lisp), [Haskell](https://en.wikipedia.org/wiki/Haskell) and [Elixir](https://en.wikipedia.org/wiki/Elixir_(programming_language)).

So which of these categories does Python fit into?

Actually Python is a pragmatic language that blends object-oriented, functional and procedural styles, rather than taking a purist approach.

On one hand, this allows Python and its users to cherry pick nice aspects of different paradigms.

On the other hand, the lack of purity might at times lead to some confusion.

Fortunately this confusion is minimized if you understand that, at a foundational level, Python *is* object-oriented.

By this we mean that, in Python, *everything is an object*.

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


## Summary

Messages in this lecture are clear:

* In Python, *everything in memory is treated as an object*.
* Zero, one or many names can be bound to a given object.

This includes not just lists, strings, etc., but also less obvious things, such as

* functions (once they have been read into memory)
* modules  (ditto)
* files opened for reading or writing
* integers, etc.



## Exercises

```{exercise-start}
:label: oop_intro_ex1
```

We have met the {any}`boolean data type <boolean>` previously. 

Using what we have learnt in this lecture, print a list of methods of the
boolean object `True`.

```{hint}
:class: dropdown

You can use `callable()` to test whether an attribute of an object can be called as a function
```

```{exercise-end}
```

```{solution-start} oop_intro_ex1
:class: dropdown
```

Firstly, we need to find all attributes of a boolean object.

You can use one of the following ways:

*1.* You can call the `.__dir__()` method

```{code-cell} python3
print(sorted(True.__dir__()))
```

*2.* You can use the built-in function `dir()`

```{code-cell} python3
print(sorted(dir(True)))
```

*3.* Since the boolean data type is a primitive type, you can also find it in the built-in namespace

```{code-cell} python3
print(dir(__builtins__.bool))
```

Next, we can use a `for` loop to filter out attributes that are callable

```{code-cell} python3
attrls = dir(__builtins__.bool)
callablels = list()

for i in attrls:
  # Use eval() to evaluate a string as an expression
  if callable(eval(f'True.{i}')):
    callablels.append(i)
print(callablels)
```

Here is a one-line solution

```{code-cell} python3
print([i for i in attrls if callable(eval(f'True.{i}'))])
```


```{solution-end}
```
