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

# OOP I: Objects and Methods

## Overview

The traditional programming paradigm (think Fortran, C, MATLAB, etc.) is called [procedural](https://en.wikipedia.org/wiki/Procedural_programming).

It works as follows

* The program has a state corresponding to the values of its variables.
* Functions are called to act on and transform the state.
* Final outputs are produced via a sequence of function calls.

Two other important paradigms are [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming) (OOP) and [functional programming](https://en.wikipedia.org/wiki/Functional_programming).


In the OOP paradigm, data and functions are bundled together into "objects" --- and functions in this context are referred to as **methods**.

Methods are called on to transform the data contained in the object.

* Think of a Python list that contains data and has methods such as `append()` and `pop()` that transform the data.

Functional programming languages are built on the idea of composing functions.

* Influential examples include [Lisp](https://en.wikipedia.org/wiki/Common_Lisp), [Haskell](https://en.wikipedia.org/wiki/Haskell) and [Elixir](https://en.wikipedia.org/wiki/Elixir_(programming_language)).

So which of these categories does Python fit into?

Actually Python is a pragmatic language that blends object-oriented, functional and procedural styles, rather than taking a purist approach.

On one hand, this allows Python and its users to cherry pick nice aspects of different paradigms.

On the other hand, the lack of purity might at times lead to some confusion.

Fortunately this confusion is minimized if you understand that, at a foundational level, Python *is* object-oriented.

By this we mean that, in Python, *everything is an object*.

In this lecture, we explain what that statement means and why it matters.

We'll make use of the following third party library


```{code-cell} python3
!pip install rich
```


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
:tags: raises-exception

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

Formally, methods are attributes of objects that are **callable** -- i.e., attributes that can be called as functions

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

## Inspection Using Rich

There's a nice package called [rich](https://github.com/Textualize/rich) that
helps us view the contents of an object.

For example,

```{code-cell} python3
from rich import inspect
x = 10
inspect(10)
```
If we want to see the methods as well, we can use

```{code-cell} python3
inspect(10, methods=True)
```

In fact there are still more methods, as you can see if you execute `inspect(10, all=True)`.



## A Little Mystery

In this lecture we claimed that Python is, at heart, an object oriented language.

But here's an example that looks more procedural.

```{code-cell} python3
x = ['a', 'b']
m = len(x)
m
```

If Python is object oriented, why don't we use `x.len()`?    

The answer is related to the fact that Python aims for readability and consistent style.

In Python, it is common for users to build custom objects --- we discuss how to
do this [later](python_oop.md).

It's quite common for users to add methods to their that measure the length of
the object, suitably defined.

When naming such a method, natural choices are `len()` and `length()`.

If some users choose `len()` and others choose `length()`, then the style will
be inconsistent and harder to remember.

To avoid this, the creator of Python chose to add 
`len()` as a built-in function, to help emphasize that `len()` is the convention.

Now, having said all of this, Python *is* still object oriented under the hood.

In fact, the list `x` discussed above has a method called `__len__()`.

All that the function `len()` does is call this method.  

In other words, the following code is equivalent:

```{code-cell} python3
x = ['a', 'b']
len(x)
```
and

```{code-cell} python3
x = ['a', 'b']
x.__len__()
```


## Summary

The message in this lecture is clear:

* In Python, *everything in memory is treated as an object*.

This includes not just lists, strings, etc., but also less obvious things, such as

* functions (once they have been read into memory)
* modules  (ditto)
* files opened for reading or writing
* integers, etc.

Remember that everything is an object will help you interact with your programs
and write clear Pythonic code.

## Exercises

```{exercise-start}
:label: oop_intro_ex1
```

We have met the [boolean data type](python_essentials.md#boolean) previously. 

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

Firstly, we need to find all attributes of `True`, which can be done via

```{code-cell} python3
print(sorted(True.__dir__()))
```

or

```{code-cell} python3
print(sorted(dir(True)))
```

Since the boolean data type is a primitive type, you can also find it in the built-in namespace

```{code-cell} python3
print(dir(__builtins__.bool))
```

Here we use a `for` loop to filter out attributes that are callable

```{code-cell} python3
attributes = dir(__builtins__.bool)
callablels = []

for attribute in attributes:
  # Use eval() to evaluate a string as an expression
  if callable(eval(f'True.{attribute}')):
    callablels.append(attribute)
print(callablels)
```


```{solution-end}
```
