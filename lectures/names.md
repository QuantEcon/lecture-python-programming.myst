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

(oop_names)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Names and Namespaces

## Overview

This lecture is all about variable names, how they can be used and how they are
understood by the Python interpreter.

This might sound a little dull but the model that Python has adopted for
handling names is elegant and interesting.

In addition, you will save yourself many hours of debugging if you have a good
understanding of how names work in Python.

(var_names)=
## Variable Names in Python

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

In Python, `x` is called a **name**, and the statement `x = 42` **binds** the name `x` to the integer object we have just discussed.

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

Here's an example of this situation, where the name `x` is first bound to one object and then **rebound** to another

```{code-cell} python3
x = 'foo'
id(x)
x = 'bar'  
id(x)
```

In this case, after we rebind `x` to `'bar'`, no names bound are to the first object `'foo'`.

This is a trigger for `'foo'` to be garbage collected.

In other words, the memory slot that stores that object is deallocated and returned to the operating system.

Garbage collection is actually an active research area in computer science.

You can [read more on garbage collection](https://rushter.com/blog/python-garbage-collector/) if you are interested.

## Namespaces

```{index} single: Python; Namespaces
```

Recall from the preceding discussion that the statement

```{code-cell} python3
x = 42
```

binds the name `x` to the integer object on the right-hand side.

We also mentioned that this process of binding `x` to the correct object is implemented as a dictionary.

This dictionary is called a namespace.

```{admonition} Definition
A **namespace** is a symbol table that maps names to objects in memory.
```


Python uses multiple namespaces, creating them on the fly as necessary.

For example, every time we import a module, Python creates a namespace for that module.

To see this in action, suppose we write a script `mathfoo.py` with a single line

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

If you wish, you can look at the dictionary directly, using `module_name.__dict__`.

```{code-cell} python3
import math

math.__dict__.items()
```

```{code-cell} python3
import mathfoo

mathfoo.__dict__
```

As you know, we access elements of the namespace using the dotted attribute notation

```{code-cell} python3
math.pi
```

This is entirely equivalent to `math.__dict__['pi']`

```{code-cell} python3
math.__dict__['pi'] 
```

## Viewing Namespaces

As we saw above, the `math` namespace can be printed by typing `math.__dict__`.

Another way to see its contents is to type `vars(math)`

```{code-cell} python3
vars(math).items()
```

If you just want to see the names, you can type

```{code-cell} python3
# Show the first 10 names
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

## Interactive Sessions

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

To see the contents of the namespace of `__main__` we use `vars()` rather than `vars(__main__)`.

If you do this in IPython, you will see a whole lot of variables that IPython
needs, and has initialized when you started up your session.

If you prefer to see only the variables you have initialized, use `%whos`

```{code-cell} ipython
x = 2
y = 3

import numpy as np

%whos
```

## The Global Namespace

```{index} single: Python; Namespace (Global)
```

Python documentation often makes reference to the "global namespace".

The global namespace is *the namespace of the module currently being executed*.

For example, suppose that we start the interpreter and begin making assignments.

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

## Local Namespaces

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

## The `__builtins__` Namespace

```{index} single: Python; Namespace (__builtins__)
```

We have been using various built-in functions, such as `max(), dir(), str(), list(), len(), range(), type()`, etc.

How does access to these names work?

* These definitions are stored in a module called `__builtin__`.
* They have their own namespace called `__builtins__`.

```{code-cell} python3
# Show the first 10 names in `__main__`
dir()[0:10]
```

```{code-cell} python3
# Show the first 10 names in `__builtins__`
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

## Name Resolution

```{index} single: Python; Namespace (Resolution)
```

Namespaces are great because they help us organize variable names.

(Type `import this` at the prompt and look at the last item that's printed)

However, we do need to understand how the Python interpreter works with multiple namespaces.

Understanding the flow of execution will help us to check which variables are in scope and how to operate on them when writing and debugging programs.


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
```

* The function object is created, and `g` is bound to it within the global namespace.
* The name `a` is bound to `0`, again in the global namespace.

```{figure} /_static/lecture_specific/oop_intro/global2.png
```

Next `g` is called via `y = g(10)`, leading to the following sequence of actions

* The local namespace for the function is created.
* Local names `x` and `a` are bound, so that the local namespace becomes `{'x': 10, 'a': 1}`.

Note that the global `a` was not affected by the local `a`.

```{figure} /_static/lecture_specific/oop_intro/local1.png
```


* Statement `x = x + a` uses the local `a` and local `x` to compute `x + a`, and binds local name `x` to the result. 
* This value is returned, and `y` is bound to it in the global namespace.
* Local `x` and `a` are discarded (and the local namespace is deallocated).

```{figure} /_static/lecture_specific/oop_intro/local_return.png
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

```{figure} /_static/lecture_specific/oop_intro/mutable1.png
```

* `x` is bound to `[1]` in the global namespace

```{figure} /_static/lecture_specific/oop_intro/mutable2.png
```

* The call `f(x)`
    * Creates a local namespace
    * Adds `x` to the local namespace, bound to `[1]`

```{figure} /_static/lecture_specific/oop_intro/mutable3.png
```

```{note}
The global `x` and the local `x` refer to the same `[1]`
```

We can see the identity of local `x` and the identity of global `x` are the same

```{code-cell} python3
def f(x):
    x[0] = x[0] + 1
    print(f'the identity of local x is {id(x)}')
    return x

x = [1]
print(f'the identity of global x is {id(x)}')
print(f(x), x)
```

* Within `f(x)`
    * The list `[1]` is modified to `[2]`
    * Returns the list `[2]`

```{figure} /_static/lecture_specific/oop_intro/mutable4.png
```
* The local namespace is deallocated, and the local `x` is lost

```{figure} /_static/lecture_specific/oop_intro/mutable5.png
```

If you want to modify the local `x` and the global `x` separately, you can create a [*copy*](https://docs.python.org/3/library/copy.html) of the list and assign the copy to the local `x`. 

We will leave this for you to explore.



