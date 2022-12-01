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

(python_done_right)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Python Essentials

```{contents} Contents
:depth: 2
```

## Overview

We have covered a lot of material quite quickly, with a focus on examples.

Now let's cover some core features of Python in a more systematic way.

This approach is less exciting but helps clear up some details.

## Data Types

```{index} single: Python; Data Types
```

Computer programs typically keep track of a range of data types.

For example, `1.5` is a floating point number, while `1` is an integer.

Programs need to distinguish between these two types for various reasons.

One is that they are stored in memory differently.

Another is that arithmetic operations are different

* For example, floating point arithmetic is implemented on most machines by a
  specialized Floating Point Unit (FPU).

In general, floats are more informative but arithmetic operations on integers
are faster and more accurate.

Python provides numerous other built-in Python data types, some of which we've already met

* strings, lists, etc.

Let's learn a bit more about them.

### Primitive Data Types

(boolean)=
#### Boolean Values

One simple data type is **Boolean values**, which can be either `True` or `False`

```{code-cell} python3
x = True
x
```

We can check the type of any object in memory using the `type()` function.

```{code-cell} python3
type(x)
```

In the next line of code, the interpreter evaluates the expression on the right of = and binds y to this value

```{code-cell} python3
y = 100 < 10
y
```

```{code-cell} python3
type(y)
```

In arithmetic expressions, `True` is converted to `1` and `False` is converted `0`.

This is called **Boolean arithmetic** and is often useful in programming.

Here are some examples

```{code-cell} python3
x + y
```

```{code-cell} python3
x * y
```

```{code-cell} python3
True + True
```

```{code-cell} python3
bools = [True, True, False, True]  # List of Boolean values

sum(bools)
```

#### Numeric Types

Numeric types are also important primitive data types.

We have seen `integer` and `float` types before.

**Complex numbers** are another primitive data type in Python

```{code-cell} python3
x = complex(1, 2)
y = complex(2, 1)
print(x * y)

type(x)
```

### Containers

Python has several basic types for storing collections of (possibly heterogeneous) data.

We've {ref}`already discussed lists <lists_ref>`.

```{index} single: Python; Tuples
```

A related data type is **tuples**, which are "immutable" lists

```{code-cell} python3
x = ('a', 'b')  # Parentheses instead of the square brackets
x = 'a', 'b'    # Or no brackets --- the meaning is identical
x
```

```{code-cell} python3
type(x)
```

In Python, an object is called **immutable** if, once created, the object cannot be changed.

Conversely, an object is **mutable** if it can still be altered after creation.

Python lists are mutable

```{code-cell} python3
x = [1, 2]
x[0] = 10
x
```

But tuples are not

```{code-cell} python3
---
tags: [raises-exception]
---
x = (1, 2)
x[0] = 10
```

We'll say more about the role of mutable and immutable data a bit later.

Tuples (and lists) can be "unpacked" as follows

```{code-cell} python3
integers = (10, 20, 30)
x, y, z = integers
x
```

```{code-cell} python3
y
```

You've actually {ref}`seen an example of this <tuple_unpacking_example>` already.

Tuple unpacking is convenient and we'll use it often.

#### Slice Notation

```{index} single: Python; Slicing
```

To access multiple elements of a sequence (a list, a tuple or a string), you can use Python's slice
notation.

For example,

```{code-cell} python3
a = ["a", "b", "c", "d", "e"]
a[1:]
```

```{code-cell} python3
a[1:3]
```

The general rule is that `a[m:n]` returns `n - m` elements, starting at `a[m]`.

Negative numbers are also permissible

```{code-cell} python3
a[-2:]  # Last two elements of the list
```

You can also use the format `[start:end:step]` to specify the step

```{code-cell} python3
a[::2]
```

Using a negative step, you can return the sequence in a reversed order

```{code-cell} python3
a[-2::-1] # Walk backwards from the second last element to the first element
```

The same slice notation works on tuples and strings

```{code-cell} python3
s = 'foobar'
s[-3:]  # Select the last three elements
```

#### Sets and Dictionaries

```{index} single: Python; Sets
```

```{index} single: Python; Dictionaries
```

Two other container types we should mention before moving on are [sets](https://docs.python.org/3/tutorial/datastructures.html#sets) and [dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries).

Dictionaries are much like lists, except that the items are named instead of
numbered

```{code-cell} python3
d = {'name': 'Frodo', 'age': 33}
type(d)
```

```{code-cell} python3
d['age']
```

The names `'name'` and `'age'` are called the *keys*.

The objects that the keys are mapped to (`'Frodo'` and `33`) are called the `values`.

Sets are unordered collections without duplicates, and set methods provide the
usual set-theoretic operations

```{code-cell} python3
s1 = {'a', 'b'}
type(s1)
```

```{code-cell} python3
s2 = {'b', 'c'}
s1.issubset(s2)
```

```{code-cell} python3
s1.intersection(s2)
```

The `set()` function creates sets from sequences

```{code-cell} python3
s3 = set(('foo', 'bar', 'foo'))
s3
```

## Input and Output

```{index} single: Python; IO
```

Let's briefly review reading and writing to text files, starting with writing

```{code-cell} python3
f = open('newfile.txt', 'w')   # Open 'newfile.txt' for writing
f.write('Testing\n')           # Here '\n' means new line
f.write('Testing again')
f.close()
```

Here

* The built-in function `open()` creates a file object for writing to.
* Both `write()` and `close()` are methods of file objects.

Where is this file that we've created?

Recall that Python maintains a concept of the present working directory (pwd) that can be located from with Jupyter or IPython via

```{code-cell} ipython
%pwd
```

If a path is not specified, then this is where Python writes to.

We can also use Python to read the contents of `newline.txt` as follows

```{code-cell} python3
f = open('newfile.txt', 'r')
out = f.read()
out
```

```{code-cell} python3
print(out)
```

We can also use a `with` statement to ensure the files are properly acquired 
and released.

Containing the operations within the same block also improves the clarity of your code.

```{note}
This kind of block is formally referred to as a [*context*](https://realpython.com/python-with-statement/#the-with-statement-approach).
```

Let's try to convert the two examples above into a `with` statement.

We change the writing example first
```{code-cell} python3

with open('newfile.txt', 'w') as f:  
    f.write('Testing\n')         
    f.write('Testing again')
```

Note that we do not need to call the `close()` method since the `with` block
will ensure the stream is closed at the end of the block.

With slight modifications, we can also read files using `with`

```{code-cell} python3
with open('newfile.txt', 'r') as fo:
    out = fo.read()
    print(out)
```
Now suppose that we want to read input from one file and write output to another. 
Here's how we could accomplish this task while correctly acquiring and returning 
resources to the operating system using `with` statements:

```{code-cell} python3
with open("newfile.txt", "r") as f:
    file = f.readlines()
    with open("output.txt", "w") as fo:
        for i, line in enumerate(file):
            fo.write(f'Line {i}: {line} \n')
```

The output file will be

```{code-cell} python3
with open('output.txt', 'r') as fo:
    print(fo.read())
```

We can simplify the example above by grouping the two `with` statements into one line

```{code-cell} python3
with open("newfile.txt", "r") as f, open("output2.txt", "w") as fo:
        for i, line in enumerate(f):
            fo.write(f'Line {i}: {line} \n')
```

The output file will be the same

```{code-cell} python3
with open('output2.txt', 'r') as fo:
    print(fo.read())
```

Suppose we want to continue to write into the existing file 
instead of overwriting it.

we can switch the mode to `a` which stands for append mode

```{code-cell} python3
with open('output2.txt', 'a') as fo:
    fo.write('\nThis is the end of the file')
```

```{code-cell} python3
with open('output2.txt', 'r') as fo:
    print(fo.read())
```

```{note}
Note that we only covered `r`, `w`, and `a` mode here, which are the most commonly used modes. 
Python provides [a variety of modes](https://www.geeksforgeeks.org/reading-writing-text-files-python/) 
that you could experiment with.
```

### Paths

```{index} single: Python; Paths
```

Note that if `newfile.txt` is not in the present working directory then this call to `open()` fails.

In this case, you can shift the file to the pwd or specify the [full path](https://en.wikipedia.org/wiki/Path_%28computing%29) to the file

```{code-block} python3
:class: no-execute

f = open('insert_full_path_to_file/newfile.txt', 'r')
```

(iterating_version_1)=
## Iterating

```{index} single: Python; Iteration
```

One of the most important tasks in computing is stepping through a
sequence of data and performing a given action.

One of Python's strengths is its simple, flexible interface to this kind of iteration via
the `for` loop.

### Looping over Different Objects

Many Python objects are "iterable", in the sense that they can be looped over.

To give an example, let's write the file us_cities.txt, which lists US cities and their population, to the present working directory.

(us_cities_data)=
```{code-cell} ipython
%%writefile us_cities.txt
new york: 8244910
los angeles: 3819702
chicago: 2707120
houston: 2145146
philadelphia: 1536471
phoenix: 1469471
san antonio: 1359758
san diego: 1326179
dallas: 1223229
```

Here `%%writefile` is an [IPython cell magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cell-magics).

Suppose that we want to make the information more readable, by capitalizing names and adding commas to mark thousands.

The program below reads the data in and makes the conversion:

```{code-cell} python3
data_file = open('us_cities.txt', 'r')
for line in data_file:
    city, population = line.split(':')         # Tuple unpacking
    city = city.title()                        # Capitalize city names
    population = f'{int(population):,}'        # Add commas to numbers
    print(city.ljust(15) + population)
data_file.close()
```

Here `format()` is a string method [used for inserting variables into strings](https://docs.python.org/3/library/string.html#formatspec).

The reformatting of each line is the result of three different string methods,
the details of which can be left till later.

The interesting part of this program for us is line 2, which shows that

1. The file object `data_file` is iterable, in the sense that it can be placed to the right of `in` within a `for` loop.
1. Iteration steps through each line in the file.

This leads to the clean, convenient syntax shown in our program.

Many other kinds of objects are iterable, and we'll discuss some of them later on.

### Looping without Indices

One thing you might have noticed is that Python tends to favor looping without explicit indexing.

For example,

```{code-cell} python3
x_values = [1, 2, 3]  # Some iterable x
for x in x_values:
    print(x * x)
```

is preferred to

```{code-cell} python3
for i in range(len(x_values)):
    print(x_values[i] * x_values[i])
```

When you compare these two alternatives, you can see why the first one is preferred.

Python provides some facilities to simplify looping without indices.

One is `zip()`, which is used for stepping through pairs from two sequences.

For example, try running the following code

```{code-cell} python3
countries = ('Japan', 'Korea', 'China')
cities = ('Tokyo', 'Seoul', 'Beijing')
for country, city in zip(countries, cities):
    print(f'The capital of {country} is {city}')
```

The `zip()` function is also useful for creating dictionaries --- for
example

```{code-cell} python3
names = ['Tom', 'John']
marks = ['E', 'F']
dict(zip(names, marks))
```

If we actually need the index from a list, one option is to use `enumerate()`.

To understand what `enumerate()` does, consider the following example

```{code-cell} python3
letter_list = ['a', 'b', 'c']
for index, letter in enumerate(letter_list):
    print(f"letter_list[{index}] = '{letter}'")
```
(list_comprehensions)=
### List Comprehensions

```{index} single: Python; List comprehension
```

We can also simplify the code for generating the list of random draws considerably by using something called a *list comprehension*.

[List comprehensions](https://en.wikipedia.org/wiki/List_comprehension) are an elegant Python tool for creating lists.

Consider the following example, where the list comprehension is on the
right-hand side of the second line

```{code-cell} python3
animals = ['dog', 'cat', 'bird']
plurals = [animal + 's' for animal in animals]
plurals
```

Here's another example

```{code-cell} python3
range(8)
```

```{code-cell} python3
doubles = [2 * x for x in range(8)]
doubles
```

## Comparisons and Logical Operators

### Comparisons

```{index} single: Python; Comparison
```

Many different kinds of expressions evaluate to one of the Boolean values (i.e., `True` or `False`).

A common type is comparisons, such as

```{code-cell} python3
x, y = 1, 2
x < y
```

```{code-cell} python3
x > y
```

One of the nice features of Python is that we can *chain* inequalities

```{code-cell} python3
1 < 2 < 3
```

```{code-cell} python3
1 <= 2 <= 3
```

As we saw earlier, when testing for equality we use `==`

```{code-cell} python3
x = 1    # Assignment
x == 2   # Comparison
```

For "not equal" use `!=`

```{code-cell} python3
1 != 2
```

Note that when testing conditions, we can use **any** valid Python expression

```{code-cell} python3
x = 'yes' if 42 else 'no'
x
```

```{code-cell} python3
x = 'yes' if [] else 'no'
x
```

What's going on here?

The rule is:

* Expressions that evaluate to zero, empty sequences or containers (strings, lists, etc.) and `None` are all equivalent to `False`.
    * for example, `[]` and `()` are equivalent to `False` in an `if` clause
* All other values are equivalent to `True`.
    * for example, `42` is equivalent to `True` in an `if` clause

### Combining Expressions

```{index} single: Python; Logical Expressions
```

We can combine expressions using `and`, `or` and `not`.

These are the standard logical connectives (conjunction, disjunction and denial)

```{code-cell} python3
1 < 2 and 'f' in 'foo'
```

```{code-cell} python3
1 < 2 and 'g' in 'foo'
```

```{code-cell} python3
1 < 2 or 'g' in 'foo'
```

```{code-cell} python3
not True
```

```{code-cell} python3
not not True
```

Remember

* `P and Q` is `True` if both are `True`, else `False`
* `P or Q` is `False` if both are `False`, else `True`

We can also use `all()` and `any()` to test a sequence of expressions

```{code-cell} python3
all([1 <= 2 <= 3, 5 <= 6 <= 7])
```
```{code-cell} python3
all([1 <= 2 <= 3, "a" in "letter"])
```
```{code-cell} python3
any([1 <= 2 <= 3, "a" in "letter"])
```

```{note}
* `all()` returns `True` when *all* boolean values/expressions in the sequence are `True`
* `any()` returns `True` when *any* boolean values/expressions in the sequence are `True`
```

## Coding Style and Documentation

A consistent coding style and the use of 
documentation can make the code easier to understand and maintain.

### Python Style Guidelines: PEP8

```{index} single: Python; PEP8
```

You can find Python programming philosophy by typing `import this` at the prompt.

Among other things, Python strongly favors consistency in programming style.

We've all heard the saying about consistency and little minds.

In programming, as in mathematics, the opposite is true

* A mathematical paper where the symbols $\cup$ and $\cap$ were
  reversed would be very hard to read, even if the author told you so on the
  first page.

In Python, the standard style is set out in [PEP8](https://www.python.org/dev/peps/pep-0008/).

(Occasionally we'll deviate from PEP8 in these lectures to better match mathematical notation)

(Docstrings)=
### Docstrings

```{index} single: Python; Docstrings
```

Python has a system for adding comments to modules, classes, functions, etc. called *docstrings*.

The nice thing about docstrings is that they are available at run-time.

Try running this

```{code-cell} python3
def f(x):
    """
    This function squares its argument
    """
    return x**2
```

After running this code, the docstring is available

```{code-cell} ipython
f?
```

```{code-block} ipython
:class: no-execute

Type:       function
String Form:<function f at 0x2223320>
File:       /home/john/temp/temp.py
Definition: f(x)
Docstring:  This function squares its argument
```

```{code-cell} ipython
f??
```

```{code-block} ipython
:class: no-execute

Type:       function
String Form:<function f at 0x2223320>
File:       /home/john/temp/temp.py
Definition: f(x)
Source:
def f(x):
    """
    This function squares its argument
    """
    return x**2
```

With one question mark we bring up the docstring, and with two we get the source code as well.

You can find conventions for docstrings in [PEP257](https://peps.python.org/pep-0257/).

## Exercises

Solve the following exercises.

(For some, the built-in function `sum()` comes in handy).

```{exercise-start}
:label: pyess_ex1
```
Part 1: Given two numeric lists or tuples `x_vals` and `y_vals` of equal length, compute
their inner product using `zip()`.

Part 2: In one line, count the number of even numbers in 0,...,99.

Part 3: Given `pairs = ((2, 5), (4, 2), (9, 8), (12, 10))`, count the number of pairs `(a, b)`
such that both `a` and `b` are even.

```{hint}
:class: dropdown

`x % 2` returns 0 if `x` is even, 1 otherwise.

```

```{exercise-end}
```


```{solution-start} pyess_ex1
:class: dropdown
```

**Part 1 Solution:**

Here's one possible solution

```{code-cell} python3
x_vals = [1, 2, 3]
y_vals = [1, 1, 1]
sum([x * y for x, y in zip(x_vals, y_vals)])
```

This also works

```{code-cell} python3
sum(x * y for x, y in zip(x_vals, y_vals))
```

**Part 2 Solution:**

One solution is

```{code-cell} python3
sum([x % 2 == 0 for x in range(100)])
```

This also works:

```{code-cell} python3
sum(x % 2 == 0 for x in range(100))
```

Some less natural alternatives that nonetheless help to illustrate the
flexibility of list comprehensions are

```{code-cell} python3
len([x for x in range(100) if x % 2 == 0])
```

and

```{code-cell} python3
sum([1 for x in range(100) if x % 2 == 0])
```

**Part 3 Solution:**

Here's one possibility

```{code-cell} python3
pairs = ((2, 5), (4, 2), (9, 8), (12, 10))
sum([x % 2 == 0 and y % 2 == 0 for x, y in pairs])
```

```{solution-end}
```

```{exercise-start}
:label: pyess_ex2
```

Consider the polynomial

```{math}
:label: polynom0

p(x)
= a_0 + a_1 x + a_2 x^2 + \cdots a_n x^n
= \sum_{i=0}^n a_i x^i
```

Write a function `p` such that `p(x, coeff)` that computes the value in {eq}`polynom0` given a point `x` and a list of coefficients `coeff` ($a_1, a_2, \cdots a_n$).

Try to use `enumerate()` in your loop.

```{exercise-end}
```

```{solution-start} pyess_ex2
:class: dropdown
```
Here’s a solution:

```{code-cell} python3
def p(x, coeff):
    return sum(a * x**i for i, a in enumerate(coeff))
```

```{code-cell} python3
p(1, (2, 4))
```

```{solution-end}
```


```{exercise-start}
:label: pyess_ex3
```

Write a function that takes a string as an argument and returns the number of capital letters in the string.

```{hint}
:class: dropdown

`'foo'.upper()` returns `'FOO'`.

```

```{exercise-end}
```

```{solution-start} pyess_ex3
:class: dropdown
```

Here's one solution:

```{code-cell} python3
def f(string):
    count = 0
    for letter in string:
        if letter == letter.upper() and letter.isalpha():
            count += 1
    return count

f('The Rain in Spain')
```

An alternative, more pythonic solution:

```{code-cell} python3
def count_uppercase_chars(s):
    return sum([c.isupper() for c in s])

count_uppercase_chars('The Rain in Spain')
```

```{solution-end}
```



```{exercise}
:label: pyess_ex4

Write a function that takes two sequences `seq_a` and `seq_b` as arguments and
returns `True` if every element in `seq_a` is also an element of `seq_b`, else
`False`.

* By "sequence" we mean a list, a tuple or a string.
* Do the exercise without using [sets](https://docs.python.org/3/tutorial/datastructures.html#sets) and set methods.
```

```{solution-start} pyess_ex4
:class: dropdown
```

Here's a solution:

```{code-cell} python3
def f(seq_a, seq_b):
    for a in seq_a:
        if a not in seq_b:
            return False
    return True

# == test == #
print(f("ab", "cadb"))
print(f("ab", "cjdb"))
print(f([1, 2], [1, 2, 3]))
print(f([1, 2, 3], [1, 2]))
```

An alternative, more pythonic solution using `all()`:

```{code-cell} python3
def f(seq_a, seq_b):
  return all([i in seq_b for i in seq_a])

# == test == #
print(f("ab", "cadb"))
print(f("ab", "cjdb"))
print(f([1, 2], [1, 2, 3]))
print(f([1, 2, 3], [1, 2]))
```

Of course, if we use the `sets` data type then the solution is easier

```{code-cell} python3
def f(seq_a, seq_b):
    return set(seq_a).issubset(set(seq_b))
```

```{solution-end}
```


```{exercise}
:label: pyess_ex5

When we cover the numerical libraries, we will see they include many
alternatives for interpolation and function approximation.

Nevertheless, let's write our own function approximation routine as an exercise.

In particular, without using any imports, write a function `linapprox` that takes as arguments

* A function `f` mapping some interval $[a, b]$ into $\mathbb R$.
* Two scalars `a` and `b` providing the limits of this interval.
* An integer `n` determining the number of grid points.
* A number `x` satisfying `a <= x <= b`.

and returns the [piecewise linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation) of `f` at `x`, based on `n` evenly spaced grid points `a = point[0] < point[1] < ... < point[n-1] = b`.

Aim for clarity, not efficiency.
```

```{solution-start} pyess_ex5
:class: dropdown
```
Here’s a solution:

```{code-cell} python3
def linapprox(f, a, b, n, x):
    """
    Evaluates the piecewise linear interpolant of f at x on the interval
    [a, b], with n evenly spaced grid points.

    Parameters
    ==========
        f : function
            The function to approximate

        x, a, b : scalars (floats or integers)
            Evaluation point and endpoints, with a <= x <= b

        n : integer
            Number of grid points

    Returns
    =======
        A float. The interpolant evaluated at x

    """
    length_of_interval = b - a
    num_subintervals = n - 1
    step = length_of_interval / num_subintervals

    # === find first grid point larger than x === #
    point = a
    while point <= x:
        point += step

    # === x must lie between the gridpoints (point - step) and point === #
    u, v = point - step, point

    return f(u) + (x - u) * (f(v) - f(u)) / (v - u)
```

```{solution-end}
```


```{exercise-start}
:label: pyess_ex6
```

Using list comprehension syntax, we can simplify the loop in the following
code.

```{code-cell} python3
import numpy as np

n = 100
ϵ_values = []
for i in range(n):
    e = np.random.randn()
    ϵ_values.append(e)
```

```{exercise-end}
```

```{solution-start} pyess_ex6
:class: dropdown
```

Here's one solution.

```{code-cell} python3
n = 100
ϵ_values = [np.random.randn() for i in range(n)]
```

```{solution-end}
```