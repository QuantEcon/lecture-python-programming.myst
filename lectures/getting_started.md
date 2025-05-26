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

# Getting Started


## Overview

In this lecture, you will learn how to

1. use Python in the cloud
1. get a local Python environment up and running
1. execute simple Python commands
1. run a sample program
1. install the code libraries that underpin these lectures

## Python in the Cloud

The easiest way to get started coding in Python is by running it in the cloud.

(That is, by using a remote server that already has Python installed.)

One option that's both free and reliable is [Google Colab](https://colab.research.google.com/).

Colab also has the advantage of providing GPUs, which we will make use of in
more advanced lectures.

Tutorials on how to get started with Google Colab can be found by web and video searches.

Most of our lectures include a "Launch notebook" button (with a play icon) on the top
right connects you to an executable version on Colab.


## Local Install

Local installs are preferable if you have access to a suitable machine and
plan to do a substantial amount of Python programming.

At the same time, local installs require more work than a cloud option like Colab.

The rest of this lecture runs you through the some details associated with local installs.


### The Anaconda Distribution

The [core Python package](https://www.python.org/downloads/) is easy to install but *not* what you should choose for these lectures.

These lectures require the entire scientific programming ecosystem, which

* the core installation doesn't provide
* is painful to install one piece at a time.

Hence the best approach for our purposes is to install a Python distribution that contains

1. the core Python language **and**
1. compatible versions of the most popular scientific libraries.

The best such distribution is [Anaconda Python](https://www.anaconda.com/).

Anaconda is

* very popular
* cross-platform
* comprehensive
* completely unrelated to the [Nicki Minaj song of the same name](https://www.youtube.com/watch?v=LDZX4ooRsWs)

Anaconda also comes with a package management system to organize your code libraries.

**All of what follows assumes that you adopt this recommendation!**

(install_anaconda)=
### Installing Anaconda


To install Anaconda, [download](https://www.anaconda.com/download/) the binary and follow the instructions.

Important points:

* Make sure you install the correct version for your OS.
* If you are asked during the installation process whether you'd like to make Anaconda your default Python installation, say yes.

### Updating `conda`

Anaconda supplies a tool called `conda` to manage and upgrade your Anaconda packages.

One `conda` command you should execute regularly is the one that updates the whole Anaconda distribution.

As a practice run, please execute the following

1. Open up a terminal
1. Type `conda update conda`

For more information on conda, type conda help in a terminal.

(ipython_notebook)=
## Jupyter Notebook

[Jupyter](http://jupyter.org/) notebooks are one of the many possible ways to interact with Python and the scientific libraries.

They use  a *browser-based* interface to Python with

* The ability to write and execute Python commands.
* Formatted output in the browser, including tables, figures, animation, etc.
* The option to mix in formatted text and mathematical expressions.

Because of these features, Jupyter is now a major player in the scientific computing ecosystem.

Here's an image showing execution of some code (borrowed from [here](http://matplotlib.org/examples/pylab_examples/hexbin_demo.html)) in a Jupyter notebook

```{figure} /_static/lecture_specific/getting_started/jp_demo.png
:figclass: auto
```

While Jupyter isn't the only way to code in Python, it's great for when you wish to

* start coding in Python
* test new ideas or interact with small pieces of code
* use powerful online interactive environments such as [Google Colab](https://research.google.com/colaboratory/)
* share or collaborate scientific ideas with students or colleagues

These lectures are designed for executing in Jupyter notebooks.

### Starting the Jupyter Notebook


Once you have installed Anaconda, you can start the Jupyter notebook.

Either

* search for Jupyter in your applications menu, or
* open up a terminal and type `jupyter notebook`
    * Windows users should substitute "Anaconda command prompt" for "terminal" in the previous line.

If you use the second option, you will see something like this

```{figure} /_static/lecture_specific/getting_started/starting_nb.png
:figclass: terminal
```

The output tells us the notebook is running at `http://localhost:8888/`

* `localhost` is the name of the local machine
* `8888` refers to [port number](https://en.wikipedia.org/wiki/Port_%28computer_networking%29) 8888 on your computer

Thus, the Jupyter kernel is listening for Python commands on port 8888 of our local machine.

Hopefully, your default browser has also opened up with a web page that looks something like this

```{figure} /_static/lecture_specific/getting_started/nb.png
:figclass: auto
```

What you see here is called the Jupyter *dashboard*.

If you look at the URL at the top, it should be `localhost:8888` or similar, matching the message above.

Assuming all this has worked OK, you can now click on `New` at the top right and select `Python 3` or similar.

Here's what shows up on our machine:

```{figure} /_static/lecture_specific/getting_started/nb2.png
:figclass: auto
```

The notebook displays an *active cell*, into which you can type Python commands.

### Notebook Basics


Let's start with how to edit code and run simple programs.

#### Running Cells

Notice that, in the previous figure, the cell is surrounded by a green border.

This means that the cell is in *edit mode*.

In this mode, whatever you type will appear in the cell with the flashing cursor.

When you're ready to execute the code in a cell, hit `Shift-Enter` instead of the usual `Enter`.

```{figure} /_static/lecture_specific/getting_started/nb3.png
:figclass: auto
```

```{note}
There are also menu and button options for running code in a cell that you can find by exploring.
```

#### Modal Editing

The next thing to understand about the Jupyter notebook is that it uses a *modal* editing system.

This means that the effect of typing at the keyboard **depends on which mode you are in**.

The two modes are

1. Edit mode
    * Indicated by a green border around one cell, plus a blinking cursor
    * Whatever you type appears as is in that cell

1. Command mode
    * The green border is replaced by a blue border
    * Keystrokes are interpreted as commands --- for example, typing `b` adds a new cell below the current one

To switch to

* command mode from edit mode, hit the `Esc` key or `Ctrl-M`
* edit mode from command mode, hit `Enter` or click in a cell

The modal behavior of the Jupyter notebook is very efficient when you get used to it.

#### Inserting Unicode (e.g., Greek Letters)

Python supports [unicode](https://docs.python.org/3/howto/unicode.html), allowing the use of characters such as $\alpha$ and $\beta$ as names in your code.

In a code cell, try typing `\alpha` and then hitting the tab key on your keyboard.

(a_test_program)=
#### A Test Program

Let's run a test program.

Here's an arbitrary program we can use: [http://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/polar_bar.html](http://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/polar_bar.html).

On that page, you'll see the following code

```{code-cell} ipython
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 20
θ = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)
colors = plt.cm.viridis(radii / 10.)

ax = plt.subplot(111, projection='polar')
ax.bar(θ, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

plt.show()
```

Don't worry about the details for now --- let's just run it and see what happens.

The easiest way to run this code is to copy and paste it into a cell in the notebook.

Hopefully you will get a similar plot.

### Working with the Notebook

Here are a few more tips on working with Jupyter notebooks.

#### Tab Completion

In the previous program, we executed the line `import numpy as np`

* NumPy is a numerical library we'll work with in depth.

After this import command, functions in NumPy can be accessed with `np.function_name` type syntax.

* For example, try `np.random.randn(3)`.

We can explore these attributes of `np` using the `Tab` key.

For example, here we type `np.random.r` and hit Tab

```{figure} /_static/lecture_specific/getting_started/nb6.png
:figclass: auto
```

Jupyter offers several possible completions for you to choose from.

In this way, the Tab key helps remind you of what's available and also saves you typing.

(gs_help)=
#### On-Line Help


To get help on `np.random.randn`, we can execute `np.random.randn?`.

Documentation appears in a split window of the browser, like so

```{figure} /_static/lecture_specific/getting_started/nb6a.png
:figclass: auto
```

Clicking on the top right of the lower split closes the on-line help.

We will learn more about how to create documentation like this {ref}`later <Docstrings>`!

#### Other Content

In addition to executing code, the Jupyter notebook allows you to embed text, equations, figures and even videos in the page.

For example, we can enter a mixture of plain text and LaTeX instead of code.

Next we `Esc` to enter command mode and then type `m` to indicate that we
are writing [Markdown](http://daringfireball.net/projects/markdown/), a mark-up language similar to (but simpler than) LaTeX.

(You can also use your mouse to select `Markdown` from the `Code` drop-down box just below the list of menu items)

```{figure} /_static/lecture_specific/getting_started/nb7.png
:figclass: auto
```

Now we `Shift+Enter` to produce this

```{figure} /_static/lecture_specific/getting_started/nb8.png
:figclass: auto
```

### Debugging Code


Debugging is the process of identifying and removing errors from a program. 

You will spend a lot of time debugging code, so it is important to [learn how to do it effectively](https://www.freecodecamp.org/news/what-is-debugging-how-to-debug-code/).

If you are using a newer version of Jupyter, you should see a bug icon on the right end of the toolbar.

```{figure} /_static/lecture_specific/getting_started/debug.png
:width: 80%
:figclass: auto
```

Clicking this icon will enable the Jupyter debugger. 

<!-- IDEA: This could be turned into a margin note once supported by quantecon-book-theme -->
```{note}
You may also need to open the Debugger Panel (View -> Debugger Panel).
```

You can set breakpoints by clicking on the line number of the cell you want to debug. 

When you run the cell, the debugger will stop at the breakpoint.  

You can then step through the code line by line using the buttons on the "Next" button on the CALLSTACK toolbar (located in the right hand window).

<!-- IDEA: add a red square around the area of interest in the image -->
```{figure} /_static/lecture_specific/getting_started/debugger_breakpoint.png
:figclass: auto
```

You can explore more functionality of the debugger in the [Jupyter documentation](https://jupyterlab.readthedocs.io/en/latest/user/debugger.html).

### Sharing Notebooks


Notebook files are just text files structured in [JSON](https://en.wikipedia.org/wiki/JSON) and typically ending with `.ipynb`.

You can share them in the usual way that you share files --- or by using web services such as [nbviewer](http://nbviewer.jupyter.org/).

The notebooks you see on that site are **static** html representations.

To run one, download it as an `ipynb` file by clicking on the download icon at the top right.

Save it somewhere, navigate to it from the Jupyter dashboard and then run as discussed above.

```{note}
If you are interested in sharing notebooks containing interactive content, you might want to check out [Binder](https://mybinder.org/).

To collaborate with other people on notebooks, you might want to take a look at

- [Google Colab](https://colab.research.google.com/)
- [Kaggle](https://www.kaggle.com/kernels)

To keep the code private and to use the familiar JupyterLab and Notebook interface, look into the [JupyterLab Real-Time Collaboration extension](https://jupyterlab-realtime-collaboration.readthedocs.io/en/latest/).
```

### QuantEcon Notes

QuantEcon has its own site for sharing Jupyter notebooks related
to economics -- [QuantEcon Notes](http://notes.quantecon.org/).

Notebooks submitted to QuantEcon Notes can be shared with a link, and are open
to comments and votes by the community.

## Installing Libraries

Most of the libraries we need come in Anaconda.

Other libraries can be installed with `pip` or `conda`.

One library we'll be using is [QuantEcon.py](http://quantecon.org/quantecon-py).

(gs_install_qe)=
You can install [QuantEcon.py](http://quantecon.org/quantecon-py) by
starting Jupyter and typing

```{code-block} ipython3
:class: no-execute

!conda install quantecon
```

into a cell.

Alternatively, you can type the following into a terminal

```{code-block} bash
:class: no-execute

conda install quantecon
```

More instructions can be found on the [library page](http://quantecon.org/quantecon-py).

To upgrade to the latest version, which you should do regularly, use

```{code-block} bash
:class: no-execute

conda upgrade quantecon
```

Another library we will be using is [interpolation.py](https://github.com/EconForge/interpolation.py).

This can be installed by typing in Jupyter

```{code-block} ipython3
:class: no-execute

!conda install -c conda-forge interpolation
```

## Working with Python Files

So far we've focused on executing Python code entered into a Jupyter notebook
cell.

Traditionally most Python code has been run in a different way.

Code is first saved in a text file on a local machine

By convention, these text files have a `.py` extension.

We can create an example of such a file as follows:

```{code-cell} ipython
%%writefile foo.py

print("foobar")
```

This writes the line `print("foobar")` into a file called `foo.py` in the local directory.

Here `%%writefile` is an example of a [cell magic](http://ipython.readthedocs.org/en/stable/interactive/magics.html#cell-magics).

### Editing and Execution

If you come across code saved in a `*.py` file, you'll need to consider the
following questions:

1. how should you execute it?
1. How should you modify or edit it?

#### Option 1: JupyterLab


[JupyterLab](https://github.com/jupyterlab/jupyterlab) is an integrated development environment built on top of Jupyter notebooks.

With JupyterLab you can edit and run `*.py` files as well as Jupyter notebooks.

To start JupyterLab, search for it in the applications menu or type `jupyter-lab` in a terminal.

Now you should be able to open, edit and run the file `foo.py` created above by opening it in JupyterLab.

Read the docs or search for a recent YouTube video to find more information.

#### Option 2: Using a Text Editor

One can also edit files using a text editor and then run them from within
Jupyter notebooks.

A text editor is an application that is specifically designed to work with text files --- such as Python programs.

Nothing beats the power and efficiency of a good text editor for working with program text.

A good text editor will provide

* efficient text editing commands (e.g., copy, paste, search and replace)
* syntax highlighting, etc.

Right now, an extremely popular text editor for coding is [VS Code](https://code.visualstudio.com/).

VS Code is easy to use out of the box and has many high quality extensions.

Alternatively, if you want an outstanding free text editor and don't mind a seemingly vertical learning curve plus long days of pain and suffering while all your neural pathways are rewired, try [Vim](http://www.vim.org/).

## Exercises

```{exercise-start}
:label: gs_ex1
```

If Jupyter is still running, quit by using `Ctrl-C` at the terminal where
you started it.

Now launch again, but this time using `jupyter notebook --no-browser`.

This should start the kernel without launching the browser.

Note also the startup message: It should give you a URL such as `http://localhost:8888` where the notebook is running.

Now

1. Start your browser --- or open a new tab if it's already running.
1. Enter the URL from above (e.g. `http://localhost:8888`) in the address bar at the top.

You should now be able to run a standard Jupyter notebook session.

This is an alternative way to start the notebook that can also be handy.

This can also work when you accidentally close the webpage as long as the kernel is still running.

```{exercise-end}
```
