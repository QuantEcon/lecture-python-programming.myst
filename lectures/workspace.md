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

(workspace)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Working with Python


```{contents} Contents
:depth: 2
```
## Overview

So far, we have explored the use of Jupyter notebooks in writing and executing Python code. 

While they are efficient and adaptable when working with short pieces of code, notebooks are not the best choice for longer programs and scripts. 

We will explore the use of Python scripts as an alternative. 

The Jupyter Lab and Visual Studio Code (VS Code) development environments are then introduced along with a primer on version control (Git).

In this lecture, you will learn to
- work with Python scripts
- set up multiple development environments
- get started with GitHub

```{note}
Going forward, it is assumed that you have an Anaconda environment up and running.
```

You may want to [create a new conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) if you haven't done so already.

## Working with Python Files 

Python files are used when writing long, reusable blocks of code - by convention, they have a ```.py``` suffix. 

Let us begin by working with the following example.


```{code-block} python
:caption: sine_wave.py
:lineno-start: 1

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.show()
```

The code is first saved locally on the computer before it is executed. Generally, the command 

```
> python <path to file>
``` 
is used, though arguments can be passed to it when necessary. 

The command is executed on the console of your choice - here, your Anaconda Prompt might be appropriate.

Here's an execution of same code in a VS Code workspace (explored in detail later in this lecture).

```{figure} /_static/lecture_specific/workspace/sine_wave.png
:figclass: auto
```

One major advantage of using Python scripts lies in the fact that you can "import" functionality from other scripts into your current script or Jupyter Notebook. 

Let's rewrite the earlier code into a function.

```{code-block} python
:caption: sine_wave.py
:lineno-start: 1
import matplotlib.pyplot as plt
import numpy as np

# Define the plot_wave function.
def plot_wave(title : str = 'Sine Wave'):
  x = np.linspace(0, 10, 100)
  y = np.sin(x)

  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title(title)
  plt.show()
```

```{code-block} python
:caption: second_script.py
:lineno-start: 1

import sine_wave # Import the sine_wave script
 
# Call the plot_wave function.
sine_wave.plot_wave("Sine Wave - Called from the Second Script")
```

Here's an execution of the above code.

```{figure} /_static/lecture_specific/workspace/sine_wave_import.png
:figclass: auto
```

This allows you to split your code into chuncks and structure your codebase better.

Look into the use of [modules](https://docs.python.org/3/tutorial/modules.html) and [packages](https://docs.python.org/3/tutorial/modules.html#packages) for more information on importing functionality.

## Development Environments

A development environment is a one stop workspace where you can 
- edit and run your code
- test and debug 
- manage project files

This lecture takes you through the workings of two development environments. 

## A Step Forward from Jupyter Notebooks: JupyterLab

JupyterLab is a browser based development environment for Jupyter notebooks, code scripts, and data files.

You can install JupyterLab using pip

```
> pip install jupyterlab
``` 

and launch it in the browser, similar to Jupyter notebooks.

```
> jupyter-lab
```

```{figure} /_static/lecture_specific/workspace/jupyter_lab_cmd.png
:figclass: auto
```

You can see that the Jupyter Server is running on port 8888 on the localhost. 

The following interface should open up on your default browser automatically - if not, CTRL + Click the server URL.

```{figure} /_static/lecture_specific/workspace/jupyter_lab.png
:figclass: auto
```

Click on 

- the Python 3 (ipykernel) button under Notebooks to open a new Jupyter notebook
- the Python File button to open a new Python script (.py)

You can always open this launcher tab by clicking the '+' button on the top.

To run a ```.py``` file, open a Terminal from the launcher and proceed as earlier.

All the files and folders in your working directory can be found in the File Browser (tab on the left).

You can create new files and folders using the buttons available at the top of the File Browser tab. 

```{figure} /_static/lecture_specific/workspace/file_browser.png
:figclass: auto
```
You can install extensions that increase the functionality of JupyterLab by visiting the Extensions tab.

```{figure} /_static/lecture_specific/workspace/extensions.png
:figclass: auto
```
## A Walk through Visual Studio Code

Visual Studio Code (VS Code) is a code editor and development workspace that can run
- in the [browser](https://vscode.dev/).
- as a local [installation](https://code.visualstudio.com/docs/?dv=win). 

Both interfaces are identical. 

When you launch VS Code, you will see the following interface.

```{figure} /_static/lecture_specific/workspace/vs_code_home.png
:figclass: auto
```

Explore how to customize VS Code to your liking through the guided walkthroughs.

```{figure} /_static/lecture_specific/workspace/vs_code_walkthrough.png
:figclass: auto
```
When presented with the following prompt, go ahead an install all recommended extensions.

```{figure} /_static/lecture_specific/workspace/vs_code_install_ext.png
:figclass: auto
```
You can also install extensions from the Extensions tab.

```{figure} /_static/lecture_specific/workspace/vs_code_extensions.png
:figclass: auto
```
To run a ```.py``` file, 

- open a Terminal from the launcher and proceed as earlier.
- click on the Run button on the top right corner of the editor.

```{figure} /_static/lecture_specific/workspace/vs_code_run.png
:figclass: auto
```
Jupyter notebooks (```.ipynb``` files) can be worked on in VS Code.

Make sure to install the Jupyter extension from the Extensions tab before you try to open a Jupyter notebook.

Create a new file (in the file Explorer tab) and save it with the ```.ipynb``` extension.

Choose a kernel/environment to run the notebook in by clicking on the Select Kernel button on the top right corner of the editor.

```{figure} /_static/lecture_specific/workspace/vs_code_kernels.png
:figclass: auto
```

VS Code also has excellent version control functionality through the Source Control tab.

```{figure} /_static/lecture_specific/workspace/vs_code_git.png
:figclass: auto
```
Link your GitHub account to VS Code to push and pull changes to and from your repositories.

Further discussions about version control can be found in the next section.

To open a new Terminal in VS Code, click on the Terminal tab and select New Terminal.

VS Code opens a new Terminal in the same directory you are working in - a PowerShell in Windows and a Bash in Linux.

You can change the shell or open a new instance through the dropdown menu on the right end of the terminal tab.

```{figure} /_static/lecture_specific/workspace/vs_code_terminal_opts.png
:figclass: auto
```

VS Code helps you manage conda environments without using the command line.

Open the Command Palette (CTRL + SHIFT + P or from the dropdown menu under View tab) and search for ```Python: Select Interpreter```.

This loads existing environments. 

You can also create new environments using ```Python: Create Environment``` in the Command Palette.

A new environment (.conda folder) is created in the the current working directory.

## Git Your Hands Dirty

This section will familiarize you with git and GitHub.

[Git](http://git-scm.com/) is a *version control system* --- a piece of software used to manage digital projects such as code libraries.

In many cases, the associated collections of files --- called *repositories* --- are stored on [GitHub](https://github.com/).

GitHub is a wonderland of collaborative coding projects.

For example, it hosts many of the scientific libraries we'll be using later
on, such as [this one](https://github.com/pydata/pandas).

Git is the underlying software used to manage these projects.

Git is an extremely powerful tool for distributed collaboration --- for
example, we use it to share and synchronize all the source files for these
lectures.

There are two main flavors of Git

1. the plain vanilla [command line Git](http://git-scm.com/downloads) version
2. the various point-and-click GUI versions
    * See, for example, the [GitHub version](https://desktop.github.com/) or Git GUI integrated into your IDE.

In case you already haven't, try

1. Installing Git.
1. Getting a copy of [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py) using Git.

For example, if you've installed the command line version, open up a terminal and enter.

```{code-block} bash
:class: no-execute

git clone https://github.com/QuantEcon/QuantEcon.py
```
(This is just `git clone` in front of the URL for the repository)

This command will download all necessary components to rebuild the lecture you are reading now.

As the 2nd task,

1. Sign up to [GitHub](https://github.com/).
1. Look into 'forking' GitHub repositories (forking means making your own copy of a GitHub repository, stored on GitHub).
1. Fork [QuantEcon.py](https://github.com/QuantEcon/QuantEcon.py).
1. Clone your fork to some local directory, make edits, commit them, and push them back up to your forked GitHub repo.
1. If you made a valuable improvement, send us a [pull request](https://help.github.com/articles/about-pull-requests/)!

For reading on these and other topics, try

* [The official Git documentation](http://git-scm.com/doc).
* Reading through the docs on [GitHub](https://docs.github.com/en).
* [Pro Git Book](http://git-scm.com/book) by Scott Chacon and Ben Straub.
* One of the thousands of Git tutorials on the Net.
