# lecture-python-programming.myst

Migration to Myst (Source files for https://python-programming.quantecon.org)

Preview builds are available on Pull Requests

**WARNING:** These are NOT the source files for the live site. Please do not edit for that purpose.

These are files automatically converted from [lecture-python-programming](https://github.com/QuantEcon/lecture-python-programming) using [sphinxcontrib-tomyst](https://github.com/mmcky/sphinxcontrib-tomyst)for the QuantEcon migration project.

This repo has been setup to identify issues with myst conversion process which will need:

1. fixing upstream in [sphinxcontrib-tomyst](https://github.com/mmcky/sphinxcontrib-tomyst)
2. Manual intervention in the transfer process

Please raise problems as Issues on this repository.

## Building Files

Currently files are built using `myst_parser` via `sphinx`:

```bash
make html
```

or for `LaTeX`:

```bash
make latexpdf
```

but with all the tools that `jupyter-book` uses.