# QuTiP Tutorials

This repositories collects tutorials of different complexity for
using [QuTiP](https://qutip.org/). Some of the notebooks are also shown on
the [QuTiP Tutorials website](https://qutip.org/tutorials).

The notebooks in this repository are stored in a Markdown format and thus
have no outputs. To generate the outputs, follow the installation guide below.

The notebooks are located in the folders `tutorials-v4` / `tutorials-v5`, 
where the version number stands for the QuTiP version they work with.

## Installation Guide

To modify and execute the notebooks yourself, you have to install an
environment with the required packages.

If you use Anaconda, you can install the required dependencies for this
repository by:

```shell
cd qutip-tutorials
conda env create --file environment.yml
conda activate qutip-tutorials
```

Alternatively, you can install the requirements using `pip` (we recommend
the usage of virtual environments):

```shell
pip install -r requirements.txt
```

Regardless of the installation method, you can now start *Jupyter Notebook* by
executing:

```shell
jupyter notebook
```

Your browser should automatically open the Jupyter Notebook frontend. Otherwise
open the link displayed in the terminal.

Navigate into the `tutorials-v4` or `tutorials-v5` directory and select one of 
the notebooks.
Note that the format of the notebooks is `.md` (markdown), which is intended
for better compatibility with git features.

## Contributing

You are most welcome to contribute to QuTiP development by forking this
repository and sending pull requests, or filing bug reports at
the [issues page](https://github.com/qutip/qutip-tutorials/issues).
Note that all notebooks are tested automatically to work with the latest
version of QuTiP. Furthermore, this repository uses notebooks in the markdown
format. See below how to convert the format of an already existing notebook.

### Add a new notebook

If you want to create a new notebook, copy the `template.md` located in the
`tutorials` directory, edit it and save it as a new markdown file. Please
keep in mind that new users might use the notebook as an entry point to
QuTiP.

### Add an existing notebook

To add an already existing notebook to the repository, copy it to the
`tutorials` directory and create a pull request. If the notebook is in the `.
ipynb` format please convert it to markdown using JupyText by executing:

```shell
jupytext --to md my_notebook.ipynb
```

### Formatting a notebook

We aim to create notebooks consistent with the PEP8 style guide. Therefore, we 
use `flake8` to check the formatting of every notebook. To format a notebook 
before adding it to this repository you can use 
[`black`](https://github.com/psf/black) and 
[`isort`](https://pycqa.github.io/isort/) to do so.
You can apply these two tools to notebook by using the tool 
[`nbQA`](https://github.com/nbQA-dev/nbQA).

To format any notebook `notebook.ipynb` (in the Jupyter format) run:

```shell
nbqa black notebook.ipynb
nbqa isort notebook.ipynb
```

To test whether the notebook conforms with the PEP8 style guide run:

```shell
nbqa flake8 notebook.ipynb
```

If the notebook is already in the MarkDown format, you can use `JupyText` to convert it back to `.ipynb`:

```shell
jupytext --to notebook notebook.md
```

If the notebook is in the MarkDown format saved via Jupytext, you can format it using:

```shell
nbqa black notebook.md
```

## LaTeX and ImageMagick installation

Some functions of the notebooks (e.g. plotting QCircuits) require a working
installation of [*ImageMagick*](https://imagemagick.org/) and *LaTeX*. If
you used `conda` to install the requirements, *ImageMagick* is already
installed. Otherwise, follow the instruction on their website.

