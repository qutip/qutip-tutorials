# QuTiP Tutorials

This repositories collects tutorials of different complexity for
using [QuTiP](qutip.org). Some of the notebooks are also shown on
the [QuTiP Tutorials website](qutip.org/tutorials).

The notebooks in this repository are stored in a MarkDown format and thus 
have no outputs. To generate the outputs, follow the installation guide below.

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

Navigate into the `tutorials` directory and select one of the notebooks.
Note that the format of the notebooks is `.md` (markdown), which is intended
for better compatibility with git features.

### LaTeX and ImageMagick installation

Some functions of the notebooks (e.g. plotting QCircuits) require a working
installation of [*ImageMagick*](https://imagemagick.org/) and *LaTeX*. If
you used `conda` to install the requirements, *ImageMagick* is already
installed. Otherwise, follow the instruction on their website.

