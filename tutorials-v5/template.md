---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Notebook Title

Author: C. Staufenbiel, 2022

### Introduction

This notebook serves is a template for new Jupyter Notebooks used as a user
guides for QuTiP. With this template we want to give you an idea of how a
user guide can look like. Furthermore, we want to ensure that all notebooks
have a similar style and that new users can easily understand them. To
create your own notebook, just copy this template and insert your own
content. The descriptions in this template should give you an idea of the
general style.

In this introductory section, you should explain the goal of this notebook,
just as I did. We continue now with some steps. It is a good practice (and most
of the times appreciated by new users) to comment every code cell with one
markdown cell. Also please update the notebook title and the headings of the
different section.

### First Section

The first thing we do in this notebook (and possibly in any notebook) is that we
import needed packages.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip import Bloch, basis, sesolve, sigmay, sigmaz

%matplotlib inline
```

In the next step we setup some qubit state and plot it on the bloch sphere. It's
always great to give some nice visuals.

```python
psi = (2.0 * basis(2, 0) + basis(2, 1)).unit()
b = Bloch()
b.add_states(psi)
b.show()
```

### Simulation

Let's define a simple Hamiltonian and use `qutip.sesolve` to solve the
Schrödinger equation and look at the expectation value of $\sigma_y$. You can
also use comments in the code section to separate the operations you perform.

```python
# simulate the unitary dynamics
H = sigmaz()
times = np.linspace(0, 10, 100)
result = sesolve(H, psi, times, [sigmay()])

# plot the expectation value
plt.plot(times, result.expect[0])
plt.xlabel("Time"), plt.ylabel("<sigma_y>")
plt.show()
```

We created a nice looking plot of the Larmor precision. Every notebook has to
include the `qutip.about()` call at the end, to show the setup under which the
notebook was executed and make it reproducible for others.

### About

```python
qutip.about()
```

### Testing

This section can include some tests to verify that the expected outputs are
generated within the notebook. We put this section at the end of the notebook,
so it's not interfering with the user experience. Please, define the tests
using `assert`, so that the cell execution fails if a wrong output is generated.

```python
assert np.allclose(result.expect[0][0], 0), "Expectation value does not start at 1"
assert 1 == 1
```

```python

```
