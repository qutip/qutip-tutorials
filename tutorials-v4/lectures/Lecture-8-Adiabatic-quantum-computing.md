---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Lecture 8 - Adiabatic sweep

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

This lecture series was developed by J.R. Johansson. The original lecture notebooks are available [here](https://github.com/jrjohansson/qutip-lectures).

This is a slightly modified version of the lectures, to work with the current release of QuTiP. You can find these lectures as a part of the [qutip-tutorials repository](https://github.com/qutip/qutip-tutorials). This lecture and other tutorial notebooks are indexed at the [QuTiP Tutorial webpage](https://qutip.org/tutorials.html).


## Introduction

In adiabatic quantum computing, an easy to prepare ground state of a Hamiltonian $H_0$ is prepared, and then the Hamiltonian is gradually transformed into $H_1$, which is constructed in such a way that the ground state of $H_1$ encodes the solution to a difficult problem. The transformation of $H_0$ to $H_1$ can for example be written on the form

$\displaystyle H(t) = \lambda(t) H_0 + (1 - \lambda(t)) H_1$

where $\lambda(t)$ is a function that goes from $0$ to $1$ when $t$ goes from $0$ to $t_{\rm final}$.

If this gradual transformation is slow enough (satisfying the adiabaticity criteria), the evolution of the system will remain in its ground state.

If the Hamiltonian is transformed from $H_0$ to $H_1$ too quickly, the system will get excited from the ground state the adiabatic computing algorithm fails.

In this notebook we explore the dynamics of a spin Hamiltonian that is transformed from a simple Hamiltonian with an easy to prepare ground state, into a random spin Hamiltonian with a complicated ground state.

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, mesolve, qeye, qobj_list_evaluate, sigmax,
                   sigmay, sigmaz, tensor)

%matplotlib inline
```

### Parameters

```python
N = 6  # number of spins
M = 20  # number of eigenenergies to plot

# array of spin energy splittings and coupling strengths (random values).
h = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))
Jz = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))
Jx = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))
Jy = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))

# increase taumax to get make the sweep more adiabatic
taumax = 5.0
taulist = np.linspace(0, taumax, 100)
```

### Precalculate operators

```python
# pre-allocate operators
si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sy
    sy_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))
```

### Construct the initial state

```python
psi_list = [basis(2, 0) for n in range(N)]
psi0 = tensor(psi_list)
H0 = 0
for n in range(N):
    H0 += -0.5 * 2.5 * sz_list[n]
```

### Construct the Hamiltonian

```python
# energy splitting terms
H1 = 0
for n in range(N):
    H1 += -0.5 * h[n] * sz_list[n]

H1 = 0
for n in range(N - 1):
    # interaction terms
    H1 += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
    H1 += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
    H1 += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]

# the time-dependent hamiltonian in list-function format
args = {"t_max": max(taulist)}
h_t = [
    [H0, lambda t, args: (args["t_max"] - t) / args["t_max"]],
    [H1, lambda t, args: t / args["t_max"]],
]
```

### Evolve the system in time

```python
#
# callback function for each time-step
#
evals_mat = np.zeros((len(taulist), M))
P_mat = np.zeros((len(taulist), M))

idx = [0]


def process_rho(tau, psi):

    # evaluate the Hamiltonian with gradually switched on interaction
    H = qobj_list_evaluate(h_t, tau, args)

    # find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0], :] = np.real(evals)

    # find the overlap between the eigenstates and psi
    for n, eket in enumerate(ekets):
        P_mat[idx[0], n] = abs((eket.dag().data * psi.data)[0, 0]) ** 2

    idx[0] += 1
```

```python
# Evolve the system, request the solver to call process_rho at each time step.

mesolve(h_t, psi0, taulist, [], process_rho, args)
```

## Visualize the results

Plot the energy levels and the corresponding occupation probabilities (encoded as the width of each line in the energy-level diagram).

```python
# rc('font', family='serif')
# rc('font', size='10')

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

#
# plot the energy eigenvalues
#

# first draw thin lines outlining the energy spectrum
for n in range(len(evals_mat[0, :])):
    ls, lw = ("b", 1) if n == 0 else ("k", 0.25)
    axes[0].plot(taulist / max(taulist), evals_mat[:, n] / (2 * np.pi), ls,
                 lw=lw)

# second, draw line that encode the occupation probability of each state in
# its linewidth. thicker line => high occupation probability.
for idx in range(len(taulist) - 1):
    for n in range(len(P_mat[0, :])):
        lw = 0.5 + 4 * P_mat[idx, n]
        if lw > 0.55:
            axes[0].plot(
                np.array([taulist[idx], taulist[idx + 1]]) / taumax,
                np.array([evals_mat[idx, n], evals_mat[idx + 1, n]])
                / (2 * np.pi),
                "r",
                linewidth=lw,
            )

axes[0].set_xlabel(r"$\tau$")
axes[0].set_ylabel("Eigenenergies")
axes[0].set_title(
    "Energy spectrum (%d lowest values) of a chain of %d spins.\n " % (M, N)
    + "The occupation probabilities are encoded in the red line widths."
)

#
# plot the occupation probabilities for the few lowest eigenstates
#
for n in range(len(P_mat[0, :])):
    if n == 0:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n], "r", linewidth=2)
    else:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n])

axes[1].set_xlabel(r"$\tau$")
axes[1].set_ylabel("Occupation probability")
axes[1].set_title(
    "Occupation probability of the %d lowest " % M
    + "eigenstates for a chain of %d spins" % N
)
axes[1].legend(("Ground state",));
```

### Software versions:

```python
about()
```
