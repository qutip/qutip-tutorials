---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# HEOM 2: Dynamics in Fenna-Mathews-Olsen complex (FMO)

+++

## Introduction

In this example notebook we outline how to employ the HEOM to
solve the FMO photosynthetic complex dynamics.

We aim to replicate the results in reference [https://www.pnas.org/content/106/41/17255](https://pubmed.ncbi.nlm.nih.gov/19815512/)
and compare them to a Bloch-Redfield (perturbative) solution.

This demonstrates how to to employ the solver for multiple baths, as well as showing how a
quantum environment reduces the effect of pure dephasing.

+++

## Setup

```{code-cell} ipython3
import contextlib
import time

import numpy as np
from matplotlib import pyplot as plt

from qutip import Qobj, about, basis, brmesolve, expect, liouvillian, mesolve
from qutip.core.environment import DrudeLorentzEnvironment, system_terminator
from qutip.solver.heom import HEOMSolver


```

## Helper functions

```{code-cell} ipython3
@contextlib.contextmanager
def timer(label):
    """ Simple utility for timing functions:

        with timer("name"):
            ... code to time ...
    """
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start}")
```

```{code-cell} ipython3
# Solver options:

options = {
    "nsteps": 15000,
    "store_states": True,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## System and bath definition

And let us set up the system Hamiltonian and bath parameters:

```{code-cell} ipython3
# System Hamiltonian:
#
# We use the Hamiltonian employed in
# https://www.pnas.org/content/106/41/17255 and operate
# in units of GHz:

Hsys = 30 * 2 * np.pi * Qobj([
    [200, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
    [-87.7, 320, 30.8, 8.2, 0.7, 11.8, 4.3],
    [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6.0],
    [-5.9, 8.2, -53.5, 110, -70.7, -17.0, -63.3],
    [6.7, 0.7, -2.2, -70.7, 270, 81.1, -1.3],
    [-13.7, 11.8, -9.6, -17.0, 81.1, 420, 39.7],
    [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 230],
])
```

```{code-cell} ipython3
# Bath parameters

lam = 35 * 30 * 2 * np.pi
gamma = 1 / 166e-6
T = 300 * 0.6949 * 30 * 2 * np.pi
beta = 1 / T
```

## Plotting the environment spectral density and correlation functions

Let's quickly plot the spectral density and environment correlation functions so that we can see what they look like.

```{code-cell} ipython3
env = DrudeLorentzEnvironment(T=T, lam=lam, gamma=gamma)
```

```{code-cell} ipython3
wlist = np.linspace(0, 200 * 30 * 2 * np.pi, 100)
tlist = np.linspace(0, 1e-3, 1000)

J = env.spectral_density(wlist) / (30 * 2 * np.pi)

fig, axes = plt.subplots(1, 2, sharex=False, figsize=(10, 3))

fig.subplots_adjust(hspace=0.1)  # reduce space between plots

# Spectral density plot:

axes[0].plot(wlist / (30 * 2 * np.pi), J, color='r', ls='--', label="J(w)")
axes[0].set_xlabel(r'$\omega$ (cm$^{-1}$)', fontsize=20)
axes[0].set_ylabel(r"$J(\omega)$ (cm$^{-1}$)", fontsize=16)
axes[0].legend()

# Correlation plot:

axes[1].plot(
    tlist, np.real(env.correlation_function(tlist, 10)),
    color='r', ls='--', label="C(t) real",
)
axes[1].plot(
    tlist, np.imag(env.correlation_function(tlist, 10)),
    color='g', ls='--', label="C(t) imaginary",
)
axes[1].set_xlabel(r'$t$', fontsize=20)
axes[1].set_ylabel(r"$C(t)$", fontsize=16)
axes[1].legend();
```

## Solve for the dynamics with the HEOM

Now let us solve for the evolution of this system using the HEOM.

```{code-cell} ipython3
# We start the excitation at site 1:
rho0 = basis(7, 0) * basis(7, 0).dag()

# HEOM solver options:
#
# Note: We set Nk=0 (i.e. a single correlation expansion term
#       per bath) and rely on the terminator to correct detailed
#       balance.
NC = 4  # Use NC=8 for more precise results
Nk = 0

Q_list = []
Ltot = liouvillian(Hsys)
env_approx, delta = env.approximate(
    method="matsubara", Nk=Nk, compute_delta=True
)
for m in range(7):
    Q = basis(7, m) * basis(7, m).dag()
    Q_list.append(Q)
    Ltot += system_terminator(Q, delta)
```

```{code-cell} ipython3
with timer("RHS construction time"):
    HEOMMats = HEOMSolver(Hsys, [(env_approx, Q) for Q in Q_list],
                          NC, options=options)

with timer("ODE solver time"):
    outputFMO_HEOM = HEOMMats.run(rho0, tlist)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, figsize=(12, 8))

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
linestyles = [
    '-', '--', ':', '-.',
    (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)),
]

for m in range(7):
    Q = basis(7, m) * basis(7, m).dag()
    axes.plot(
        np.array(tlist) * 1e6,
        np.real(expect(outputFMO_HEOM.states, Q)),
        label=m + 1,
        color=colors[m % len(colors)],
        linestyle=linestyles[m % len(linestyles)],
    )
    axes.set_xlabel(r'$t$ (fs)', fontsize=30)
    axes.set_ylabel(r"Population", fontsize=30)
    axes.locator_params(axis='y', nbins=6)
    axes.locator_params(axis='x', nbins=6)

axes.set_title('HEOM solution', fontsize=24)
axes.legend(loc=0)
axes.set_xlim(0, 1000)
plt.yticks([0.0, 0.5, 1], [0, 0.5, 1])
plt.xticks([0.0, 500, 1000], [0, 500, 1000]);
```

## Comparison with Bloch-Redfield solver

Now let us solve the same problem using the Bloch-Redfield solver. We will see that the Bloch-Redfield technique fails to model the oscillation of population of the states that we saw in the HEOM.

In the next section, we will examine the role of pure dephasing in the evolution to understand why this happens.

```{code-cell} ipython3
with timer("BR ODE solver time"):
    outputFMO_BR = brmesolve(
        Hsys, rho0, tlist,
        a_ops=[[Q, env] for Q in Q_list],
        options=options,
    )
```

And now let's plot the Bloch-Redfield solver results:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, figsize=(12, 8))

for m, Q in enumerate(Q_list):
    axes.plot(tlist * 1e6, expect(outputFMO_BR.states, Q), label=m + 1)

axes.set_xlabel(r'$t$ (fs)', fontsize=30)
axes.set_ylabel(r"Population", fontsize=30)

axes.set_title('Bloch-Redfield solution ', fontsize=24)
axes.legend()
axes.set_xlim(0, 1000)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.xticks([0, 500, 1000], [0, 500, 1000]);
```

Notice how the oscillations are gone and the populations decay much more rapidly.

Next let us try to understand why.

+++

## Role of pure dephasing

It is useful to construct the various parts of the Bloch-Redfield master equation explicitly and to solve them using the Master equation solver, `mesolve`. We will do so and show that it is the pure-dephasing terms which suppresses coherence in these oscillations.

First we will write a function to return the list of collapse operators for a given system, either with or without the dephasing operators:

```{code-cell} ipython3
def J0_dephasing():
    """ Under-damped brownian oscillator dephasing probability.

        This returns the limit as w -> 0 of J0(w) * n_th(w, T) / T.
    """
    return 2 * lam * gamma / gamma**2
```

```{code-cell} ipython3
def get_collapse(H, T, dephasing=1):
    """ Calculate collapse operators for a given system H and
        temperature T.
    """
    all_energy, all_state = H.eigenstates(sort="low")
    Nmax = len(all_energy)

    Q_list = [basis(Nmax, n) * basis(Nmax, n).dag() for n in range(Nmax)]

    collapse_list = []

    for Q in Q_list:
        for j in range(Nmax):
            for k in range(j + 1, Nmax):
                Deltajk = abs(all_energy[k] - all_energy[j])
                if abs(Deltajk) > 0:
                    rate = np.abs(
                        Q.matrix_element(all_state[j].dag(), all_state[k])
                    ) ** 2 * env.power_spectrum(Deltajk)
                    if rate > 0.0:
                        # emission:
                        collapse_list.append(
                            np.sqrt(rate) * all_state[j] * all_state[k].dag()
                        )

                    rate = np.abs(
                        Q.matrix_element(all_state[k].dag(), all_state[j])
                    ) ** 2 * env.power_spectrum(-Deltajk)
                    if rate > 0.0:
                        # absorption:
                        collapse_list.append(
                            np.sqrt(rate) * all_state[k] * all_state[j].dag()
                        )

        if dephasing:
            for j in range(Nmax):
                rate = (
                    np.abs(
                        Q.matrix_element(all_state[j].dag(), all_state[j])
                    ) ** 2 * env.power_spectrum(0) / 2
                )
                if rate > 0.0:
                    # emission:
                    collapse_list.append(
                        np.sqrt(rate) * all_state[j] * all_state[j].dag()
                    )

    return collapse_list
```

Now we are able to switch the pure dephasing terms on and off.

Let us starting by including the dephasing operators. We expect to see the same behaviour that we saw when using the Bloch-Redfield solver.

```{code-cell} ipython3
# dephasing terms on, we recover the full BR solution:

with timer("Building the collapse operators"):
    collapse_list = get_collapse(Hsys, T=T, dephasing=True)

with timer("ME ODE solver"):
    outputFMO_ME = mesolve(Hsys, rho0, tlist, collapse_list)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, figsize=(12, 8))

for m, Q in enumerate(Q_list):
    axes.plot(tlist * 1e6, expect(outputFMO_ME.states, Q), label=m + 1)

axes.set_xlabel(r'$t$', fontsize=20)
axes.set_ylabel(r"Population", fontsize=16)
axes.set_xlim(0, 1000)
axes.set_title('With pure dephasing', fontsize=24)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.xticks([0, 500, 1000], [0, 500, 1000])
axes.legend(fontsize=18);
```

We see similar results to before.

Now let us examine what happens when we remove the dephasing collapse operators:

```{code-cell} ipython3
# dephasing terms off

with timer("Building the collapse operators"):
    collapse_list = get_collapse(Hsys, T, dephasing=False)

with timer("ME ODE solver"):
    outputFMO_ME_nodephase = mesolve(Hsys, rho0, tlist, collapse_list)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, figsize=(12, 8))
for m, Q in enumerate(Q_list):
    axes.plot(
        tlist * 1e6,
        expect(outputFMO_ME_nodephase.states, Q),
        label=m + 1,
    )

axes.set_xlabel(r'$t$', fontsize=20)
axes.set_ylabel(r"Population", fontsize=16)
axes.set_xlim(0, 1000)
axes.set_title('Without pure dephasing', fontsize=24)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.xticks([0, 500, 1000], [0, 500, 1000])
axes.legend(fontsize=18);
```

And now we see that without the dephasing, the oscillations reappear. The full dynamics captured by the HEOM are still not capture by this simpler model, however.

+++

## About

```{code-cell} ipython3
about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
assert np.allclose(
    expect(outputFMO_BR.states, Q_list[0]),
    expect(outputFMO_ME.states, Q_list[0]),
    rtol=2e-2,
)
assert np.allclose(
    expect(outputFMO_BR.states, Q_list[1]),
    expect(outputFMO_ME.states, Q_list[1]),
    rtol=2e-2,
)
```
