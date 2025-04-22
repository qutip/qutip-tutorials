---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: qutip-tutorials
  language: python
  name: python3
---

# HEOM 1b: Spin-Bath model (very strong coupling)

+++

## Introduction

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single Bosonic environment.  The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The Bosonic environment is implicitly assumed to obey a particular Hamiltonian, the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how to model the overdamped Drude-Lorentz Spectral Density, commonly used with the HEOM. We show how to do this using the Matsubara, Pade and fitting decompositions, and compare their convergence.

This notebook shows a similar example to notebook 1a, but with much stronger coupling as discussed in [Shi *et al.*, J. Chem. Phys **130**, 084105 (2009)](https://doi.org/10.1063/1.3077918). Please refer to notebook HEOM 1a for a more detailed explanation.

As in notebook 1a, we present a variety of simulations using different techniques to showcase the effect of different approximations of the correlation function on the results:

- Simulation 1: Matsubara decomposition, not using Ishizaki-Tanimura terminator
- Simulation 2: Matsubara decomposition (including terminator)
- Simulation 3: Pade decomposition
- Simulation 4: Fitting approach

Lastly we compare the results to using the Bloch-Redfield approach:

- Simulation 5: Bloch-Redfield

which does not give the correct evolution in this case.


### Drude-Lorentz (overdamped) spectral density

The Drude-Lorentz spectral density is:

$$J_D(\omega)= \frac{2\omega\lambda\gamma}{{\gamma}^2 + \omega^2}$$

where $\lambda$ scales the coupling strength, and $\gamma$ is the cut-off frequency.  We use the convention,
\begin{equation*}
C(t) = \int_0^{\infty} d\omega \frac{J_D(\omega)}{\pi}[\coth(\beta\omega) \cos(\omega \tau) - i \sin(\omega \tau)]
\end{equation*}

With the HEOM we must use an exponential decomposition:

\begin{equation*}
C(t)=\sum_{k=0}^{k=\infty} c_k e^{-\nu_k t}
\end{equation*}

As an example, the Matsubara decomposition of the Drude-Lorentz spectral density is given by:

\begin{equation*}
    \nu_k = \begin{cases}
               \gamma               & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    c_k = \begin{cases}
               \lambda \gamma (\cot(\beta \gamma / 2) - i)             & k = 0\\
               4 \lambda \gamma \nu_k / \{(nu_k^2 - \gamma^2)\beta \}    & k \geq 1\\
           \end{cases}
\end{equation*}

Note that in the above, and the following, we set $\hbar = k_\mathrm{B} = 1$.

+++

## Setup

```{code-cell} ipython3
import contextlib
import time

import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip import basis, brmesolve, expect, liouvillian, sigmax, sigmaz
from qutip.core.environment import DrudeLorentzEnvironment, system_terminator
from qutip.solver.heom import HEOMSolver

%matplotlib inline
```

## Helper functions

Let's define some helper functions for calculating correlation function expansions, plotting results and timing how long operations take:

```{code-cell} ipython3
def cot(x):
    """Vectorized cotangent of x."""
    return 1.0 / np.tan(x)
```

```{code-cell} ipython3
@contextlib.contextmanager
def timer(label):
    """Simple utility for timing functions:

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
    "rtol": 1e-14,
    "atol": 1e-14,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## System and bath definition

And let us set up the system Hamiltonian, bath and system measurement operators:

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0.0  # Energy of the 2-level system.
Del = 0.2  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# Initial state of the system.
rho0 = basis(2, 0) * basis(2, 0).dag()
```

```{code-cell} ipython3
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz()  # coupling operator

# Bath properties (see Shi et al., J. Chem. Phys. 130, 084105 (2009)):
gamma = 1.0  # cut off frequency
lam = 2.5  # coupling strength
T = 1.0  # in units where Boltzmann factor is 1
beta = 1.0 / T

# HEOM parameters:

# number of exponents to retain in the Matsubara expansion of the
# bath correlation function:
Nk = 1

# Number of levels of the hierarchy to retain:
NC = 13

# Times to solve for:
tlist = np.linspace(0, np.pi / Del, 600)
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

### Plot the spectral density

Let us briefly inspect the spectral density.

```{code-cell} ipython3
bath = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T, Nk=500)
w = np.linspace(0, 5, 1000)
J = bath.spectral_density(w)

# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
axes.plot(w, J, "r", linewidth=2)
axes.set_xlabel(r"$\omega$", fontsize=28)
axes.set_ylabel(r"J", fontsize=28);
```

## Simulation 1: Matsubara decomposition, not using Ishizaki-Tanimura terminator

```{code-cell} ipython3
with timer("RHS construction time"):
    matsBath = bath.approximate(method="matsubara", Nk=Nk)
    HEOMMats = HEOMSolver(Hsys, (matsBath, Q), NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

## Simulation 2: Matsubara decomposition (including terminator)

```{code-cell} ipython3
with timer("RHS construction time"):
    matsBath, delta = bath.approximate(
        method="matsubara", Nk=Nk, compute_delta=True
    )
    terminator = system_terminator(Q, delta)
    Ltot = liouvillian(Hsys) + terminator
    HEOMMatsT = HEOMSolver(Ltot, (matsBath, Q), NC, options=options)

with timer("ODE solver time"):
    resultMatsT = HEOMMatsT.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

P11_mats = np.real(expect(resultMats.states, P11p))
axes.plot(
    tlist,
    np.real(P11_mats),
    "b",
    linewidth=2,
    label="P11 (Matsubara)",
)

P11_matsT = np.real(expect(resultMatsT.states, P11p))
axes.plot(
    tlist,
    np.real(P11_matsT),
    "b--",
    linewidth=2,
    label="P11 (Matsubara + Terminator)",
)

axes.set_xlabel(r"t", fontsize=28)
axes.legend(loc=0, fontsize=12);
```

## Simulation 3: Pade decomposition

```{code-cell} ipython3
# First, compare Matsubara and Pade decompositions
padeBath = bath.approximate("pade", Nk=Nk)


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(16, 8))

ax1.plot(
    tlist,
    np.real(bath.correlation_function(tlist)),
    "r",
    linewidth=2,
    label="Exact",
)
ax1.plot(
    tlist,
    np.real(matsBath.correlation_function(tlist)),
    "g--",
    linewidth=2,
    label=f"Mats (Nk={Nk})",
)
ax1.plot(
    tlist,
    np.real(padeBath.correlation_function(tlist)),
    "b--",
    linewidth=2,
    label=f"Pade (Nk={Nk})",
)

ax1.set_xlabel(r"t", fontsize=28)
ax1.set_ylabel(r"$C_R(t)$", fontsize=28)
ax1.legend(loc=0, fontsize=12)

tlist2 = tlist[0:50]
ax2.plot(
    tlist2,
    np.abs(
        matsBath.correlation_function(tlist2)
        - bath.correlation_function(tlist2)
    ),
    "g",
    linewidth=2,
    label="Mats Error",
)
ax2.plot(
    tlist2,
    np.abs(
        padeBath.correlation_function(tlist2)
        - bath.correlation_function(tlist2)
    ),
    "b--",
    linewidth=2,
    label="Pade Error",
)

ax2.set_xlabel(r"t", fontsize=28)
ax2.legend(loc=0, fontsize=12);
```

```{code-cell} ipython3
with timer("RHS construction time"):
    HEOMPade = HEOMSolver(Hsys, (padeBath, Q), NC, options=options)

with timer("ODE solver time"):
    resultPade = HEOMPade.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
fig, axes = plt.subplots(figsize=(8, 8))

axes.plot(
    tlist,
    np.real(P11_mats),
    "b",
    linewidth=2,
    label="P11 (Matsubara)",
)
axes.plot(
    tlist,
    np.real(P11_matsT),
    "b--",
    linewidth=2,
    label="P11 (Matsubara + Terminator)",
)

P11_pade = np.real(expect(resultPade.states, P11p))
axes.plot(
    tlist,
    np.real(P11_pade),
    "r",
    linewidth=2,
    label="P11 (Pade)",
)

axes.set_xlabel(r"t", fontsize=28)
axes.legend(loc=0, fontsize=12);
```

## Simulation 4: Fitting approach

In `HEOM 1a: Spin-Bath model (introduction)` a fit is performed manually, here
we will use the built-in tools. More details about them can be seen in 
`HEOM 1d: Spin-Bath model, fitting of spectrum and correlation functions`

```{code-cell} ipython3
tfit = np.linspace(0, 10, 10000)
lower = [0, -np.inf, -1e-6, -3]
guess = [np.real(bath.correlation_function(0)) / 10, -10, 0, 0]
upper = [5, 0, 1e-6, 0]
# for better fits increase the first element in upper, or change approximate
# method that makes the simulation much slower (Larger C(t) as C(0) is fit
# better)
envfit, fitinfo = bath.approximate(
    "cf",
    tlist=tfit,
    Nr_max=2,
    Ni_max=1,
    full_ansatz=True,
    sigma=0.1,
    maxfev=1e6,
    target_rmse=None,
    lower=lower,
    upper=upper,
    guess=guess,
)
```

```{code-cell} ipython3
print(fitinfo["summary"])
```

We can quickly compare the result of the Fit with the Pade expansion

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

ax1.plot(
    tlist,
    np.real(bath.correlation_function(tlist)),
    "r",
    linewidth=2,
    label="Exact",
)
ax1.plot(
    tlist,
    np.real(envfit.correlation_function(tlist)),
    "g--",
    linewidth=2,
    label="Fit",
    marker="o",
    markevery=50,
)
ax1.plot(
    tlist,
    np.real(padeBath.correlation_function(tlist)),
    "b--",
    linewidth=2,
    label=f"Pade (Nk={Nk})",
)

ax1.set_xlabel(r"t", fontsize=28)
ax1.set_ylabel(r"$C_R(t)$", fontsize=28)
ax1.legend(loc=0, fontsize=12)

ax2.plot(
    tlist,
    np.imag(bath.correlation_function(tlist)),
    "r",
    linewidth=2,
    label="Exact",
)
ax2.plot(
    tlist,
    np.imag(envfit.correlation_function(tlist)),
    "g--",
    linewidth=2,
    label="Fit",
    marker="o",
    markevery=50,
)
ax2.plot(
    tlist,
    np.imag(padeBath.correlation_function(tlist)),
    "b--",
    linewidth=2,
    label=f"Pade (Nk={Nk})",
)

ax2.set_xlabel(r"t", fontsize=28)
ax2.set_ylabel(r"$C_I(t)$", fontsize=28)
ax2.legend(loc=0, fontsize=12)
```

```{code-cell} ipython3
with timer("RHS construction time"):
    # We reduce NC slightly here for speed of execution because we retain
    # 3 exponents in ckAR instead of 1. Please restore full NC for
    # convergence though:
    HEOMFit = HEOMSolver(Hsys, (envfit, Q), int(NC * 0.7), options=options)

with timer("ODE solver time"):
    resultFit = HEOMFit.run(rho0, tlist)
```

## Simulation 5: Bloch-Redfield

```{code-cell} ipython3
with timer("ODE solver time"):
    resultBR = brmesolve(
        Hsys,
        rho0,
        tlist,
        a_ops=[[sigmaz(), bath]],
        sec_cutoff=0,
        options=options,
    )
```

## Let's plot all our results

Finally, let's plot all of our different results to see how they shape up against each other.

```{code-cell} ipython3
# Calculate expectation values in the bases:
P11_mats = np.real(expect(resultMats.states, P11p))
P11_matsT = np.real(expect(resultMatsT.states, P11p))
P11_pade = np.real(expect(resultPade.states, P11p))
P11_fit = np.real(expect(resultFit.states, P11p))
P11_br = np.real(expect(resultBR.states, P11p))
```

```{code-cell} ipython3
rcParams = {
    "axes.titlesize": 25,
    "axes.labelsize": 30,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
    "axes.grid": False,
    "savefig.bbox": "tight",
    "lines.markersize": 5,
    "font.family": "STIXgeneral",
    "mathtext.fontset": "stix",
    "font.serif": "STIX",
    "text.usetex": False,
}
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

with plt.rc_context(rcParams):
    # Plot the results
    plt.yticks([0.99, 1.0], [0.99, 1])
    axes.plot(
        tlist,
        np.real(P11_mats),
        "b",
        linewidth=2,
        label=f"Matsubara $N_k={Nk}$",
    )
    axes.plot(
        tlist,
        np.real(P11_matsT),
        "g--",
        linewidth=3,
        label=f"Matsubara $N_k={Nk}$ & terminator",
    )
    axes.plot(
        tlist,
        np.real(P11_pade),
        "y-.",
        linewidth=2,
        label=f"Padé $N_k={Nk}$",
    )
    axes.plot(
        tlist,
        np.real(P11_fit),
        "r",
        dashes=[3, 2],
        linewidth=2,
        label=r"Fit $N_f = 3$, $N_k=15 \times 10^3$",
    )
    axes.plot(
        tlist,
        np.real(P11_br),
        "b-.",
        linewidth=1,
        label="Bloch Redfield",
    )

    axes.locator_params(axis="y", nbins=6)
    axes.locator_params(axis="x", nbins=6)
    axes.set_ylabel(r"$\rho_{11}$", fontsize=30)
    axes.set_xlabel(r"$t\;\gamma$", fontsize=30)
    axes.set_xlim(tlist[0], tlist[-1])
    axes.set_ylim(0.98405, 1.0005)
    axes.legend(loc=0)
```

## About

```{code-cell} ipython3
qutip.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
assert np.allclose(P11_matsT, P11_pade, rtol=1e-3)
assert np.allclose(P11_matsT, P11_fit, rtol=1e-3)
```
