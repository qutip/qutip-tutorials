---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: qutip-dev
  language: python
  name: python3
---

# HEOM 1d: Spin-Bath model, fitting of spectrum and correlation functions

+++

## Introduction

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded 
in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single bosonic environment.

The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The bosonic environment is implicitly assumed to obey a particular Hamiltonian ([see paper](https://arxiv.org/abs/2010.10806)), the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how to model an Ohmic environment with exponential cut-off in three ways:

* First we fit the spectral density with a set of underdamped brownian oscillator functions.
* Second, we evaluate the correlation functions, and fit those with a certain choice of exponential functions.
* Third, we use the available OhmicBath class 

In each case we will use the fit parameters to determine the correlation function expansion co-efficients needed to construct a description of the bath (i.e. a `BosonicBath` object) to supply to the `HEOMSolver` so that we can solve for the system dynamics.

+++

## Setup

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
import qutip
from qutip import (
    basis,
    expect,
    sigmax,
    sigmaz,
)
from qutip.solver.heom import (
    HEOMSolver
)
from qutip.core.environment import BosonicEnvironment,OhmicEnvironment

# Import mpmath functions for evaluation of gamma and zeta
# functions in the expression for the correlation:

from mpmath import mp

mp.dps = 15
mp.pretty = True

%matplotlib inline
```

## System and bath definition

Let us set up the system Hamiltonian, bath and system measurement operators:

+++

### System Hamiltonian

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0  # Energy of the 2-level system.
Del = 0.2  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
rho0 = basis(2, 0) * basis(2, 0).dag()
```

### System measurement operators

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

### Bath and HEOM parameters

+++

Finally, let's set the bath parameters we will work with and write down some measurement operators:

```{code-cell} ipython3
Q = sigmaz()
alpha = 3.25
T = 0.5
wc = 1.0
s = 1
```

And set the cut-off for the HEOM hierarchy:

```{code-cell} ipython3
# HEOM parameters:

# The max_depth defaults to 5 so that the notebook executes more
# quickly. Change it to 11 to wait longer for more accurate results.
max_depth = 5 #could not do 11 my laptop rans out of ram
# I used 7 because I wanted to make sure things were working correctly
# cf is terribly slow at 7, probably can be done better by changing guess, lower
# upper, use 5 to play around :)

# options used for the differential equation solver, while default works it 
# is way slower than using bdf
options = {
    "nsteps":15000, "store_states":True, "rtol":1e-12, "atol":1e-12, "method":"bdf",
}
```

#### Plotting function

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """Plot the expectation values of operators as functions of time.

    Each plot in plots consists of (solver_result,
    measurement_operation, color, label).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
        fig_created = True
    else:
        fig = None
        fig_created = False

    # add kw arguments to each plot if missing
    plots = [p if len(p) == 5 else p + ({},) for p in plots]
    for result, m_op, color, label, kw in plots:
        exp = np.real(expect(result.states, m_op))
        kw.setdefault("linewidth", 2)
        if color == "rand":
            axes.plot(
                result.times,
                exp,
                c=np.random.rand(
                    3,
                ),
                label=label,
                **kw,
            )
        else:
            axes.plot(result.times, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig
```

# Obtaining a decaying Exponential description of the environment

In order to carry out our HEOM simulation, we need to express the correlation 
function as a sum of decaying exponentials, that is we need to express it as 

$$C(\tau)= \sum_{k=0}^{N-1}c_{k}e^{-\nu_{k}t}$$

As the correlation function of the environment is tied to it's power spectrum via 
a Fourier transform, such a representation of the correlation function implies a 
power spectrum of the form

$$S(\omega)= \sum_{k}2 Re\left( \frac{c_{k}}{\nu_{k}- i \omega}\right)$$

There are several ways one can obtain such a decomposition, in this tutorial we 
will cover the following approaches:

- Non-Linear Least Squares:
    - On the Spectral Density (`sd`)
    - On the Correlation Function (`cf`)
    - On the Power Spectrum (`ps`)
- Methods based on the Prony Polynomial
    - Prony on the correlation function(`prony`)
    - The Matrix Pencil method on the correlation function (`mp`) :question:
    - ESPRIT on the correlation function(`esprit`)
- Methods based on rational Approximations
    - The AAA algorithm on the Power Spectrum (`aaa`)
    - ESPIRA-I (`espira-I`) :question:
    - ESPIRA-II (`espira-II`)

the ones with a question mark are the ones I think maybe can be deleted.
Here's a quick high level comparison between the three different families 
of methods


<table>
  <tr>
    <th>Class</th>
    <th>Requires Extra information</th>
    <th>Fast</th>
    <th>Resilient to Noise</th>
    <th>Allows constraitns</th>
    <th>Stable</th>

    
  </tr>
  <tr>
    <td align="center"> Non-Linear Least Squares</td>
    <td align="center">✔️</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
    <td align="center">✔️</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td align="center">Prony Polynomial</td>
    <td align="center">❌</td>
    <td align="center">✔️</td>
    <td align="center">❗</td>
    <td align="center">❌</td>
    <td align="center">❗</td>
  </tr>
  <tr>
    <td align="center">Rational Approximations </td>
    <td align="center">❌</td>
    <td align="center">✔️</td>
    <td align="center">❗</td>
    <td align="center">❗</td>
    <td align="center">✔️</td>
  </tr>
</table>

Legend:

❌: NO ✔️: Yes ❗: Partially

+++

# Non-Linear Least Squares

```{code-cell} ipython3
obs = OhmicEnvironment(T, alpha, wc,s=1)
tlist = np.linspace(0, 30 * np.pi / Del, 600)
```

## Correlation Function

```{code-cell} ipython3
t=np.linspace(0,20,500)
Obath, fitinfo = obs.approximate(method="cf",tlist=t,Nr_max=4,Ni_max=4,maxfev=1e9,target_rsme=None)
print(fitinfo["summary"])
HEOM_ohmic_corr_fit = HEOMSolver(
    Hsys,
    (Obath,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_corr_fit = HEOM_ohmic_corr_fit.run(rho0, tlist)
```

## Spectral Density

```{code-cell} ipython3
w=np.linspace(0,30,500)
Obath2, fitinfo = obs.approximate(method="sd",wlist=w,Nmax=4,Nk=3)
print(fitinfo["summary"])
HEOM_ohmic_sd_fit = HEOMSolver(
    Hsys,
    (Obath2,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_sd_fit = HEOM_ohmic_sd_fit.run(rho0, tlist)
```

## Power Spectrum

```{code-cell} ipython3
w=np.linspace(-50,30,500)
Obath3, fitinfo = obs.approximate(method="ps",wlist=w,Nmax=5)
print(fitinfo["summary"])
HEOM_ohmic_ps_fit = HEOMSolver(
    Hsys,
    (Obath3,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_ps_fit = HEOM_ohmic_ps_fit.run(rho0, tlist)
```

# Methods based on the Prony Polinomial

+++

## Prony

```{code-cell} ipython3
tlist2=np.linspace(0,40,100)
pbath,fitinfo=obs.approximate("prony",tlist2,Nr=4)
print(fitinfo["summary"])
HEOM_ohmic_prony_fit = HEOMSolver(
    Hsys,
    (pbath,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_prony_fit = HEOM_ohmic_prony_fit.run(rho0, tlist)
```

## Matrix Pencil

```{code-cell} ipython3
mpbath,fitinfo=obs.approximate(method="mp",tlist=tlist2,Nr=5,Ni=5,separate=True)
print(fitinfo["summary"])
HEOM_ohmic_mp_fit = HEOMSolver(
    Hsys,
    (mpbath,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_mp_fit = HEOM_ohmic_mp_fit.run(rho0, tlist)
```

## ESPRIT

```{code-cell} ipython3
esbath,fitinfo=obs.approximate("esprit",tlist2,Nr=4)
print(fitinfo["summary"])
HEOM_ohmic_es_fit = HEOMSolver(
    Hsys,
    (esbath,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_es_fit = HEOM_ohmic_es_fit.run(rho0, tlist)
```

# Rational Approximations

+++

## AAA

```{code-cell} ipython3
aaabath,fitinfo=obs.approximate("aaa",np.concatenate((-np.logspace(3,-8,3500),np.logspace(-8,3,3500))),N_max=6,tol=1e-15)
print(fitinfo["summary"])
HEOM_ohmic_aaa_fit = HEOMSolver(
    Hsys,
    (aaabath,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_aaa_fit = HEOM_ohmic_aaa_fit.run(rho0, tlist)
```

# ESPIRA I

```{code-cell} ipython3
tlist4=np.linspace(0,20,1000)
espibath,fitinfo=obs._approx_by_prony("espira-I",tlist4,Nr=4,Ni=4,separate=True)
print(fitinfo["summary"])
HEOM_ohmic_espira_fit = HEOMSolver(
    Hsys,
    (espibath,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_espira_fit = HEOM_ohmic_espira_fit.run(rho0, tlist)
```

# ESPIRA II

```{code-cell} ipython3
espibath2,fitinfo=obs._approx_by_prony("espira-II",tlist4,Nr=4,Ni=4,separate=True)
print(fitinfo["summary"])
HEOM_ohmic_espira_fit2 = HEOMSolver(
    Hsys,
    (espibath2,Q),
    max_depth=max_depth,
    options=options,
)
results_ohmic_espira2_fit = HEOM_ohmic_espira_fit2.run(rho0, tlist)
```

Finally we plot the dynamics obtained by the different methods

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

plot_result_expectations(
    [

        (results_ohmic_corr_fit, P11p, "r", "Correlation Fit"),
        (results_ohmic_sd_fit, P11p, "g", "SD Fit"),
        (results_ohmic_sd_fit, P11p, "y", "PS Fit"),
        (results_ohmic_prony_fit, P11p, "k", " Prony Fit"),
        (results_ohmic_mp_fit, P11p, "r", "Matrix Pencil Fit"),
        (results_ohmic_es_fit, P11p, "b-.", "ESPRIT Fit"),
        (results_ohmic_aaa_fit, P11p, "r-.", "Matrix AAA Fit"),
        (results_ohmic_espira_fit, P11p, "k", "ESPIRA I Fit"),
        (results_ohmic_espira2_fit, P11p, "--", "ESPIRA II Fit"),

    ],
    axes=axes,
)
axes.set_ylabel(r"$\rho_{11}$", fontsize=30)
axes.set_xlabel(r"$t\;\omega_c$", fontsize=30)
axes.legend(loc=0, fontsize=20);
axes.set_yscale("log")
```

## About

```{code-cell} ipython3
qutip.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
tol=1e-2
assert np.allclose(
    expect(P11p, results_ohmic_ps_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
assert np.allclose(
    expect(P11p, results_ohmic_corr_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
assert np.allclose(
    expect(P11p, results_ohmic_aaa_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
assert np.allclose(
    expect(P11p, results_ohmic_mp_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
assert np.allclose(
    expect(P11p, results_ohmic_prony_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)

assert np.allclose(
    expect(P11p, results_ohmic_es_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
assert np.allclose(
    expect(P11p, results_ohmic_espira_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
assert np.allclose(
    expect(P11p, results_ohmic_espira2_fit.states),
    expect(P11p, results_ohmic_sd_fit.states),
    rtol=tol,
)
```
