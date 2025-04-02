---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: qutip-dev
  language: python
  name: python3
---

# HEOM 1c: Spin-Bath model (Underdamped Case)

+++

## Introduction

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single Bosonic environment.  The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The Bosonic environment is implicitly assumed to obey a particular Hamiltonian ([see paper](https://arxiv.org/abs/2010.10806)), the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how to model the underdamped Brownian motion Spectral Density.

Note that in the following, we set $\hbar = k_\mathrm{B} = 1$.

###  Brownian motion (underdamped) spectral density
The underdamped spectral density is:

$$J_U = \frac{\alpha^2 \Gamma \omega}{(\omega_c^2 - \omega^2)^2 + \Gamma^2 \omega^2)}.$$

Here $\alpha$  scales the coupling strength, $\Gamma$ is the cut-off frequency, and $\omega_c$ defines a resonance frequency.  With the HEOM we must use an exponential decomposition:

The Matsubara decomposition of this spectral density is, in real and imaginary parts:



\begin{equation*}
    c_k^R = \begin{cases}
               \alpha^2 \coth(\beta( \Omega + i\Gamma/2)/2)/4\Omega & k = 0\\
               \alpha^2 \coth(\beta( \Omega - i\Gamma/2)/2)/4\Omega & k = 0\\
              -2\alpha^2\Gamma/\beta \frac{\epsilon_k }{((\Omega + i\Gamma/2)^2 + \epsilon_k^2)(\Omega - i\Gamma/2)^2 + \epsilon_k^2)}      & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    \nu_k^R = \begin{cases}
               -i\Omega  + \Gamma/2, i\Omega  +\Gamma/2,             & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}




\begin{equation*}
    c_k^I = \begin{cases}
               i\alpha^2 /4\Omega & k = 0\\
                -i\alpha^2 /4\Omega & k = 0\\
           \end{cases}
\end{equation*}

\begin{equation*}
    \nu_k^I = \begin{cases}
               i\Omega  + \Gamma/2, -i\Omega  + \Gamma/2,             & k = 0\\
           \end{cases}
\end{equation*}

Note that in the above, and the following, we set $\hbar = k_\mathrm{B} = 1$.

+++

## Setup

```{code-cell}
import contextlib
import time

import numpy as np
from matplotlib import pyplot as plt

import qutip
from qutip import (
    basis,
    brmesolve,
    destroy,
    expect,
    qeye,
    sigmax,
    sigmaz,
    tensor,
)
from qutip.solver.heom import (
    HEOMSolver,
)
from qutip.core.environment import (
    UnderDampedEnvironment,
    ExponentialBosonicEnvironment
)

%matplotlib inline
```

## Helper functions

Let's define some helper functions for calculating correlation function expansions, plotting results and timing how long operations take:

```{code-cell}
def cot(x):
    """ Vectorized cotangent of x. """
    return 1. / np.tan(x)
```

```{code-cell}
def coth(x):
    """ Vectorized hyperbolic cotangent of x. """
    return 1. / np.tanh(x)
```

```{code-cell}
def underdamped_matsubara_params(lam, gamma, T, nk):
    """ Calculation of the real and imaginary expansions of the
        underdamped correlation functions.
    """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.
    beta = 1. / T

    ckAR = [
        (lam**2 / (4*Om)) * coth(beta * (Om + 1.0j * Gamma) / 2),
        (lam**2 / (4*Om)) * coth(beta * (Om - 1.0j * Gamma) / 2),
    ]
    ckAR.extend(
        (-2 * lam**2 * gamma / beta) * (2 * np.pi * k / beta) /
        (((Om + 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2) *
         ((Om - 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2)) + 0.j
        for k in range(1, nk + 1)
    )
    vkAR = [
        -1.0j * Om + Gamma,
        1.0j * Om + Gamma,
    ]
    vkAR.extend(
        2 * np.pi * k * T + 0.j
        for k in range(1, nk + 1)
    )

    factor = 1. / 4

    ckAI = [
        -factor * lam**2 * 1.0j / Om,
        factor * lam**2 * 1.0j / Om,
    ]
    vkAI = [
        -(-1.0j * Om - Gamma),
        -(1.0j * Om - Gamma),
    ]

    return ckAR, vkAR, ckAI, vkAI
```

```{code-cell}
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.

        Each plot in plots consists of: (solver_result, measurement_operation,
        color, label).
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
        axes.plot(result.times, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig
```

```{code-cell}
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

```{code-cell}
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

```{code-cell}
# Defining the system Hamiltonian
eps = .5     # Energy of the 2-level system.
Del = 1.0    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell}
# Initial state of the system.
rho0 = basis(2, 0) * basis(2, 0).dag()
```

```{code-cell}
# System-bath coupling (underdamed spectral density)
Q = sigmaz()  # coupling operator

# Bath properties:
gamma = .1  # cut off frequency
lam = .5  # coupling strength
w0 = 1.  # resonance frequency
T = 1.
beta = 1. / T

# HEOM parameters:

# number of exponents to retain in the Matsubara expansion of the
# bath correlation function:
Nk = 2

# Number of levels of the hierarchy to retain:
NC = 10

# Times to solve for:
tlist = np.linspace(0, 50, 1000)
```

```{code-cell}
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

### First let us look at what the underdamped spectral density looks like:

```{code-cell}
def plot_spectral_density():
    """ Plot the underdamped spectral density """
    w = np.linspace(0, 5, 1000)
    J = lam**2 * gamma * w / ((w0**2 - w**2)**2 + (gamma**2) * (w**2))

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(w, J, 'r', linewidth=2)
    axes.set_xlabel(r'$\omega$', fontsize=28)
    axes.set_ylabel(r'J', fontsize=28)


plot_spectral_density()
```

The correlation functions are now very oscillatory, because of the Lorentzian peak in the spectral density.

+++

### So next, let us plot the correlation functions themselves:

```{code-cell}
def Mk(t, k, gamma, w0, beta):
    """ Calculate the Matsubara terms for a given t and k. """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.
    ek = 2 * np.pi * k / beta

    return (
        (-2 * lam**2 * gamma / beta) * ek * np.exp(-ek * np.abs(t))
        / (((Om + 1.0j * Gamma)**2 + ek**2) * ((Om - 1.0j * Gamma)**2 + ek**2))
    )


def c(t, Nk, lam, gamma, w0, beta):
    """ Calculate the correlation function for a vector of times, t. """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.

    Cr = (
        coth(beta * (Om + 1.0j * Gamma) / 2) * np.exp(1.0j * Om * t)
        + coth(beta * (Om - 1.0j * Gamma) / 2) * np.exp(-1.0j * Om * t)
    )

    Ci = np.exp(-1.0j * Om * t) - np.exp(1.0j * Om * t)

    return (
        (lam**2 / (4 * Om)) * np.exp(-Gamma * np.abs(t)) * (Cr + Ci) +
        np.sum([
            Mk(t, k, gamma=gamma, w0=w0, beta=beta)
            for k in range(1, Nk + 1)
        ], 0)
    )


def plot_correlation_function():
    """ Plot the underdamped correlation function. """
    t = np.linspace(0, 20, 1000)
    corr = c(t, Nk=3, lam=lam, gamma=gamma, w0=w0, beta=beta)

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(t, np.real(corr), '-', color="black", label="Re[C(t)]")
    axes.plot(t, np.imag(corr), '-', color="red", label="Im[C(t)]")
    axes.set_xlabel(r't', fontsize=28)
    axes.set_ylabel(r'C', fontsize=28)
    axes.legend(loc=0, fontsize=12)


plot_correlation_function()
```

It is useful to look at what the Matsubara contributions do to this spectral density. We see that they modify the real part around $t=0$:

```{code-cell}
def plot_matsubara_correlation_function_contributions():
    """ Plot the underdamped correlation function. """
    t = np.linspace(0, 20, 1000)

    M_Nk2 = np.sum([
        Mk(t, k, gamma=gamma, w0=w0, beta=beta)
        for k in range(1, 2 + 1)
    ], 0)

    M_Nk100 = np.sum([
        Mk(t, k, gamma=gamma, w0=w0, beta=beta)
        for k in range(1, 100 + 1)
    ], 0)

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(t, np.real(M_Nk2), '-', color="black", label="Re[M(t)] Nk=2")
    axes.plot(t, np.real(M_Nk100), '--', color="red", label="Re[M(t)] Nk=100")
    axes.set_xlabel(r't', fontsize=28)
    axes.set_ylabel(r'M', fontsize=28)
    axes.legend(loc=0, fontsize=12)


plot_matsubara_correlation_function_contributions()
```

## Solving for the dynamics as a function of time

+++

Next we calculate the exponents using the Matsubara decompositions. Here we split them into real and imaginary parts.

The HEOM code will optimize these, and reduce the number of exponents when real and imaginary parts have the same exponent. This is clearly the case for the first term in the vkAI and vkAR lists.

```{code-cell}
ckAR, vkAR, ckAI, vkAI = underdamped_matsubara_params(
    lam=lam, gamma=gamma, T=T, nk=Nk,
)
```

Having created the lists which specify the bath correlation functions, we create a `BosonicBath` from them and pass the bath to the `HEOMSolver` class.

The solver constructs the "right hand side" (RHS) determinining how the system and auxiliary density operators evolve in time. This can then be used to solve for dynamics or steady-state.

Below we create the bath and solver and then solve for the dynamics by calling `.run(rho0, tlist)`.

```{code-cell}
with timer("RHS construction time"):
    bath = ExponentialBosonicEnvironment(ckAR, vkAR, ckAI, vkAI)
    HEOMMats = HEOMSolver(Hsys, (bath,Q), NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

```{code-cell}
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
]);
```

In practice, one would not perform this laborious expansion for the underdamped correlation function, because
QuTiP already has a class, `UnderDampedEnvironment`, that can construct this bath for you. Nevertheless, knowing how
to perform this expansion is an useful skill.

Below we show how to use this built-in functionality:

```{code-cell}
# Compare to built-in under-damped bath:

with timer("RHS construction time"):
    bath = UnderDampedEnvironment(lam=lam, gamma=gamma, w0=w0, T=T)
    bath_approx=bath.approximate(method="matsubara",Nk=Nk)
    HEOM_udbath = HEOMSolver(Hsys, (bath_approx,Q), NC, options=options)

with timer("ODE solver time"):
    result_udbath = HEOM_udbath.run(rho0, tlist)
```

```{code-cell}
plot_result_expectations([
    (result_udbath, P11p, 'b', "P11 (UnderDampedEnvironment)"),
    (result_udbath, P12p, 'r', "P12 (UnderDampedEnvironment)"),
    (resultMats, P11p, 'r--', "P11 Mats"),
    (resultMats, P12p, 'b--', "P12 Mats"),
]);
```

The `UnderDampedEnvironment` class also allows us to easily evaluate analytical expressions for the power spectrum, correlation function, and spectral density. In the following plots, the solid lines are the exact expressions, and the dashed lines are based on our approximation of the correlation function with a finite number of exponents. In this case, there is an excellent agreement.

```{code-cell}
w = np.linspace(-3, 3, 1000)
w2 = np.linspace(0, 3, 1000)
t = np.linspace(0, 10, 1000)
bath_cf = bath.correlation_function(t)  

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(w, bath.power_spectrum(w))
axs[0, 0].plot(w, bath_approx.power_spectrum(w), '--')
axs[0, 0].set(xlabel=r'$\omega$', ylabel=r'$S(\omega)$')
axs[0, 1].plot(w2, bath.spectral_density(w2))
axs[0, 1].plot(w2, bath_approx.spectral_density(w2), '--')
axs[0, 1].set(xlabel=r'$\omega$', ylabel=r'$J(\omega)$')
axs[1, 0].plot(t, np.real(bath_cf))
axs[1, 0].plot(t, np.real(bath_approx.correlation_function(t)), '--')
axs[1, 0].set(xlabel=r'$t$', ylabel=r'$C_{R}(t)$')
axs[1, 1].plot(t, np.imag(bath_cf))
axs[1, 1].plot(t, np.imag(bath_approx.correlation_function(t)), '--')
axs[1, 1].set(xlabel=r'$t$', ylabel=r'$C_{I}(t)$')

fig.tight_layout()
plt.show()
```

## Compare the results

+++

### We can compare these results to those of the Bloch-Redfield solver in QuTiP:

```{code-cell}
with timer("ODE solver time"):
    resultBR = brmesolve(
        Hsys, rho0, tlist,
        a_ops=[[sigmaz(), bath]], options=options,
    )
```

```{code-cell}
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultBR, P11p, 'g--', "P11 Bloch Redfield"),
    (resultBR, P12p, 'g--', "P12 Bloch Redfield"),
]);
```

### Lastly, let us calculate the analytical steady-state result and compare all of the results:

+++

The thermal state of a reaction coordinate (treating the environment as a single damped mode) should, at high temperatures and small gamma, tell us the steady-state:

```{code-cell}
dot_energy, dot_state = Hsys.eigenstates()
deltaE = dot_energy[1] - dot_energy[0]

gamma2 = gamma
wa = w0  # reaction coordinate frequency
g = lam / np.sqrt(2 * wa)  # coupling

NRC = 10

Hsys_exp = tensor(qeye(NRC), Hsys)
Q_exp = tensor(qeye(NRC), Q)
a = tensor(destroy(NRC), qeye(2))

H0 = wa * a.dag() * a + Hsys_exp
# interaction
H1 = (g * (a.dag() + a) * Q_exp)

H = H0 + H1

energies, states = H.eigenstates()
rhoss = 0 * states[0] * states[0].dag()
for kk, energ in enumerate(energies):
    rhoss += (states[kk] * states[kk].dag() * np.exp(-beta * energies[kk]))
rhoss = rhoss / rhoss.norm()

P12RC = tensor(qeye(NRC), basis(2, 0) * basis(2, 1).dag())
P12RC = expect(rhoss, P12RC)

P11RC = tensor(qeye(NRC), basis(2, 0) * basis(2, 0).dag())
P11RC = expect(rhoss, P11RC)
```

```{code-cell}
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

```{code-cell}
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

with plt.rc_context(rcParams):
    plt.yticks([P11RC, 0.6, 1.0], [0.38, 0.6, 1])

    plot_result_expectations([
        (resultBR, P11p, 'y-.', "Bloch-Redfield"),
        (resultMats, P11p, 'b', "Matsubara $N_k=3$"),
    ], axes=axes)
    axes.plot(
        tlist, [P11RC for t in tlist],
        color='black', linestyle="-.", linewidth=2,
        label="Thermal state",
    )

    axes.set_xlabel(r'$t \Delta$', fontsize=30)
    axes.set_ylabel(r'$\rho_{11}$', fontsize=30)

    axes.locator_params(axis='y', nbins=4)
    axes.locator_params(axis='x', nbins=4)

    axes.legend(loc=0)

    fig.tight_layout()
```

## About

```{code-cell}
qutip.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell}
assert np.allclose(
    expect(P11p, resultMats.states[-100:]), P11RC, rtol=1e-2,
)
assert np.allclose(
    expect(P11p, resultBR.states[-100:]), P11RC, rtol=1e-2,
)
```
