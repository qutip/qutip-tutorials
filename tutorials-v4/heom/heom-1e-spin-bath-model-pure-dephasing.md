---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# HEOM 1e: Spin-Bath model (pure dephasing)

+++

## Introduction

The HEOM method solves the dynamics and steady state of a system and its environment, the latter of which is encoded in a set of auxiliary density matrices.

In this example we show the evolution of a single two-level system in contact with a single Bosonic environment.  The properties of the system are encoded in Hamiltonian, and a coupling operator which describes how it is coupled to the environment.

The Bosonic environment is implicitly assumed to obey a particular Hamiltonian (see paper), the parameters of which are encoded in the spectral density, and subsequently the free-bath correlation functions.

In the example below we show how to model the overdamped Drude-Lorentz Spectral Density, commonly used with the HEOM. We show how to do the Matsubara and Pade analytical decompositions, as well as how to fit the latter with a finite set of approximate exponentials.  This differs from examble 1a in that we assume that the system and coupling parts of the Hamiltonian commute, hence giving an analytically solvable ''pure dephasing'' model. This is a useful example to look at when introducing other approximations  (e.g., fitting of correlation functions) to check for validity/convergence against the analytical results.  (Note that, generally, for the fitting examples, the pure dephasing model is the 'worst possible case'.  

### Drude-Lorentz spectral density

The Drude-Lorentz spectral density is:

$$J(\omega)=\omega \frac{2\lambda\gamma}{{\gamma}^2 + \omega^2}$$

where $\lambda$ scales the coupling strength, and $\gamma$ is the cut-off frequency.
We use the convention,
\begin{equation*}
C(t) = \int_0^{\infty} d\omega \frac{J_D(\omega)}{\pi}[\coth(\beta\omega) \cos(\omega \tau) - i \sin(\omega \tau)]
\end{equation*}

With the HEOM we must use an exponential decomposition:

\begin{equation*}
C(t)=\sum_{k=0}^{k=\infty} c_k e^{-\nu_k t}
\end{equation*}

The Matsubara decomposition of the Drude-Lorentz spectral density is given by:

\begin{equation*}
    \nu_k = \begin{cases}
               \gamma               & k = 0\\
               {2 \pi k} / {\beta \hbar}  & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    c_k = \begin{cases}
               \lambda \gamma (\cot(\beta \gamma / 2) - i) / \hbar               & k = 0\\
               4 \lambda \gamma \nu_k / \{(nu_k^2 - \gamma^2)\beta \hbar^2 \}    & k \geq 1\\
           \end{cases}
\end{equation*}

Note that in the above, and the following, we set $\hbar = k_\mathrm{B} = 1$.

+++

## Setup

```{code-cell} ipython3
import contextlib
import time

import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.optimize import curve_fit

import qutip
from qutip import (
    Options,
    basis,
    expect,
    liouvillian,
    sigmax,
    sigmaz,
)
from qutip.nonmarkov.heom import (
    HEOMSolver,
    BosonicBath,
    DrudeLorentzBath,
    DrudeLorentzPadeBath,
)

%matplotlib inline
```

## Helper functions

Let's define some helper functions for calculating correlation function expansions, plotting results and timing how long operations take:

```{code-cell} ipython3
def cot(x):
    """ Vectorized cotangent of x. """
    return 1. / np.tan(x)


def coth(x):
    """ Vectorized hyperbolic cotangent of x. """
    return 1. / np.tanh(x)
```

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.

        Each plot in plots consists of (solver_result, measurement_operation,
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
        if m_op is None:
            t, exp = result
        else:
            t = result.times
            exp = np.real(expect(result.states, m_op))
        kw.setdefault("linewidth", 2)
        axes.plot(t, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig
```

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

## System and bath definition

And let us set up the system Hamiltonian, bath and system measurement operators:

+++

Here we set $H_{sys}=0$, which means the interaction Hamiltonian and the system Hamiltonian commute, and we can compare the numerical results to a known analytical one.  We could in principle keep $\epsilon \neq 0$, but it just introduces fast system oscillations, so it is more convenient to set it to zero.

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0.0  # Energy of the 2-level system.
Del = 0.0  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz()  # coupling operator

# Bath properties:
gamma = 0.5  # cut off frequency
lam = 0.1  # coupling strength
T = 0.5
beta = 1. / T

# HEOM parameters:
# cut off parameter for the bath:
NC = 6
# number of exponents to retain in the Matsubara expansion
# of the correlation function:
Nk = 3

# Times to solve for
tlist = np.linspace(0, 50, 1000)
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

To get a non-trivial result we prepare the initial state in a superposition, and see how the bath destroys the coherence.

```{code-cell} ipython3
# Initial state of the system.
psi = (basis(2, 0) + basis(2, 1)).unit()
rho0 = psi * psi.dag()
```

## Simulation 1: Matsubara decomposition, not using Ishizaki-Tanimura terminator

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    HEOMMats = HEOMSolver(Hsys, bath, NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

```{code-cell} ipython3
:tags: []

# Plot the results so far
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Matsubara"),
    (resultMats, P12p, 'r', "P12 Matsubara"),
]);
```

## Simulation 2: Matsubara decomposition (including terminator)

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    _, terminator = bath.terminator()
    Ltot = liouvillian(Hsys) + terminator
    HEOMMatsT = HEOMSolver(Ltot, bath, NC, options=options)

with timer("ODE solver time"):
    resultMatsT = HEOMMatsT.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Matsubara"),
    (resultMats, P12p, 'r', "P12 Matsubara"),
    (resultMatsT, P11p, 'b--', "P11 Matsubara and terminator"),
    (resultMatsT, P12p, 'r--', "P12 Matsubara and terminator"),
]);
```

## Simulation 3: Pade decomposition

As in example 1a, we can compare to Pade and Fitting approaches.

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = DrudeLorentzPadeBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
    HEOMPade = HEOMSolver(Hsys, bath, NC, options=options)

with timer("ODE solver time"):
    resultPade = HEOMPade.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
plot_result_expectations([
    (resultMatsT, P11p, 'b', "P11 Matsubara (+term)"),
    (resultMatsT, P12p, 'r', "P12 Matsubara (+term)"),
    (resultPade, P11p, 'b--', "P11 Pade"),
    (resultPade, P12p, 'r--', "P12 Pade"),
]);
```

## Simulation 4: Fitting approach

```{code-cell} ipython3
:tags: []

def c(t, Nk):
    """ Calculates real and imag. parts of the correlation function
        using Nk Matsubara terms.
    """
    vk = 2 * np.pi * T * np.arange(1, Nk)

    result = (
        lam * gamma * (-1.0j + cot(gamma * beta / 2.)) *
        np.exp(-gamma * t[None, :])
    )
    result += np.sum(
        (4 * lam * gamma * T * vk[:, None] / (vk[:, None]**2 - gamma**2)) *
        np.exp(-vk[:, None] * t[None, :]),
        axis=0,
    )
    result = result.squeeze(axis=0)

    return result


tlist_fit = np.linspace(0, 2, 10000)
lmaxmats = 15000

corr_ana = c(tlist_fit, lmaxmats)
corrRana, corrIana = np.real(corr_ana), np.imag(corr_ana)
```

```{code-cell} ipython3
def wrapper_fit_func(x, N, *args):
    """ Wrapper for fitting function. """
    a, b = args[0][:N], args[0][N:2*N]
    return fit_func(x, a, b)


def fit_func(x, a, b):
    """ Fitting function. """
    a = np.array(a)
    b = np.array(b)
    x = np.atleast_1d(np.array(x))
    return np.sum(
        a[:, None] * np.exp(b[:, None] * x[None, :]),
        axis=0,
    )


def fitter(ans, tlist, i):
    """ Compute the fit. """
    upper_a = abs(max(ans, key=np.abs)) * 10
    # set initial guess:
    guess = [ans[0]] * i + [0] * i
    # set bounds: a's = anything, b's = negative
    # sets lower bound
    b_lower = [-upper_a] * i + [-np.inf] * i
    # sets higher bound
    b_higher = [upper_a] * i + [0] * i
    param_bounds = (b_lower, b_higher)
    p1, p2 = curve_fit(
        lambda x, *params: wrapper_fit_func(x, i, params),
        tlist,
        ans,
        p0=guess,
        sigma=[0.01] * len(tlist),
        bounds=param_bounds,
        maxfev=1e8,
    )
    return p1[:i], p1[i:]


# Fits of the real part with up to 4 exponents
popt1 = []
for i in range(4):
    a, b = fitter(corrRana, tlist_fit, i + 1)
    popt1.append((a, b))
    y = fit_func(tlist_fit, a, b)
    plt.plot(tlist_fit, corrRana, label="C_R(t)")
    plt.plot(tlist_fit, y, label=f"Fit with k={i + 1}")
    plt.xlabel("t")
    plt.ylabel("C_R(t)")
    plt.legend()
    plt.show()

# Fit of the imaginary part with 1 exponent
popt2 = []
for i in range(1):
    a, b = fitter(corrIana, tlist_fit, i + 1)
    popt2.append((a, b))
    y = fit_func(tlist_fit, a, b)
    plt.plot(tlist_fit, corrIana, label="C_I(t)")
    plt.plot(tlist_fit, y, label=f"Fit with k={i + 1}")
    plt.xlabel("t")
    plt.ylabel("C_I(t)")
    plt.legend()
    plt.show()
```

```{code-cell} ipython3
# Set the exponential coefficients from the fit parameters

ckAR = popt1[-1][0]
vkAR = -1 * popt1[-1][1]

ckAI = popt2[-1][0]
vkAI = -1 * popt2[-1][1]

# The imaginary fit can also be determined analytically and is
# a single term:
#
# ckAI = [complex(lam * gamma * (-1.0))]
# vkAI = [complex(gamma)]
```

```{code-cell} ipython3
options = Options(nsteps=15000, store_states=True, rtol=1e-14, atol=1e-14)

with timer("RHS construction time"):
    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)
    HEOMFit = HEOMSolver(Hsys, bath, NC, options=options)

with timer("ODE solver time"):
    resultFit = HEOMFit.run(rho0, tlist)
```

## Analytic calculations

```{code-cell} ipython3
def pure_dephasing_evolution_analytical(tlist, wq, ck, vk):
    """
    Computes the propagating function appearing in the pure dephasing model.

    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.

    wq: float
        The qubit frequency in the Hamiltonian.

    ck: ndarray
        The list of coefficients in the correlation function.

    vk: ndarray
        The list of frequencies in the correlation function.

    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    evolution = np.array([
        np.exp(-1j * wq * t - correlation_integral(t, ck, vk))
        for t in tlist
    ])
    return evolution


def correlation_integral(t, ck, vk):
    r"""
    Computes the integral sum function appearing in the pure dephasing model.

    If the correlation function is a sum of exponentials then this sum
    is given by:

    .. math:

        \int_0^{t}d\tau D(\tau) = \sum_k\frac{c_k}{\mu_k^2}e^{\mu_k t}
        + \frac{\bar c_k}{\bar \mu_k^2}e^{\bar \mu_k t}
        - \frac{\bar \mu_k c_k + \mu_k \bar c_k}{\mu_k \bar \mu_k} t
        + \frac{\bar \mu_k^2 c_k + \mu_k^2 \bar c_k}{\mu_k^2 \bar \mu_k^2}

    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.

    ck: ndarray
        The list of coefficients in the correlation function.

    vk: ndarray
        The list of frequencies in the correlation function.

    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    t1 = np.sum(
        (ck / vk**2) *
        (np.exp(vk * t) - 1)
    )
    t2 = np.sum(
        (ck.conj() / vk.conj()**2) *
        (np.exp(vk.conj() * t) - 1)
    )
    t3 = np.sum(
        (ck / vk + ck.conj() / vk.conj()) * t
    )
    return 2 * (t1 + t2 - t3)
```

For the pure dephasing analytics, we just sum up as many matsubara terms as we can:

```{code-cell} ipython3
lmaxmats2 = 15000

vk = [complex(-gamma)]
vk.extend([
    complex(-2. * np.pi * k * T)
    for k in range(1, lmaxmats2)
])

ck = [complex(lam * gamma * (-1.0j + cot(gamma * beta / 2.)))]
ck.extend([
    complex(4 * lam * gamma * T * (-v) / (v**2 - gamma**2))
    for v in vk[1:]
])

P12_ana = 0.5 * pure_dephasing_evolution_analytical(
    tlist, 0, np.asarray(ck), np.asarray(vk)
)
```

Alternatively, we can just do the integral of the propagator directly, without using the correlation functions at all

```{code-cell} ipython3
def JDL(omega, lamc, omega_c):
    return 2. * lamc * omega * omega_c / (omega_c**2 + omega**2)


def integrand(omega, lamc, omega_c, Temp, t):
    return (
        (-4. * JDL(omega, lamc, omega_c) / omega**2) *
        (1. - np.cos(omega*t)) * (coth(omega/(2.*Temp)))
        / np.pi
    )


P12_ana2 = [
    0.5 * np.exp(
        scipy.integrate.quad(integrand, 0, np.inf, args=(lam, gamma, T, t))[0]
    )
    for t in tlist
]
```

## Compare results

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultMatsT, P12p, 'r--', "P12 Mats + Term"),
    (resultPade, P12p, 'b--', "P12 Pade"),
    (resultFit, P12p, 'g', "P12 Fit"),
    ((tlist, np.real(P12_ana)), None, 'b', "Analytic 1"),
    ((tlist, np.real(P12_ana2)), None, 'y--', "Analytic 2"),
]);
```

We can't see much difference in the plot above, so let's do a log plot instead:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

plot_result_expectations([
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultMatsT, P12p, 'r--', "P12 Mats + Term"),
    (resultPade, P12p, 'b-.', "P12 Pade"),
    (resultFit, P12p, 'g', "P12 Fit"),
    ((tlist, np.real(P12_ana)), None, 'b', "Analytic 1"),
    ((tlist, np.real(P12_ana2)), None, 'y--', "Analytic 2"),
], axes)

axes.set_yscale('log')
axes.legend(loc=0, fontsize=12);
```

## About

```{code-cell} ipython3
qutip.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
assert np.allclose(
    expect(P12p, resultMats.states[:15]), np.real(P12_ana)[:15],
    rtol=1e-2,
)
assert np.allclose(
    expect(P12p, resultMatsT.states[:100]), np.real(P12_ana)[:100],
    rtol=1e-3,
)
assert np.allclose(
    expect(P12p, resultPade.states[:100]), np.real(P12_ana)[:100],
    rtol=1e-3,
)
assert np.allclose(
    expect(P12p, resultFit.states[:50]), np.real(P12_ana)[:50],
    rtol=1e-3,
)
assert np.allclose(P12_ana, P12_ana2, rtol=1e-3)
```
