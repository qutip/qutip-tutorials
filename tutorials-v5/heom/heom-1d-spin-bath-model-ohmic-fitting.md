---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
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
    liouvillian,
    sigmax,
    sigmaz,
    spost,
    spre,
)
from qutip.solver.heom import (
    HEOMSolver,
    BosonicBath,
   FitSpectral,
    FitCorr,
    OhmicBath,
)

# Import mpmath functions for evaluation of gamma and zeta
# functions in the expression for the correlation:

from mpmath import mp

mp.dps = 15
mp.pretty = True

%matplotlib inline
```

## Helper functions

Let's define some helper functions for plotting the resutls

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

+++

### System Hamiltonian

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0    # Energy of the 2-level system.
Del = 0.2    # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# Initial state of the system.
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

### Analytical expressions for the Ohmic bath correlation function and spectral density

+++

Before we begin fitting, let us examine the analytic expressions for the correlation and spectral density functions and write Python equivalents. 

The correlation function is given by (see, e.g., http://www1.itp.tu-berlin.de/brandes/public_html/publications/notes.pdf for a derivation, equation 7.59, but with a factor of $\pi$ moved into the definition of the correlation function):

\begin{align}
C(t) =& \: \frac{1}{\pi}\alpha \omega_{c}^{1 - s} \beta^{- (s + 1)} \: \times \\
      & \: \Gamma(s + 1) \left[ \zeta \left(s + 1, \frac{1 + \beta \omega_c - i \omega_c t}{\beta \omega_c}\right) + \zeta \left(s + 1, \frac{1 + i \omega_c t}{\beta \omega_c}\right) \right]
\end{align}

where $\Gamma$ is the Gamma function and

\begin{equation}
\zeta(z, u) \equiv \sum_{n=0}^{\infty} \frac{1}{(n + u)^z}, \; u \neq 0, -1, -2, \ldots
\end{equation}

is the generalized Zeta function. The Ohmic case is given by $s = 1$.

The corresponding spectral density for the Ohmic case is:

\begin{equation}
J(\omega) = \omega \alpha e^{- \frac{\omega}{\omega_c}}
\end{equation}

```{code-cell} ipython3
def ohmic_correlation(t, alpha, wc, beta, s=1):
    """ The Ohmic bath correlation function as a function of t
        (and the bath parameters).
    """
    corr = (
        (1 / np.pi) * alpha * wc**(1 - s) * beta**(-(s + 1)) * mp.gamma(s + 1)
    )
    z1_u = (1 + beta * wc - 1.0j * wc * t) / (beta * wc)
    z2_u = (1 + 1.0j * wc * t) / (beta * wc)
    # Note: the arguments to zeta should be in as high precision as possible.
    # See http://mpmath.org/doc/current/basics.html#providing-correct-input
    return np.array([
        complex(corr * (mp.zeta(s + 1, u1) + mp.zeta(s + 1, u2)))
        for u1, u2 in zip(z1_u, z2_u)
    ], dtype=np.complex128)
```

```{code-cell} ipython3
def ohmic_spectral_density(w, alpha, wc):
    """ The Ohmic bath spectral density as a function of w
        (and the bath parameters).
    """
    return w * alpha * np.e**(-w / wc)
```

```{code-cell} ipython3
def ohmic_power_spectrum(w, alpha, wc, beta):
    """ The Ohmic bath power spectrum as a function of w
        (and the bath parameters).
    """
    return (
        w * alpha * np.e**(-abs(w) / wc) *
        ((1 / (np.e**(w * beta) - 1)) + 1) * 2
    )
```

### Bath and HEOM parameters

+++

Finally, let's set the bath parameters we will work with and write down some measurement operators:

```{code-cell} ipython3
Q = sigmaz()
alpha = 3.25
T = 0.5
wc= 1.0
s= 1
```

And set the cut-off for the HEOM hierarchy:

```{code-cell} ipython3
# HEOM parameters:

# The max_depth defaults to 5 so that the notebook executes more
# quickly. Change it to 11 to wait longer for more accurate results.
max_depth = 5
```

## Building the HEOM bath by fitting the spectral density

+++

We begin by fitting the spectral density, using a series of $k$ underdamped harmonic oscillators case with the Meier-Tannor form (J. Chem. Phys. 111, 3365 (1999); https://doi.org/10.1063/1.479669):

\begin{equation}
J_{\mathrm approx}(\omega; a, b, c) = \sum_{i=0}^{k-1} \frac{2 a_i b_i w}{((w + c_i)^2 + b_i^2) ((w - c_i)^2 + b_i^2)}
\end{equation}

where $a, b$ and $c$ are the fit parameters and each is a vector of length $k$.

+++

This type of fit can be performed quickly using the FitSpectral bath, which takes the target spectral density as an array and fits it with a series of **k** underdamped harmonic oscillators with the Meier-Tannor form.

```{code-cell} ipython3
w = np.linspace(0, 15, 20000)
J = ohmic_spectral_density(w, alpha, wc)
```

We first initialize our FitSpectral class

```{code-cell} ipython3
fs=FitSpectral(T,Q,Nk=4)
```

To obtain a fit we simply pass our desired spectral density and range, into the get_fit method

```{code-cell} ipython3
fs.get_fit(J,w)
```

To obtain an overview of the results of the fit we may call the summary method

```{code-cell} ipython3
fs.summary()
```

By default the get_fit method, has a threshold normalized root mean squared error (NRMSE) of $5\times 10^{-6}$ and selects the number of oscillators automatically to obtain that value, one may on the other hand specify the Number of oscillators that can be done using the optional argument N, or may want a more accurate NRMSE, which can be specified with the final_rmse optional argument

+++

or by requiring a lower NRMSE

```{code-cell} ipython3
fs.get_fit(J,w,final_rmse=2e-6)
```

```{code-cell} ipython3
fs.summary()
```

Alternatively one may choose the number of oscillators in the fit instead of a desired NRMSE

```{code-cell} ipython3
fs.get_fit(J,w,N=4)
```

```{code-cell} ipython3
fs.summary()
```

Let's take a closer look at our last fit by plotting the contribution of each term of the fit:

```{code-cell} ipython3
# Plot the components of the fit separately:
plt.rcParams['font.size'] = 25
plt.rcParams['figure.figsize'] = (10,5)
def plot_fit(func,J, w, lam, gamma, w0):
    """ Plot the individual components of a fit to the spectral density. and how they contribute to the full as an animation """
    total=0
    plt.plot(w, J, 'r--', linewidth=2, label="original")
    for i in range(len(lam)):
        component=func(w,[lam[i]],[gamma[i]],[w0[i]])
        total+=component
        plt.plot(w, J, 'r--', linewidth=2, label="original")
        plt.plot(w,total,label=rf"$k={i+1}$")
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$J(\omega)$")
        plt.legend()
        plt.pause(1)
        plt.show()
def plot_fit_components(func,J, w, lam, gamma, w0):
    """ Plot the individual components of a fit to the spectral density. and how they contribute to the full as an animation """
    total=0
    plt.plot(w, J, 'r--', linewidth=2, label="original")
    for i in range(len(lam)):
        component=func(w,[lam[i]],[gamma[i]],[w0[i]])
        plt.plot(w,component,label=rf"$k={i+1}$")
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$J(\omega)$")
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.show()
lam, gamma, w0 = fs.params_spec
plot_fit(fs.spectral_density_approx,J, w, lam, gamma, w0)
```

```{code-cell} ipython3
plot_fit_components(fs.spectral_density_approx,J, w, lam, gamma, w0)
```

And let's also compare the power spectrum of the fit and the analytical spectral density:

```{code-cell} ipython3
plt.rcParams['figure.figsize'] = (10,5)

def plot_power_spectrum(alpha, wc, beta, lam, gamma, w0, save=True):
    """ Plot the power spectrum of a fit against the actual power spectrum. """
    w = np.linspace(-10, 10, 50000)
    s_orig = ohmic_power_spectrum(w, alpha=alpha, wc=wc, beta=beta)
    s_fit = fs.power_spectrum(w,T)
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(w, s_orig, 'r', linewidth=2, label="original")
    axes.plot(w, s_fit, 'b', linewidth=2, label="fit")

    axes.set_xlabel(r'$\omega$', fontsize=28)
    axes.set_ylabel(r'$S(\omega)$', fontsize=28)
    axes.legend()

    if save:
        fig.savefig('powerspectrum.eps')


plot_power_spectrum(alpha, wc, 1/T, lam, gamma, w0, save=False)
```

Now that we have a good fit to the spectral density, Let us obtain its dynamics, by passing our FitSpectral bath specifications into the HEOMSolver

```{code-cell} ipython3
tlist = np.linspace(0, 30 * np.pi / Del, 600)
options = {'nsteps':15000, 'store_states':True, 'rtol':1e-12, 'atol':1e-12, 'method':"bdf"}
Ltot = liouvillian(Hsys) + fs.terminator
HEOM_spectral_fit = HEOMSolver(Ltot, fs.Bath_spec, max_depth=4, options=options,)
result_spectral=HEOM_spectral_fit.run(rho0,tlist)
```

Now if we want to see the systems's behaviour as we change the Number of terms in the fit, we may use this auxiliary function

```{code-cell} ipython3
def generate_spectrum_results(Q,beta, N, Nk, max_depth):
    """ Run the HEOM with the given bath parameters and
        and return the results of the evolution.
    """
    fs=FitSpectral(T,Q,Nk)
    fs.get_fit(J,w,N)
    Ltot = liouvillian(Hsys) + fs.terminator
    tlist = np.linspace(0, 30 * np.pi / Del, 600)

    # This problem is a little stiff, so we use  the BDF method to solve
    # the ODE ^^^
    print(f'Starting calculations for N={N} and max_depth={max_depth} ... \n ')
    HEOM_spectral_fit = HEOMSolver(
        Ltot, fs.Bath_spec, max_depth=max_depth, options=options,
    )
    results_spectral_fit = (HEOM_spectral_fit.run(rho0, tlist))
    print('\n')
    return results_spectral_fit
```

Below we generate results for different convergence parameters (number of terms in the fit, number of matsubara terms, and depth of the hierarchy).  For the parameter choices here, we need a relatively large depth of around '11', which can be a little slow.

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.

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
        if color == 'rand':
            axes.plot(
                result.times, exp,
                c=np.random.rand(3,), label=label, **kw,
            )
        else:
            axes.plot(result.times, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig
```

```{code-cell} ipython3
# Generate results for different number of lorentzians in fit:

results_spectral_fit_pk = [
    generate_spectrum_results(Q,1/T, n, Nk=1, max_depth=max_depth)
    for n in range(1,5)
]

plot_result_expectations([
    (
        result, P11p, 'rand',
        f"P11 (spectral fit) $k_J$={pk + 1}",
    )
    for pk, result in enumerate(results_spectral_fit_pk)
]);
```

```{code-cell} ipython3
# generate results for different number of Matsubara terms per Lorentzian
# for max number of Lorentzians:

Nk_list = range(2, 4)
results_spectral_fit_nk = [
    generate_spectrum_results(Q,1/T, 4, Nk=Nk, max_depth=max_depth)
    for Nk in Nk_list
]

plot_result_expectations([
    (
        result, P11p, 'rand',
        f"P11 (spectral fit) K={nk}",
    )
    for nk, result in zip(Nk_list, results_spectral_fit_nk)
]);
```

```{code-cell} ipython3
# Generate results for different depths:

Nc_list = range(2, max_depth)
results_spectral_fit_nc = [
    generate_spectrum_results(Q,1/T, 4, Nk=1, max_depth=Nc)
    for Nc in Nc_list
]

plot_result_expectations([
    (
        result, P11p, 'rand',
        f"P11 (spectral fit) $N_C={nc}$",
    )
    for nc, result in zip(Nc_list, results_spectral_fit_nc)
]);
```

We now combine the fitting and correlation function data into one large plot.

```{code-cell} ipython3
t = np.linspace(0, 15, 100)
C =ohmic_correlation(t, alpha=alpha, wc=wc, beta=1/T)
w2 = np.concatenate(
    [np.linspace(-10, -0.1, 5000),
     np.linspace(0.1, 10, 5000)],
)
S=ohmic_power_spectrum(w2,alpha=alpha,beta=1/T,wc=wc)


fs.fit_plots(w,J,t, C,w2,S);
```

## Building the HEOM bath by fitting the correlation function

+++

Having successfully fitted the spectral density and used the result to calculate the Matsubara expansion and terminator for the HEOM bosonic bath, we now proceed to the second case of fitting the correlation function itself instead.

Here we fit the real and imaginary parts separately, using the following ansatz

$$C_R^F(t) = \sum_{i=1}^{k_R} c_R^ie^{-\gamma_R^i t}\cos(\omega_R^i t)$$

$$C_I^F(t) = \sum_{i=1}^{k_I} c_I^ie^{-\gamma_I^i t}\sin(\omega_I^i t)$$

Analogously to the spectral density case, one may use the FitCorr class

```{code-cell} ipython3
fc=FitCorr(Q)
```

```{code-cell} ipython3
t = np.linspace(0, 25, 1500)
C = ohmic_correlation(t, alpha=alpha, wc=wc, beta=1/T)
```

```{code-cell} ipython3
fc.fit_correlation(t,C,Ni=3,Nr=3)
```

```{code-cell} ipython3
fc.summary()
```

```{code-cell} ipython3
fc.fit_correlation(t,C,final_rmse=5e-5)
```

```{code-cell} ipython3
fc.summary()
```

Now we construct the `BosonicBath` co-efficients and frequencies from the fit to the correlation function:

```{code-cell} ipython3
t = np.linspace(0, 15, 100)
C = ohmic_correlation(t, alpha=alpha, wc=wc, beta=1/T)
fc.fit_plots(w, J, t, C, w2, S,beta=1/T)
```

```{code-cell} ipython3
def generate_corr_results(N, max_depth):
    tlist = np.linspace(0, 30 * np.pi / Del, 600)
    t = np.linspace(0, 15, 100)
    C = ohmic_correlation(t, alpha=alpha, wc=wc, beta=1/T)
    fc.fit_correlation(t,C,Ni=N,Nr=N)
    HEOM_corr_fit = HEOMSolver(
        Hsys, fc.Bath_corr, max_depth=max_depth, options=options,
    )

    results_corr_fit = (HEOM_corr_fit.run(rho0, tlist))

    return results_corr_fit


# Generate results for different number of lorentzians in fit:
results_corr_fit_pk = [
    print(f"{i + 1}") or generate_corr_results(i, max_depth=max_depth,
    )
    for i in range(1,4)
]
```

```{code-cell} ipython3
plot_result_expectations([
    (
        result, P11p, 'rand',
        f"P11 (correlation fit) k_R=k_I={pk + 1}",
    )
    for pk, result in enumerate(results_corr_fit_pk)
]);
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

plot_result_expectations([
    (
        results_corr_fit_pk[0], P11p,
        'y', "Correlation Function Fit $k_R=k_I=1$",
    ),
    (
        results_corr_fit_pk[2], P11p,
        'y-.', "Correlation Function Fit $k_R=k_I=3$",
    ),
    (results_spectral_fit_pk[0], P11p, 'b', "Spectral Density Fit $k_J=1$"),
    (results_spectral_fit_pk[2], P11p, 'g--', "Spectral Density Fit $k_J=3$"),
    (results_spectral_fit_pk[3], P11p, 'r-.', "Spectral Density Fit $k_J=4$"),
], axes=axes)

axes.set_yticks([0.6, 0.8, 1])
axes.set_ylabel(r'$\rho_{11}$', fontsize=30)
axes.set_xlabel(r'$t\;\omega_c$', fontsize=30)
axes.legend(loc=0, fontsize=20);
```

# Using the Ohmic Bath class

While the two classes above are designed for general fits of either correlation functions or spectral densities, as the ohmic spectrum is popular in the modeling of open quantum systems, it has its own dedicated class, the results above can be reproduced shortly by using the OhmicBath class. By default the method fits using the  spectral density, however it can use the correlation function if method is specified

```{code-cell} ipython3
obs=OhmicBath(T,Q,alpha,wc,s,rmse=9e-5,method='spectral')
```

```{code-cell} ipython3
obs.summary()
```

```{code-cell} ipython3
obs.fit.fit_plots(w,J,t, C,w2,S);
```

```{code-cell} ipython3
obc=OhmicBath(T,Q,alpha,wc,s,rmse=1e-4,method='correlation')
```

```{code-cell} ipython3
obc.summary()
```

```{code-cell} ipython3
obc.fit.fit_plots(w, J, t, C, w2, S,beta=1/T)
```

```{code-cell} ipython3
tlist = np.linspace(0, 30 * np.pi / Del, 600)

HEOM_ohmic_corr_fit = HEOMSolver(Hsys, obc.bath, max_depth=5, options=options,)
Ltot = liouvillian(Hsys) + fs.terminator
HEOM_ohmic_spectral_fit = HEOMSolver(Hsys, obs.bath, max_depth=5, options=options,)

#results__ohmic_corr_fit = (HEOM_ohmic_corr_fit.run(rho0, tlist))
results_ohmic_spectral_fit = (HEOM_ohmic_spectral_fit.run(rho0, tlist))
```

```{code-cell} ipython3
results_ohmic_corr_fit = (HEOM_ohmic_corr_fit.run(rho0, tlist))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

plot_result_expectations([
 #   (
 #       results_corr_fit_pk[0], P11p,
 #       'y', "Correlation Function Fit $k_R=k_I=1$",
 #   ),
    (
        results_corr_fit_pk[2], P11p,
        'y-.', "Correlation Function Fit $k_R=k_I=3$",
    ),
    (results_spectral_fit_pk[0], P11p, 'b', "Spectral Density Fit $k_J=1$"),
    (results_spectral_fit_pk[2], P11p, 'g--', "Spectral Density Fit $k_J=3$"),
    (results_spectral_fit_pk[3], P11p, 'r-.', "Spectral Density Fit $k_J=4$"),
    (results_ohmic_spectral_fit, P11p, 'g-.', "Spectral Density Fit Ohmic Bath"),
    (results_ohmic_corr_fit, P11p, 'k-.', "Correlation Fit Ohmic Bath")

], axes=axes)

axes.set_yticks([0.6, 0.8, 1])
axes.set_ylabel(r'$\rho_{11}$', fontsize=30)
axes.set_xlabel(r'$t\;\omega_c$', fontsize=30)
axes.legend(loc=0, fontsize=20);
```

## About

```{code-cell} ipython3
qutip.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
assert np.allclose(
    expect(P11p, results_spectral_fit_pk[2].states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)
```
