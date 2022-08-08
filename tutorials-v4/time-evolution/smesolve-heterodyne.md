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

# Stochastic Solver: Heterodyne detection


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qutip import (Options, about, coherent, destroy, general_stochastic,
                   ket2dm, lindblad_dissipator, liouvillian, mesolve,
                   parallel_map, plot_expectation_values, smesolve, spost,
                   spre, stochastic_solvers)
from qutip.expect import expect_rho_vec

%matplotlib inline
```

## Introduction


Homodyne and hetrodyne detection are techniques for measuring the quadratures of a field using photocounters. Homodyne detection (on-resonant) measures one quadrature and with heterodyne detection (off-resonant) both quadratures can be detected simulateously.

The evolution of a quantum system that is coupled to a field that is monitored with homodyne and heterodyne detector can be described with stochastic master equations. This notebook compares two different ways to implement the heterodyne detection stochastic master equation in QuTiP.


## Deterministic reference

```python
N = 15
w0 = 1.0 * 2 * np.pi
A = 0.1 * 2 * np.pi
times = np.linspace(0, 15, 201)
gamma = 0.25

ntraj = 50
nsubsteps = 50

a = destroy(N)
x = a + a.dag()
y = -1.0j * (a - a.dag())

H = w0 * a.dag() * a + A * (a + a.dag())

rho0 = coherent(N, np.sqrt(5.0), method="analytic")
c_ops = [np.sqrt(gamma) * a]
e_ops = [a.dag() * a, x, y]
```

```python
result_ref = mesolve(H, rho0, times, c_ops, e_ops)
```

```python
plot_expectation_values(result_ref);
```

## Heterodyne implementation #1

<!-- #region -->
Stochastic master equation for heterodyne in Milburn's formulation

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + \gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_1(t) \sqrt{\gamma} \mathcal{H}[a] \rho(t) + \frac{1}{\sqrt{2}} dW_2(t) \sqrt{\gamma} \mathcal{H}[-ia] \rho(t)$

where $\mathcal{D}$ is the standard Lindblad dissipator superoperator, and $\mathcal{H}$ is defined as above,
and $dW_i(t)$ is a normal distributed increment with $E[dW_i(t)] = \sqrt{dt}$.


In QuTiP format we have:

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt +  D_{1}[A]\rho(t) dt + D_{2}^{(1)}[A]\rho(t) dW_1 + D_{2}^{(2)}[A]\rho(t) dW_2$

where $A = \sqrt{\gamma} a$, so we can identify
<!-- #endregion -->

$\displaystyle D_{1}[A]\rho = \gamma \mathcal{D}[a]\rho = \mathcal{D}[A]\rho$

```python
L = liouvillian(H)
D = lindblad_dissipator(c_ops[0])
d1_operator = L + D


def d1_rho_func(t, rho_vec):
    return d1_operator * rho_vec
```

$D_{2}^{(1)}[A]\rho = \frac{1}{\sqrt{2}} \sqrt{\gamma} \mathcal{H}[a] \rho =
\frac{1}{\sqrt{2}} \mathcal{H}[A] \rho =
\frac{1}{\sqrt{2}}(A\rho + \rho A^\dagger - \mathrm{Tr}[A\rho + \rho A^\dagger] \rho)
\rightarrow \frac{1}{\sqrt{2}} \left\{(A_L +  A_R^\dagger)\rho_v - \mathrm{Tr}[(A_L +  A_R^\dagger)\rho_v] \rho_v\right\}$

$D_{2}^{(2)}[A]\rho = \frac{1}{\sqrt{2}} \sqrt{\gamma} \mathcal{H}[-ia] \rho 
= \frac{1}{\sqrt{2}} \mathcal{H}[-iA] \rho =
\frac{-i}{\sqrt{2}}(A\rho - \rho A^\dagger - \mathrm{Tr}[A\rho - \rho A^\dagger] \rho)
\rightarrow \frac{-i}{\sqrt{2}} \left\{(A_L -  A_R^\dagger)\rho_v - \mathrm{Tr}[(A_L - A_R^\dagger)\rho_v] \rho_v\right\}$

```python
B1 = spre(c_ops[0]) + spost(c_ops[0].dag())
B2 = spre(c_ops[0]) + spost(c_ops[0].dag())


def d2_rho_func(t, rho_vec):
    e1 = expect_rho_vec(B1.data, rho_vec, False)
    drho1 = B1 * rho_vec - e1 * rho_vec

    e1 = expect_rho_vec(B2.data, rho_vec, False)
    drho2 = B2 * rho_vec - e1 * rho_vec

    return np.vstack([1.0 / np.sqrt(2) * drho1, -1.0j / np.sqrt(2) * drho2])
```

The heterodyne currents for the $x$ and $y$ quadratures are

$J_x(t) = \sqrt{\gamma}\left<x\right> + \sqrt{2} \xi(t)$

$J_y(t) = \sqrt{\gamma}\left<y\right> + \sqrt{2} \xi(t)$

where $\xi(t) = \frac{dW}{dt}$.

In qutip we define these measurement operators using the `m_ops = [[x, y]]` and the coefficients to the noise terms `dW_factor = [sqrt(2/gamma), sqrt(2/gamma)]`.

```python
result = general_stochastic(
    ket2dm(rho0),
    times,
    d1_rho_func,
    d2_rho_func,
    e_ops=[spre(op) for op in e_ops],
    len_d2=2,
    ntraj=ntraj,
    nsubsteps=nsubsteps,
    solver="platen",
    dW_factors=[np.sqrt(2 / gamma), np.sqrt(2 / gamma)],
    m_ops=[spre(x), spre(y)],
    store_measurement=True,
    map_func=parallel_map,
)
```

```python
plot_expectation_values([result, result_ref]);
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times, m[:, 0].real, "b", alpha=0.05)
    ax.plot(times, m[:, 1].real, "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_ylim(-10, 10)
ax.set_xlim(0, times.max())
ax.set_xlabel("time", fontsize=12)
ax.plot(times, np.array(result.measurement).mean(axis=0)[:, 0].real, "k", lw=2)
ax.plot(times, np.array(result.measurement).mean(axis=0)[:, 1].real, "k", lw=2);
```

## Heterodyne implementation #2: using two homodyne measurements

<!-- #region -->

We can also write the heterodyne equation as

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + \frac{1}{2}\gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_1(t) \sqrt{\gamma} \mathcal{H}[a] \rho(t) + \frac{1}{2}\gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_2(t) \sqrt{\gamma} \mathcal{H}[-ia] \rho(t)$


And using the QuTiP format for two stochastic collapse operators, we have:

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + D_{1}[A_1]\rho(t) dt + D_{2}[A_1]\rho(t) dW_1 + D_{1}[A_2]\rho(t) dt + D_{2}[A_2]\rho(t) dW_2$

so we can also identify
<!-- #endregion -->

$\displaystyle D_{1}[A_1]\rho = \frac{1}{2}\gamma \mathcal{D}[a]\rho = \mathcal{D}[\sqrt{\gamma}a/\sqrt{2}]\rho = \mathcal{D}[A_1]\rho$

$\displaystyle D_{1}[A_2]\rho = \frac{1}{2}\gamma \mathcal{D}[a]\rho = \mathcal{D}[-i\sqrt{\gamma}a/\sqrt{2}]\rho = \mathcal{D}[A_2]\rho$



$D_{2}[A_1]\rho = \frac{1}{\sqrt{2}} \sqrt{\gamma} \mathcal{H}[a] \rho = \mathcal{H}[A_1] \rho$

$D_{2}[A_2]\rho = \frac{1}{\sqrt{2}} \sqrt{\gamma} \mathcal{H}[-ia] \rho  = \mathcal{H}[A_2] \rho $


where $A_1 = \sqrt{\gamma} a / \sqrt{2}$ and $A_2 = -i \sqrt{\gamma} a / \sqrt{2}$.


In summary we have

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + \sum_i\left\{\mathcal{D}[A_i]\rho(t) dt + \mathcal{H}[A_i]\rho(t) dW_i\right\}$

which is a simultaneous homodyne detection with $A_1 = \sqrt{\gamma}a/\sqrt{2}$ and $A_2 = -i\sqrt{\gamma}a/\sqrt{2}$


Here the two heterodyne currents for the $x$ and $y$ quadratures are

$J_x(t) = \sqrt{\gamma/2}\left<x\right> + \xi(t)$

$J_y(t) = \sqrt{\gamma/2}\left<y\right> + \xi(t)$

where $\xi(t) = \frac{dW}{dt}$.

In qutip we can use the predefined homodyne solver for solving this problem.

```python
opt = Options()
opt.store_states = True
result = smesolve(
    H,
    rho0,
    times,
    [],
    [np.sqrt(gamma / 2) * a, -1.0j * np.sqrt(gamma / 2) * a],
    e_ops,
    ntraj=100,
    nsubsteps=nsubsteps,
    solver="taylor15",
    m_ops=[x, y],
    dW_factors=[np.sqrt(2 / gamma), np.sqrt(2 / gamma)],
    method="homodyne",
    store_measurement=True,
    map_func=parallel_map,
)
```

```python
plot_expectation_values([result, result_ref]);
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times, m[:, 0].real, "b", alpha=0.05)
    ax.plot(times, m[:, 1].real, "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_xlim(0, times.max())
ax.set_ylim(-25, 25)
ax.set_xlabel("time", fontsize=12)
ax.plot(times, np.array(result.measurement).mean(axis=0)[:, 0].real, "k", lw=2)
ax.plot(times, np.array(result.measurement).mean(axis=0)[:, 1].real, "k", lw=2);
```

## Implementation #3: builtin function for heterodyne

```python
result = smesolve(
    H,
    rho0,
    times,
    [],
    [np.sqrt(gamma) * a],
    e_ops,
    ntraj=ntraj,
    nsubsteps=nsubsteps,
    solver="taylor15",
    method="heterodyne",
    store_measurement=True,
    map_func=parallel_map,
)
```

```python
plot_expectation_values([result, result_ref]);
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times, m[:, 0, 0].real / np.sqrt(gamma), "b", alpha=0.05)
    ax.plot(times, m[:, 0, 1].real / np.sqrt(gamma), "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_xlim(0, times.max())
ax.set_ylim(-15, 15)
ax.set_xlabel("time", fontsize=12)
ax.plot(
    times,
    np.array(result.measurement).mean(axis=0)[:, 0, 0].real / np.sqrt(gamma),
    "k",
    lw=2,
)
ax.plot(
    times,
    np.array(result.measurement).mean(axis=0)[:, 0, 1].real / np.sqrt(gamma),
    "k",
    lw=2,
);
```

## Common problem

For some systems, the resulting density matrix can become unphysical due to the accumulation of computation error.

```python
N = 5
w0 = 1.0 * 2 * np.pi
A = 0.1 * 2 * np.pi
times = np.linspace(0, 15, 301)
gamma = 0.25

ntraj = 150
nsubsteps = 50

a = destroy(N)
x = a + a.dag()
y = -1.0j * (a - a.dag())

H = w0 * a.dag() * a + A * (a + a.dag())

rho0 = coherent(N, np.sqrt(5.0), method="analytic")
c_ops = [np.sqrt(gamma) * a]
e_ops = [a.dag() * a, x, y]

opt = Options()
opt.store_states = True
result = smesolve(
    H,
    rho0,
    times,
    [],
    [np.sqrt(gamma) * a],
    e_ops,
    ntraj=1,
    nsubsteps=5,
    solver="euler",
    method="heterodyne",
    store_measurement=True,
    map_func=parallel_map,
    options=opt,
    normalize=False,
)
```

```python
result.states[0][100]
```

```python
sp.linalg.eigh(result.states[0][10].full())
```

<!-- #region -->
Using smaller integration steps by increasing the nsubstep will lower the numerical errors.  
The solver algorithm used affect the convergence and numerical error.
Notable solvers are:  
- euler: order 0.5 fastest, but lowest order. Only solver that accept non-commuting sc_ops
- rouchon: order 1.0?, build to keep the density matrix physical
- taylor1.5: order 1.5, default solver, reasonably fast for good convergence.
- taylor2.0: order 2.0, even better convergence but can only take 1 homodyne sc_ops.


To list list all available solver, use help(stochastic_solvers)
<!-- #endregion -->

```python
help(stochastic_solvers)
```

## About

```python
about()
```

```python

```
