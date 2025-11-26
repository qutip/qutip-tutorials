---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Stochastic Solver: Heterodyne Detection


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qutip import (
    SMESolver,
    about,
    coherent,
    destroy,
    mesolve,
    plot_expectation_values,
    smesolve,
)

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
times = np.linspace(0, 10, 201)
gamma = 0.25
ntraj = 50

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


In QuTiP, this is available with the stochactic master equation solver ``smesolve`` and ``SMESolver`` with heterodyne detection.
<!-- #endregion -->

The heterodyne currents for the $x$ and $y$ quadratures are

$J_x(t) = \sqrt{\gamma}\left<x\right> + \sqrt{2} \xi(t)$

$J_y(t) = \sqrt{\gamma}\left<y\right> + \sqrt{2} \xi(t)$

where $\xi(t) = \frac{dW}{dt}$.

In qutip, these measurements are build from the operators passed as ``sc_ops``.

```python
options = {"store_measurement": True, "map": "parallel"}

result = smesolve(
    H,
    rho0,
    times,
    sc_ops=c_ops,
    heterodyne=True,
    e_ops=e_ops,
    ntraj=ntraj,
    options=options,
)
```

```python
plot_expectation_values([result, result_ref])
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times[1:], m[0, 0, :].real, "b", alpha=0.05)
    ax.plot(times[1:], m[0, 1, :].real, "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_ylim(-10, 10)
ax.set_xlim(0, times.max())
ax.set_xlabel("time", fontsize=12)
ax.plot(times[1:], np.mean(result.measurement, axis=0)[0, 0, :].real, "k", lw=2)
ax.plot(times[1:], np.mean(result.measurement, axis=0)[0, 1, :].real, "k", lw=2)
```

## Heterodyne implementation #2: using two homodyne measurements

<!-- #region -->
We can also write the heterodyne equation as

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + \frac{1}{2}\gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_1(t) \sqrt{\gamma} \mathcal{H}[a] \rho(t) + \frac{1}{2}\gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_2(t) \sqrt{\gamma} \mathcal{H}[-ia] \rho(t)$


Which correspond to the homodyne detection with two stochastic collapse operators:  $A_1 = \sqrt{\gamma} a / \sqrt{2}$ and $A_2 = -i \sqrt{\gamma} a / \sqrt{2}$.
<!-- #endregion -->

Here the two homodyne currents associated to this problem are

$J_x(t) = \sqrt{\gamma/2}\left<x\right> + \xi(t)$

$J_y(t) = \sqrt{\gamma/2}\left<y\right> + \xi(t)$

where $\xi(t) = \frac{dW}{dt}$.

However, we desire the homodyne currents for the $x$ and $y$ quadratures:

$J_x(t) = \sqrt{\gamma}\left<x\right> + \sqrt{2}\xi(t)$

$J_y(t) = \sqrt{\gamma}\left<y\right> + \sqrt{2}\xi(t)$

In qutip we can use the predefined homodyne solver for solving this problem, but rescale the `m_ops` and `dW_factors`.

```python
options = {
    "method": "platen",
    "dt": 0.001,
    "store_measurement": True,
    "map": "parallel",
}
sc_ops = [np.sqrt(gamma / 2) * a, -1.0j * np.sqrt(gamma / 2) * a]

solver = SMESolver(H, sc_ops=sc_ops, heterodyne=False, options=options)
solver.m_ops = [np.sqrt(gamma) * x, np.sqrt(gamma) * y]
solver.dW_factors = [np.sqrt(2), np.sqrt(2)]
result = solver.run(rho0, times, e_ops=e_ops, ntraj=ntraj)
```

```python
plot_expectation_values([result, result_ref])
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times[1:], m[0, :].real, "b", alpha=0.05)
    ax.plot(times[1:], m[1, :].real, "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_xlim(0, times.max())
ax.set_ylim(-25, 25)
ax.set_xlabel("time", fontsize=12)
ax.plot(
    times[1:], np.array(result.measurement).mean(axis=0)[0, :].real, "k", lw=2
)
ax.plot(
    times[1:], np.array(result.measurement).mean(axis=0)[1, :].real, "k", lw=2
)
```

## Common problem

For some systems, the resulting density matrix can become unphysical due to the accumulation of computation error.

```python
options = {
    "method": "euler",
    "dt": 0.1,
    "store_states": True,
    "store_measurement": True,
    "map": "parallel",
}

result = smesolve(
    H,
    rho0,
    np.linspace(0, 2, 21),
    sc_ops=c_ops,
    heterodyne=True,
    e_ops=e_ops,
    ntraj=ntraj,
    options=options,
)

result.expect
```

```python
result.states[-1].full()
```

```python
sp.linalg.eigh(result.states[10].full(), eigvals_only=True)
```

Using smaller integration steps by lowering the ``dt`` option will lower the numerical errors.
The solver algorithm used affect the convergence and numerical error.
Notable solvers are:  
- euler: order 0.5 fastest, but lowest order. Only solver that accept non-commuting sc_ops.
- rouchon: order 1.0?, build to keep the density matrix physical,
- taylor1.5: order 1.5, reasonably fast for good convergence.

To list list all available solver, use ``SMESolver.avail_integrators()``

```python
SMESolver.avail_integrators()
```

## About

```python
about()
```

```python

```
