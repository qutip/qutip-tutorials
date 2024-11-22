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

# QuTiPv5 Paper Example: Stochastic Solver - Homodyne Detection

Authors: Maximilian Meyer-Mölleringhof (m.meyermoelleringhof@gmail.com), Neill Lambert (nwlambert@gmail.com)

## Introduction

When modelling an open quantum system, stochastic noise can be used to simulate a large range of phenomena.
In the `smesolve()` solver, noise is introduced by continuous measurement.
This allows us to generate the trajectory evolution of a quantum system conditioned on a noisy measurement record.
Historically speaking, such models were used by the quantum optics community to model homodyne and heterodyne detection of light emitted from a cavity.
However, this solver is of course quite general and can thus also be applied to other problems.

In this example we look at an optical cavity whose output is subject to homodyne detection.
Such a cavity obeys the general stochastic master equation

$d \rho(t) = -i [H, \rho(t)] dt + \mathcal{D}[a] \rho (t) dt + \mathcal{H}[a] \rho dW(t)$

with the Hamiltonian

$H = \Delta a^\dagger a$

and the Lindblad dissipator

$\mathcal{D}[a] = a \rho a^\dagger - \dfrac{1}{2} a^\dagger a \rho - \dfrac{1}{2} \rho a^\dagger a$.

The stochastic part

$\mathcal{H}[a]\rho = a \rho + \rho a^\dagger - tr[a \rho + \rho a^\dagger]$

captures the conditioning of the trajectory through continious monitoring of the operator $a$.
The term $dW(t)$ is the increment of a Wiener process that obeys $\mathbb{E}[dW] = 0$ and $\mathbb{E}[dW^2] = dt$.

Note that a similiar example is available in the [QuTiP user guide](https://qutip.readthedocs.io/en/qutip-5.0.x/guide/dynamics/dynamics-stochastic.html#stochastic-master-equation).

```python
import numpy as np
from matplotlib import pyplot as plt
from qutip import about, coherent, destroy, mesolve, smesolve

%matplotlib inline
```

## Problem Parameters

```python
N = 20  # dimensions of Hilbert space
delta = 10 * np.pi  # cavity detuning
kappa = 1  # decay rate
A = 4  # initial coherent state intensity
```

```python
a = destroy(N)
x = a + a.dag()  # operator for expectation value
H = delta * a.dag() * a  # Hamiltonian
mon_op = np.sqrt(kappa) * a  # continiously monitored operators
```

## Solving for the Time Evolution

We calculate the predicted trajectory conditioned on the continious monitoring of operator $a$.
This is compared to the regular `mesolve()` solver for the same model but without resolving conditioned trajectories.

```python
rho_0 = coherent(N, np.sqrt(A))  # initial state
times = np.arange(0, 1, 0.0025)
num_traj = 500  # number of computed trajectories
opt = {"dt": 0.00125, "store_measurement": True, "map": "parallel"}
```

```python
me_solution = mesolve(H, rho_0, times, c_ops=[mon_op], e_ops=[x])
```

```python
stoc_solution = smesolve(
    H, rho_0, times, sc_ops=[mon_op], e_ops=[x], ntraj=num_traj, options=opt
)
```

## Comparison of Results

We plot the averaged homodyne current $J_x = \langle x \rangle + dW / dt$ and the average system behaviour $\langle x \rangle$ for 500 trajectories.
This is compared with the prediction of the regular `mesolve()` solver that does not include the conditioned trajectories.
Since the conditioned expectation values do not depend on the trajectories, we expect that this reproduces the result of the standard `me_solve`.

```python
stoc_meas_mean = np.array(stoc_solution.measurement).mean(axis=0)[0, :].real
```

```python
plt.figure()
plt.plot(times[1:], stoc_meas_mean, lw=2, label=r"$J_x$")
plt.plot(times, stoc_solution.expect[0], label=r"$\langle x \rangle$")
plt.plot(
    times,
    me_solution.expect[0],
    "--",
    color="gray",
    label=r"$\langle x \rangle$ mesolve",
)

plt.legend()
plt.xlabel(r"$t \cdot \kappa$")
plt.show()
```

## About

```python
about()
```

## Testing

```python
assert np.allclose(
    stoc_solution.expect[0], me_solution.expect[0], atol=1e-1
), "smesolve and mesolve do not preoduce the same trajectory."
```
