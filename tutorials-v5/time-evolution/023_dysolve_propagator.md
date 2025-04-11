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

# Calculate time propagators with Dysolve

Author: Mathis Beaudoin, 2025

### Introduction

This notebook shows how to compute time propagators with Dysolve using QuTiP. Dysolve is a method to compute time propagators for hamiltonians of the form $H(t) = H_0 + \cos(\omega t)X$ where $H_0$ is some base hamiltonian and $X$ a perturbation. It performs better than other general methods for this class of hamiltonians and, later on, should support more complicated oscillating perturbations. It is still in development with more features to come in the future. For more details on Dysolve, see the corresponding guide in the documentation.

For the moment, Dysolve can be used with the class `DysolvePropagator` and the function `dysolve_propagator` from QuTiP's solvers. They follow a similar structure to the class `Propagator` and the function `propagator`, another solver that also computes time propagators. 

Let's start by importing the necessary packages.

```python
from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, about
```

### One qubit example using `DysolvePropagator`

We have to define what $H_0$, $X$ and $\omega$ will be. Let's say $H(t) = \sigma_z + \cos(10t)\sigma_x$.

```python
H_0 = sigmaz()
X = sigmax()
omega = 10.0
```

Some options can be defined. `max_order` will be the order of approximation used when calculating a propagator. The higher this integer is, the more precise the results will be (at a cost of taking more time to calculate). `a_tol` is simply the absolute tolerance used in the calculations. Finally, a time propagator can be computed using subpropagators of time increment `max_dt`. Let's say `max_dt` is set to 0.25, then the propagator $U(1, 0)$ will come from the multiplication of supropagators $U(0.25, 0)$, $U(0.5, 0.25)$, $U(0.75, 0.5)$ and $U(1, 0.75)$. This allows for more precise results when the evolution is over a long period of time. In our case, let's keep `a_tol` and `max_dt` to their default value, but let's change `max_order`.

```python
options = {'max_order': 5}
```

Everything is now defined to initialize an instance.

```python
dy = DysolvePropagator(H_0, X, omega, options=options)
```

Then, to compute a time propagator, simply call the instance with a given initial time and final time. Also, only a final time can be given and, in that case, the initial time is considered to be 0.

```python
t_i = -1
t_f = 1
U = dy(t_f, t_i)
```

This returns a single time propagator $U(t_f, t_i)$.

### Two qubits example with `dysolve_propagator`

We proceed like the previous example.

```python
# Define the system
H_0 = tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay())
X = tensor(qeye(2), sigmaz())
omega = 5.0

# Keep options to default
```

`dysolve_propagator` can take more than one time value. If a single time is passed, a single propagator $U(t,0)$ is returned. If a list of times is given, the function will return a list of propagator $[U(\text{times}[i], \text{times}[0])]$ for all $i$. 

```python
times = [-0.1, 0, 0.1]
Us = dysolve_propagator(H_0, X, omega, times)
```

### About

```python
about()
```