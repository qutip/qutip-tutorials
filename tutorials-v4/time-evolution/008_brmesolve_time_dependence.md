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

# Bloch-Redfield Solver: Time dependent operators

Authors: C. Staufenbiel, 2022

following the instructions in the [Bloch-Redfield documentation](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html?#time-dependent-bloch-redfield-dynamics).

### Introduction
This notebook introduces the usage of time-dependent operators in the Bloch-Redfield solver, which is also described in the [corresponding documentation](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html?#time-dependent-bloch-redfield-dynamics).

We will discuss time-dependent Hamiltonians and time-dependent dissipations. The Bloch-Redfield solver is especially efficient since it uses Cython internally. For correct functioning we have to pass the time dependence in a string-based format. 

### Imports

```python
import numpy as np
from qutip import about, basis, brmesolve, destroy, plot_expectation_values


```

For our small example, we setup a system with `N` states and the number operator as Hamiltonian. We can observe that for the constant Hamiltonian and no given `a_ops` the expectation value $\langle n \rangle $ is a constant.

```python
# num modes
N = 2
# Hamiltonian
a = destroy(N)
H = a.dag() * a

# initial state
psi0 = basis(N, N - 1)

# times for simulation
times = np.linspace(0, 10, 100)

# solve using brmesolve
result_const = brmesolve(H, psi0, times, e_ops=[a.dag() * a])
```

```python
plot_expectation_values(result_const, ylabels=["<n>"]);
```

Next we define a string, which describes some time-dependence. We can use functions that are supported by the Cython implementation. A list of all supported functions can be found in the  [docs](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-time.html#time). For example, supported functions are `sin` or `exp`. The time variable is denoted by `t`.

```python
time_dependence = "sin(t)"
```

### Time-dependent Hamiltonian

As a first example, we define a time-dependent Hamiltonian (as described [here](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-time.html)). 

$$ H = \hat{n} + sin(t) \hat{x} $$

Again, we can solve the dynamics using `brmesolve()`.

```python
H_t = [H, [a + a.dag(), time_dependence]]
result_brme = brmesolve(H_t, psi0, times, e_ops=[a.dag() * a])
plot_expectation_values(result_brme, ylabels=["<n>"]);
```

### Time-dependent dissipation

Above we did not use the noise-power-spectrum, which the Bloch-Redfield solver is mainly used for. This spectrum is passed in the argument `a_ops`. We can also add a string-based time dependence to `a_ops` and thereby make the dissipation itself time-dependent. 

Here we will define a a noice power spectrum of the form:

$$ J(\omega, t) = \kappa * e^{-t} \quad \text{for} \; \omega \geq 0$$

```python
# setup dissipation
kappa = 0.2
a_ops = [[a + a.dag(), "{kappa}*exp(-t)*(w>=0)".format(kappa=kappa)]]

# solve
result_brme_aops = brmesolve(H, psi0, times, a_ops, e_ops=[a.dag() * a])

plot_expectation_values([result_brme_aops], ylabels=["<n>"]);
```

The coupling to the bath is sometimes described by operators of the form

$$ A = f(t)a + f(t)^* a^\dagger $$

To add such a coupling to `brmesolve` we can pass tuple in the `a_ops` argument. For example if we have $f(t) = e^{i * t}$ we can define the coupling of operator $A$ with strength $\kappa$ by the following `a_ops`. Note that t

```python
a_ops = [[(a, a.dag()),
          ('{kappa} * (w>=0)'.format(kappa=kappa),
           'exp(1j*t)', 'exp(-1j*t)')]]

# solve using brmesolve and plot expecation
result_brme_aops_sum = brmesolve(H, psi0, times, a_ops, e_ops=[a.dag() * a])
plot_expectation_values([result_brme_aops_sum], ylabels=["<n>"]);
```

### About

```python
about()
```

### Testing

```python
assert np.allclose(result_const.expect[0], 1.0)

# compare result from brme with a_ops to analytic solution
analytic_aops = (N - 1) * np.exp(-kappa * (1.0 - np.exp(-times)))
assert np.allclose(result_brme_aops.expect[0], analytic_aops)

assert np.all(np.diff(result_brme_aops_sum.expect[0]) <= 0.0)
```
