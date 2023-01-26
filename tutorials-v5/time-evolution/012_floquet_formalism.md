---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Floquet Formalism

Author: C. Staufenbiel, 2022

Inspirations taken from the [floquet notebook](https://github.com/qutip/qutip-notebooks/blob/master/examples/floquet-dynamics.ipynb) by P.D. Nation and J.R. Johannson, and the [qutip documentation](https://qutip.org/docs/latest/guide/dynamics/dynamics-floquet.html).

### Introduction

In the [floquet_solver notebook](011_floquet_solver.md) we introduced the two functions to solve the Schrödinger and Master equation using the Floquet formalism (i.e. `fsesolve` and  `fmmesolv`). In this notebook, we will work with on `FloquetBasis` class which is used by solvers. In particular, we will focus on the Floquet modes and quasi-energies.

More information on the implementation of the Floquet Formalism in QuTiP can be found in the [documentation](https://qutip.org/docs/latest/guide/dynamics/dynamics-floquet.html).

### Imports 

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, expect, FloquetBasis,
                   num, plot_wigner, ket, sesolve, sigmax, sigmaz)
```

### System setup
For consistency with the documentation we consider the driven system with the following Hamiltonian: 

$$ H = - \frac{\Delta}{2} \sigma_x - \frac{\epsilon_0}{2} \sigma_z + \frac{A}{2} \sigma_x sin(\omega t) $$

```python
# Constants
delta = 0.2 * 2 * np.pi
eps0 = 1 * 2 * np.pi
A = 2.5 * 2 * np.pi
omega = 1.0 * 2 * np.pi
T = 2 * np.pi / omega

# Hamiltonian
H = [
    -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz(),
    [A / 2 * sigmax(), "sin({w}*t)".format(w=omega)],
]
```

### Floquet modes and quasienergies
For periodic Hamiltonians the solution to the Schrödinger equation can be represented by the Floquet modes $\phi_\alpha(t)$ and the quasienergies $\epsilon_\alpha$. We can obtain these for the initial time $t=0$ by using the class `FloquetBasis(H, T)` and its method `.mode(t=0)`.

For example, we can display the first Floquet mode at $t=0$ using a Wigner distribution:

```python
fbasis = FloquetBasis(H, T)
f_modes_t0 = fbasis.mode(t=0)
plot_wigner(f_modes_t0[0]);
```

For the system defined above there are two eigenenergies. We can plot these two quasienergies for varying strength of driving $A$.

We access the eigenenergies via the `.e_quasi` attribute of `FloquetBasis` while passing `sort=True` to ensure that the energies are sorted from lowest to highest:

```python
A_list = np.linspace(1.0 * omega, 4.5 * omega, 20)
quasienergies1, quasienergies2 = [], []
for A_tmp in A_list:
    # temporary Hamiltonian
    H_tmp = [
        -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz(),
        [A_tmp / 2 * sigmax(), "sin({w}*t)".format(w=omega)],
    ]
    # floquet modes and quasienergies
    e1, e2 = FloquetBasis(H_tmp, T, sort=True).e_quasi
    quasienergies1.append(e1), quasienergies2.append(e2)
```

```python
plt.scatter(A_list / omega, quasienergies1, label="e1")
plt.scatter(A_list / omega, quasienergies2, label="e2")
plt.xlabel("A / w"), plt.ylabel("Quasienergies")
plt.legend();
```

### Time evolution with Floquet mode
To calculate the time evolution of a random initial state $\psi(0)$, we have to decompose the state in the Floquet basis (formed by the Floquet modes):

$$ \psi(0) = \sum_\alpha c_\alpha \phi_\alpha(0) $$

The $c_\alpha$ are calculated using the `.to_floquet_basis` method:

```python
# Define an initial state:
psi0 = ket("0") + ket("1")
psi0 = psi0.unit()

# Decompose the initial state into its components in the Floquet modes:
f_coeff = fbasis.to_floquet_basis(psi0, t=0)
f_coeff
```

The Floquet mode $\phi_\alpha(t)$ for later times $t>0$ can be calculated using the wave function propagator $U(t,0)$ by:

$$ \phi_\alpha(t) = exp(-i\epsilon_\alpha t / \hbar) \, U(t,0) \, \phi_\alpha(0) $$

In QuTiP this is done by the `FloquetBasis.mode(t=t)` function. Here we propagate the initial state to the state at $t=1$:

```python
t = 1.0
f_modes_t1 = fbasis.mode(t=t)
f_modes_t1
```

The propagated Floquet modes $\phi_\alpha(t)$ can be combined to describe the full system state $\psi(t)$ at the time `t`.

The method `.from_floquet_basis(f_coeff, t)` is used to calculate the new state in this manner:

```python
psi_t = fbasis.from_floquet_basis(f_coeff, t)
psi_t
```

### Precomputing and reusing the Floquet modes of one period

The Floquet modes have the same periodicity as the Hamiltonian: 

$$ \phi_\alpha(t + T) = \phi_\alpha(t) $$

Hence it is enough to evaluate the modes at times $t \in [0,T]$. From these modes we can extrapolate the system state $\psi(t)$ for any time $t$. 

The function `FloquetBasis` allows one to calculate the Floquet mode propagators for multiple times in the first period by specifying a list of times to `precompute`:

```python
precompute_tlist = np.linspace(0, T, 50)
fbasis = FloquetBasis(H, T, precompute=precompute_tlist)
```

Again, the function `FloquetBasis.from_floquet_basis(...)` (introduced above) can be used to build the wavefunction $\psi(t)$. Here, we calculate the expectation value for the number operator in the first period:

```python
p_ex_period = []
for t in precompute_tlist:
    psi_t = fbasis.from_floquet_basis(f_coeff, t)
    p_ex_period.append(expect(num(2), psi_t))

plt.plot(precompute_tlist, p_ex_period)
plt.ylabel("Occupation prob."), plt.xlabel("Time");
```

The pre-computed modes for the first period can be used by the `FloquetBasis` class to calculate Floquet modes and states in later periods too. However if a time $t'$ is not exactly $t' = t + nT$ (where $t$ is a time used in the pre-computation) the Floquet modes for time $t'$ will be computed and one of the precomputed modes will be forgotten.

Under the hood, `FloquetBasis` uses `qutip.Propagator` to manage the precomputed modes. The documentation for `Propagator` describes the details. The propagator is directly available as `FloquetBasis.U` if needed.

Below we show how this works in practice over the first ten periods. If the times in `tlist` correspond to those in the first period that we have already been precomputed, calculating the expecations in later periods should be fast:

```python
p_ex = []
tlist = np.linspace(0, 10 * T, 10 * precompute_tlist.shape[0])
for t in tlist:
    psi_t = fbasis.from_floquet_basis(f_coeff, t)
    p_ex.append(expect(num(2), psi_t))

# Plot the occupation Probability
plt.plot(tlist, p_ex, label="Lookup method")
plt.plot(precompute_tlist, p_ex_period, label="First period - precomputed")
plt.legend(loc="upper right")
plt.xlabel("Time"), plt.ylabel("Occupation prob.");
```

### About

```python
about()
```

## Testing

```python
# compute prediction using sesolve
res_sesolve = sesolve(H, psi0, tlist, [num(2)])
assert np.allclose(res_sesolve.expect[0], p_ex, atol=0.15)
```
