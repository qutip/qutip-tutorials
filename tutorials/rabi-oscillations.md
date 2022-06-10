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

# QuTiP example: Vacuum Rabi oscillations in the Jaynes-Cummings model

Authors: J.R. Johansson and P.D. Nation

Slight modifications: C. Staufenbiel (2022)

This notebook demonstrates how to simulate the quantum vacuum rabi oscillations in the Jaynes-Cumming model, using QuTiP. We also consider the dissipative version of the Jaynes-Cumming model, i.e., the cavity and the atom are subject to dissipation.


### Package import

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
```

# Introduction

The Jaynes-Cumming model is the simplest possible model of quantum mechanical light-matter interaction, describing a single two-level atom interacting with a single electromagnetic cavity mode. The Hamiltonian for this system is (in dipole interaction form)

$H = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger + a)(\sigma_- + \sigma_+)$

or with the rotating-wave approximation

$H_{\rm RWA} = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger\sigma_- + a\sigma_+)$

where $\omega_c$ and $\omega_a$ are the frequencies of the cavity and atom, respectively, and $g$ is the interaction strength.

**TODO : ADD EXPLANATION OF DISSIPATION RATES AND GENERATION FOR POSITIVE TEMPERATURE**

### Problem parameters


Here we use units where $\hbar = 1$: 

**TODO: Add some more description on the factors (already above)**

```python
wc = 1.0  * 2 * np.pi  # cavity frequency
wa = 1.0  * 2 * np.pi  # atom frequency
g  = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005          # cavity dissipation rate
gamma = 0.05           # atom dissipation rate
N = 15                 # number of cavity fock states
n_th_a = 0.0           # temperature in frequency units
use_rwa = True

tlist = np.linspace(0,25,100)
```

### Setup the operators, the Hamiltonian and initial state

Here we define the initial state and operators for the combined system, which consists of the cavity and the atom. We make use of the tensor product, where the first part refers to the cavity and the second part to the atom.

The initial state  consists of the cavity ground state and the atom in the excited state. We define the collapse operator for the cavity/atom in the combined system and the Hamiltonian with and without the rotating-wave-approach.

```python
# intial state
psi0 = tensor(basis(N,0), basis(2,1))  

# collapse operators
a  = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))

# Hamiltonian
if use_rwa:
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
else:
    H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * (sm + sm.dag())
```

### Create a list of collapse operators that describe the dissipation

We create a list of collapse operators `c_ops`, which is later passed on to `qutip.mesolve`. For each of the three processes one collapse operator is defined.

```python
c_op_list = []

# Cavity annihilation
rate = kappa * (1 + n_th_a)
c_op_list.append(np.sqrt(rate) * a)

# Cavity creation 
rate = kappa * n_th_a
c_op_list.append(np.sqrt(rate) * a.dag())

# Atom annihilation
rate = gamma
c_op_list.append(np.sqrt(rate) * sm)
```

### Evolve the system

Here we evolve the system with the Lindblad master equation solver `qutip.mesolve`, and we request that the expectation values of the operators $a^\dagger a$ and $\sigma_+\sigma_-$ are returned by the solver by passing the list `[a.dag()*a, sm.dag()*sm]` as the fifth argument to the solver.

```python
output = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])
```

## Visualize the results

Here we plot the excitation probabilities of the cavity and the atom (these expectation values were calculated by the `mesolve` above). We can clearly see how energy is being coherently transferred back and forth between the cavity and the atom.

```python
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(tlist, output.expect[0], label="Cavity")
ax.plot(tlist, output.expect[1], label="Atom excited state")
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Occupation probability')
ax.set_title('Vacuum Rabi oscillations');
```

### Software version:

```python
from qutip import about
about()
```

### Testing

```python
assert np.allclose(output.expect[0][0], 0.0), output.expect[0][0]
assert np.allclose(output.expect[1][0], 1.0), output.expect[1][0]
```

```python

```
