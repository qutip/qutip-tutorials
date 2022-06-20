---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# QuTiP lecture: Superconducting Josephson charge qubits

J.R. Johansson (robert@riken.jp)

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

```python
from qutip import *
```

### Introduction

The Hamiltonian for a Josephson charge qubit is

$\displaystyle H = \sum_n 4 E_C (n_g - n)^2 \left|n\right\rangle\left\langle n\right| - \frac{1}{2}E_J\sum_n\left(\left|n+1\right\rangle\left\langle n\right| + \left|n\right\rangle\left\langle n+1\right| \right)$

where $E_C$ is the charge energy, $E_J$ is the Josephson energy, and $\left| n\right\rangle$ is the charge state with $n$ Cooper-pairs on the island that makes up the charge qubit.


#### References

 * [J. Koch et al, Phys. Rec. A 76, 042319 (2007)](http://link.aps.org/doi/10.1103/PhysRevA.76.042319)
 * [Y.A. Pashkin et al, Quantum Inf Process 8, 55 (2009)](http://dx.doi.org/10.1007/s11128-009-0101-5)


### Helper functions

Below we will repeatedly need to obtain the charge qubit Hamiltonian for different parameters, and to plot the eigenenergies, so here we define two functions to do these tasks. 

```python
def hamiltonian(Ec, Ej, N, ng):
    """
    Return the charge qubit hamiltonian as a Qobj instance.
    """
    m = np.diag(4 * Ec * (arange(-N,N+1)-ng)**2) + 0.5 * Ej * (np.diag(-np.ones(2*N), 1) + 
                                                               np.diag(-np.ones(2*N), -1))
    return Qobj(m)
```

```python
def plot_energies(ng_vec, energies, ymax=(20, 3)):
    """
    Plot energy levels as a function of bias parameter ng_vec.
    """
    fig, axes = plt.subplots(1,2, figsize=(16,6))

    for n in range(len(energies[0,:])):
        axes[0].plot(ng_vec, energies[:,n])
    axes[0].set_ylim(-2, ymax[0])
    axes[0].set_xlabel(r'$n_g$', fontsize=18)
    axes[0].set_ylabel(r'$E_n$', fontsize=18)

    for n in range(len(energies[0,:])):
        axes[1].plot(ng_vec, (energies[:,n]-energies[:,0])/(energies[:,1]-energies[:,0]))
    axes[1].set_ylim(-0.1, ymax[1])
    axes[1].set_xlabel(r'$n_g$', fontsize=18)
    axes[1].set_ylabel(r'$(E_n-E_0)/(E_1-E_0)$', fontsize=18)
    return fig, axes
```

```python
def visualize_dynamics(result, ylabel):
    """
    Plot the evolution of the expectation values stored in result.
    """
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(result.times, result.expect[0])

    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(r'$t$', fontsize=16);
```

### Charge qubit regime

```python
N = 10
Ec = 1.0
Ej = 1.0
```

```python
ng_vec = np.linspace(-4, 4, 200)

energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies);
```

```python
ng_vec = np.linspace(-1, 1, 200)

energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(7.5, 3.0));
```

### Intermediate regime

```python
ng_vec = np.linspace(-4, 4, 200)
```

```python
Ec = 1.0
Ej = 5.0
```

```python
energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(50, 3));
```

```python
Ec = 1.0
Ej = 10.0
```

```python
energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(50, 3));
```

### Transmon regime

```python
Ec = 1.0
Ej = 50.0
```

```python
energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(50, 3));
```

Note that the energy-level splitting is essentially independent of the gate bias $n_g$, at least for the lowest few states. This device insensitive to charge noise. But at the same time the two lowest energy states are no longer well separated from higher states (it has become more like an harmonic oscillator). But some anharmonicity still remains, and it can still be used as a qubit if the leakage of occupation probability of the higher states can be kept under control.


## Focus on the two lowest energy states

Let's go back to the charge regime, and look at the lowest few energy levels again:

```python
N = 10
Ec = 1.0
Ej = 1.0
```

```python
ng_vec = np.linspace(-1, 1, 200)
```

```python
energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(10, 3));
```

We can see that around $n_g = 0.5$ we have two lowest energy levels that are well separated for the higher energy levels:

```python
ng_vec = np.linspace(0.25, 0.75, 200)
energies = array([hamiltonian(Ec, Ej, N, ng).eigenenergies() for ng in ng_vec])
plot_energies(ng_vec, energies, ymax=(10, 1.1));
```

Let's tune the system to $n_g = 0.5$ and look at the Hamiltonian and its eigenstates in detail

```python
H = hamiltonian(Ec, Ej, N, 0.5)
```

```python
H
```

```python
evals, ekets = H.eigenstates()
```

The eigenenergies are sorted:

```python
evals
```

Only two states have a significant weight in the two lowest eigenstates:

```python
ekets[0].full() > 0.1
```

```python
abs(ekets[1].full()) > 0.1
```

We can use these two isolated eigenstates to define a qubit basis:

```python
psi_g = ekets[0] # basis(2, 0)
psi_e = ekets[1] # basis(2, 1)

#psi_g = basis(2, 0)
#psi_e = basis(2, 1)
```

and corresponding Pauli matrices:

```python
sx = psi_g * psi_e.dag() + psi_e * psi_g.dag()
```

```python
sz = psi_g * psi_g.dag() - psi_e * psi_e.dag()
```

and an effective qubit Hamiltonian

```python
evals[1]-evals[0]
```

```python
H0 = 0.5 * (evals[1]-evals[0]) * sz

A = 0.25  # some driving amplitude
Hd = 0.5 * A * sx # obtained by driving ng(t), 
                  #but now H0 is in the eigenbasis so the drive becomes a sigma_x
```

Doing this we have a bunch of extra energy levels in the system that aren't involved in the dynamics, but so far they are still in the Hamiltonian.

```python
qubit_evals = H0.eigenenergies()

qubit_evals - qubit_evals[0]
```

```python
energy_level_diagram([H0, Hd], figsize=(4,2));
```

Imagine that we also can drive a $\sigma_x$ type of interaction (e.g., external field):

```python
Heff = [H0, [Hd, 'sin(wd*t)']]

args = {'wd': (evals[1]-evals[0])}
```

Let's look at the Rabi oscillation dynamics of the qubit when initially placed in the ground state:

```python
psi0 = psi_g
```

```python
tlist = np.linspace(0.0, 100.0, 500)
result = mesolve(Heff, psi0, tlist, [], [ket2dm(psi_e)], args=args)
```

```python
visualize_dynamics(result, r'$\rho_{ee}$');
```

We can see that only the two selected states are included in the dynamics, and very little leakage to other levels occur.

Instead of keeping all the inactive quantum states in the calculation we can eliminate them using Qobj.extract_state, so that we obtain a true two-level system.

```python
where(abs(ekets[0].full().flatten()) > 0.1)[0]
```

```python
where(abs(ekets[1].full().flatten()) > 0.1)[0]
```

```python
keep_states = where(abs(ekets[1].full().flatten()) > 0.1)[0]
```

```python
H0 = H0.extract_states(keep_states)

H0
```

```python
Hd = Hd.extract_states(keep_states)

Hd
```

And if we look at the energy level diagram now we see that we only have two states in the system, as desired.

```python
energy_level_diagram([H0, Hd], figsize=(4,2));
```

```python
Heff = [H0, [Hd, 'sin(wd*t)']]

args = {'wd': (evals[1]-evals[0])}
```

```python
psi0 = psi0.extract_states(keep_states)
```

```python
psi_e = psi_e.extract_states(keep_states)
```

```python
tlist = np.linspace(0.0, 100.0, 500)
result = mesolve(Heff, psi0, tlist, [], [ket2dm(psi_e)], args=args)
```

```python
visualize_dynamics(result, r'$\rho_{ee}$');
```

### Software versions

```python
from qutip.ipynbtools import version_table; version_table()
```
