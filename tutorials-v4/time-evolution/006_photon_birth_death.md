---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Monte Carlo Solver: Birth and Death of Photons in a Cavity

<!-- #region -->
Authors: J.R. Johansson and P.D. Nation

Modifications: C. Staufenbiel (2022)

### Introduction

In this tutorial we demonstrate the *Monte Carlo Solver* functionality implemented in `qutip.mcsolve()`. For more information on the *MC Solver* refer to the [QuTiP documentation](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-monte.html). 

We aim to reproduce the experimental results from:



>  Gleyzes et al., "Quantum jumps of light recording the birth and death of a photon in a cavity", [Nature **446**,297 (2007)](http://dx.doi.org/10.1038/nature05589).


In particular, we will simulate the creation and annihilation of photons inside an optical cavity due to the thermal environment when the initial cavity is a single-photon Fock state $ |1\rangle$, as presented in Fig. 3 from the article.

## Imports
First we import the relevant functionalities:
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import about, basis, destroy, mcsolve, mesolve

%matplotlib inline
```

## System Setup
In this example, we consider a simple oscillator Hamiltonian $H = a^\dagger a$ and one initial photon in the cavity.

```python
# number of modes in the cavity
N = 5
# Destroy operator
a = destroy(N)
# oscillator Hamiltonian
H = a.dag() * a
# Initial Fock state with one photon
psi0 = basis(N, 1)
```

The coupling to the external heat bath is described by a coupling constant $\kappa$ and the temperature of the heat bath is defined via the average photon number $\langle n \rangle$. In QuTiP the interaction between the system and heat bath is defined via the collapse operators. For this example, there are two collapse operators. One for photon annihilation ($C_1$) and one for photon creation ($C_2$): 

$C_1 = \sqrt{\kappa (1 + \langle n \rangle)} \; a$

$C_2 = \sqrt{\kappa \langle n \rangle} \; a^\dagger$

We give some numerical values to the coupling constant $\kappa$ and the average photon number of the heat bath $\langle n \rangle$.

```python
kappa = 1.0 / 0.129  # Coupling rate to heat bath
nth = 0.063  # Temperature with <n>=0.063

# collapse operators for the thermal bath
c_ops = []
c_ops.append(np.sqrt(kappa * (1 + nth)) * a)
c_ops.append(np.sqrt(kappa * nth) * a.dag())
```

## Monte Carlo Simulation
The *Monte Carlo Solver* allows simulating an individual realization of the system dynamics. This is in contrast to e.g. the *Master Equation Solver*, which solves for the ensemble average over many identical realizations of the system. `qutip.mcsolve()` also offers to average over many runs of identical system setups by passing the *number of trajectories* `ntraj` to the function. If we choose `ntraj = 1` the system is only simulated once and we see it's dynamics. If we choose a large value for `ntraj`, the predictions will be averaged and therefore converge to the solution from `qutip.mesolve()`. 

We can also pass a list to `ntraj`. `qutip.mcsolve()` will calculate the results for the specified number of trajectories. Note that the entries need to be in ascending order, as the previous results are reused.

Here we are interested in the time evolution of $a^\dagger a$ for different numbers of `ntraj`. We will compare the results to the predictions by `qutip.mesolve().

```python
ntraj = [1, 5, 15, 904]  # number of MC trajectories
tlist = np.linspace(0, 0.8, 100)

# Solve using MCSolve for different ntraj
mc = mcsolve(H, psi0, tlist, c_ops, [a.dag() * a], ntraj)
me = mesolve(H, psi0, tlist, c_ops, [a.dag() * a])
```

## Reproduce plot from article
Using the above results we can reproduce Fig. 3 from the article mentioned above. The individual figures plot the time evolution of $\langle a^\dagger a \rangle$ for the system we set up above. The effect of using different `ntraj` for the simulation using `mcsolve` is shown. When choosing `ntraj = 1` we see the dynamics of one particular quantum system. If `ntraj > 1` the output is averaged over the number of realizations. 

```python
fig = plt.figure(figsize=(8, 8), frameon=False)
plt.subplots_adjust(hspace=0.0)

for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    ax.plot(
        tlist, mc.expect[i][0], "b", lw=2,
        label="#trajectories={}".format(ntraj[i])
    )
    ax.plot(tlist, me.expect[0], "r--", lw=2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel(r"$\langle P_{1}(t)\rangle$")
    ax.legend()

ax.set_xlabel(r"Time (s)");
```

## About

```python
about()
```

## Testing

```python
np.testing.assert_allclose(me.expect[0], mc.expect[3][0], atol=10**-1)
assert np.all(np.diff(me.expect[0]) <= 0)
```
