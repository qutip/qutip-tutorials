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

# Monte Carlo Solver: Trilinear Oscillator Coupling


Authors: J.R. Johansson and P.D. Nation

Modifications: C. Staufenbiel, 2022

### Introduction

### Imports
First we import all relevant functions.

```python
%matplotlib inline
import matplotlib.pyplot as plt
from qutip import coherent, basis, tensor, destroy, qeye, mcsolve, mesolve
import numpy as np
```

### System setup

```python
# Number of modes for each oscillator
N0, N1, N2 = 8, 4, 4

# Damping Rates
gamma0, gamma1, gamma2 = 0.1, 0.1, 0.5

# Initial State
alpha = np.sqrt(3) # coherent state param for mode 0
psi0 = tensor(coherent(N0, alpha),basis(N1,0),basis(N2,0))


# destroy operators
a0 = tensor(destroy(N0),qeye(N1),qeye(N2))
a1 = tensor(qeye(N0),destroy(N1),qeye(N2))
a2 = tensor(qeye(N0),qeye(N1),destroy(N2))

# number operators
num0=a0.dag()*a0
num1=a1.dag()*a1
num2=a2.dag()*a2
```

## Hamiltonian

```python
#trilinear Hamiltonian
H=1j*(a0*a1.dag()*a2.dag()-a0.dag()*a1*a2)

#dissipative operators for zero-temp. baths
c_ops = []
c_ops.append(np.sqrt(gamma0)*a0)
c_ops.append(np.sqrt(gamma1)*a1)
c_ops.append(np.sqrt(gamma2)*a2)

# Times and number of trajectories for Monte Carlo Solver
tlist = np.linspace(0,5,100)
ntraj = [1,10,500,1000]

#run Monte-Carlo and Master Equation Solver
mc = mcsolve(H, psi0, tlist, c_ops, [num0,num1,num2], ntraj=ntraj)
```

### Plot

```python
fig = plt.figure(figsize=(8, 8), frameon=False)
plt.subplots_adjust(hspace=0.0)

for i in range(len(ntraj)):
    ax = plt.subplot(len(ntraj),1,i+1)
    ax.plot(tlist, mc.expect[i][0], label='Osc. 1')
    ax.plot(tlist, mc.expect[i][1], label='Osc. 2')
    ax.plot(tlist, mc.expect[i][2], label='Osc. 3')
    ax.set_ylabel('<n>')
    ax.legend(loc='upper right')
    ax.set_yticks([0,1,2,3])
    ax.set_ylim([-0.2,3.2])
ax.set_xlabel('Time');
```

### Comparison to Master Equation Solution


```python
# Run Master Equation Solver
me = mesolve(H, psi0, tlist, c_ops, [num0,num1,num2])
```

```python
# Compare Oscillator 1 to solution by Master Equation
fig = plt.figure(figsize=(8, 8), frameon=False)
plt.subplots_adjust(hspace=0.0)

for i in range(len(ntraj)):
    ax = plt.subplot(len(ntraj), 1, i+1)
    ax.plot(tlist, mc.expect[i][0], '--',label='MC #ntraj={}'.format(ntraj[i]))
    ax.plot(tlist, me.expect[0], label='ME')
    ax.set_ylabel('<n>')
    ax.legend(loc='upper right')
    ax.set_yticks([0,1,2,3])
    ax.set_ylim([-0.2,3.2])
ax.set_xlabel('Time');
```

### About

```python
from qutip import about
about()
```

### Testing

```python
assert np.allclose(me.expect[0], mc.expect[-1][0],atol=10**-1)
assert np.allclose(me.expect[1], mc.expect[-1][1],atol=10**-1)
assert np.allclose(me.expect[2], mc.expect[-1][2],atol=10**-1)
```