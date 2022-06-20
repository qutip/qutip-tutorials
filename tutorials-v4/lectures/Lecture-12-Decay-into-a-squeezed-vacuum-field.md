---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

# Lecture 12: Decay into a squeezed vacuum field


Author: J. R. Johansson (robert@riken.jp), http://dml.riken.jp/~rob/

The latest version of this [IPython notebook](http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html) lecture is available at [http://github.com/jrjohansson/qutip-lectures](http://github.com/jrjohansson/qutip-lectures).

The other notebooks in this lecture series are indexed at [http://jrjohansson.github.com](http://jrjohansson.github.com).

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

```python
from qutip import *
```

## Introduction

We follow *The theory of open quantum systems*, by Breuer and Pretruccione, section 3.4.3 - 3.4.4, which gives the master equation for a two-level system that decays into an environment that is in a squeezed vacuum state:

$\frac{d}{dt}\rho = \gamma_0(N+1)\left(\sigma_-\rho(t)\sigma_+ - \frac{1}{2}\sigma_+\sigma_-\rho(t) - \frac{1}{2}\rho(t)\sigma_+\sigma_-\right)$

$ + \gamma_0 N \left(\sigma_+\rho(t)\sigma_- - \frac{1}{2}\sigma_-\sigma_+\rho(t) - \frac{1}{2}\rho(t)\sigma_-\sigma_+\right)$

$ -\gamma_0 M \sigma_+\rho(t)\sigma_+ -\gamma_0 M^* \sigma_-\rho(t)\sigma_-$

where the parameters $N$ and $M$ describes the temperature and squeezing of the environmental modes:

$\displaystyle N = N_{\rm th} ({\cosh}^2 r + {\sinh}^2 r) + \sinh^2 r$

$\displaystyle M = - \cosh r \sinh r e^{i\theta} (2 N_{\rm th} + 1)$

Alternatively, this master equation can be written in standard Lindblad form,

$\frac{d}{dt}\rho = \gamma_0\left(C\rho(t)C^\dagger - \frac{1}{2}C^\dagger C\rho(t) - \frac{1}{2}\rho(t)C^\dagger C\right)$

where $C = \sigma_-\cosh r + \sigma_+ \sinh r e^{i\theta}$.

Below we will solve these master equations numerically using QuTiP, and visualize at the resulting dynamics.



### Problem parameters

```python
w0 = 1.0 * 2 * pi
gamma0 = 0.05
```

```python
# the temperature of the environment in frequency units
w_th = 0.0 * 2 * pi
```

```python
# the number of average excitations in the environment mode w0 at temperture w_th
Nth = n_thermal(w0, w_th)

Nth
```

#### Parameters that describes the squeezing of the bath

```python
# squeezing parameter for the environment
r = 1.0
theta = 0.1 * pi
```

```python
N = Nth * (cosh(r) ** 2 + sinh(r) ** 2) + sinh(r) ** 2

N
```

```python
M = - cosh(r) * sinh(r) * exp(-1j * theta) * (2 * Nth + 1)

M
```

```python
# Check, should be zero according to Eq. 3.261 in Breuer and Petruccione
abs(M)**2 - (N * (N + 1) - Nth * (Nth + 1))
```

### Operators, Hamiltonian and initial state

```python
sm = sigmam()
sp = sigmap()
```

```python
H = - 0.5 * w0 * sigmaz()  # by adding the hamiltonian here, so we move back to the schrodinger picture
```

```python
c_ops = [sqrt(gamma0 * (N + 1)) * sm, sqrt(gamma0 * N) * sp]
```

Let's first construct the standard part of the Liouvillian, corresponding the unitary contribution and the first two terms in the first master equation given above:

```python
L0 = liouvillian(H, c_ops)

L0
```

Next we manually construct the Liouvillian for the effect of the squeeing in the environment, which is not on standard form we can therefore not use the `liouvillian` function in QuTiP

```python
Lsq = - gamma0 * M * spre(sp) * spost(sp) - gamma0 * conjugate(M) * spre(sm) * spost(sm)

Lsq
```

The total Liouvillian for the master equation is now

```python
L = L0 + Lsq

L
```

### Evolution

We can now solve the master equation numerically using QuTiP's `mesolve` function:

```python
tlist = np.linspace(0, 50, 1000)
```

```python
# start in the qubit superposition state
psi0 = (2j * basis(2, 0) + 1 * basis(2, 1)).unit()
```

```python
e_ops = [sigmax(), sigmay(), sigmaz()]
```

```python
result1 = mesolve(L, psi0, tlist, [], e_ops)
```

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(result1.times, result1.expect[0], 'r', label=r'$\langle\sigma_x\rangle$')
ax.plot(result1.times, result1.expect[1], 'g', label=r'$\langle\sigma_y\rangle$')
ax.plot(result1.times, result1.expect[2], 'b', label=r'$\langle\sigma_z\rangle$')

sz_ss_analytical = - 1 / (2 * N + 1)
ax.plot(result1.times, sz_ss_analytical * np.ones(shape(result1.times)), 'k--', 
        label=r'$\langle\sigma_z\rangle_s$ analytical')


ax.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=16)
ax.set_xlabel("time", fontsize=16)
ax.legend()
ax.set_ylim(-1, 1);
```

```python
b = Bloch()
b.add_points(result1.expect, meth='l')
b.show()
```

### Alternative master equation on Lindblad form

We can solve the alternative master equation, which is on the standard Lindblad form, directly using the QuTiP `mesolve` function:

```python
c_ops = [sqrt(gamma0) * (sm * cosh(r) + sp * sinh(r) * exp(1j*theta))]
```

```python
result2 = mesolve(H, psi0, tlist, c_ops, e_ops)
```

And we can verify that it indeed gives the same results:

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(result2.times, result2.expect[0], 'r', label=r'$\langle\sigma_x\rangle$')
ax.plot(result2.times, result2.expect[1], 'g', label=r'$\langle\sigma_y\rangle$')
ax.plot(result2.times, result2.expect[2], 'b', label=r'$\langle\sigma_z\rangle$')

sz_ss_analytical = - 1 / (2 * N + 1)
ax.plot(result2.times, sz_ss_analytical * np.ones(shape(result2.times)), 'k--', 
        label=r'$\langle\sigma_z\rangle_s$ analytical')


ax.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=16)
ax.set_xlabel("time", fontsize=16)
ax.legend()
ax.set_ylim(-1, 1);
```

### Compare the two forms of master equations

```python
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))

axes[0].plot(result1.times, result1.expect[0], 'r', label=r'$\langle\sigma_x\rangle$ - me')
axes[0].plot(result2.times, result2.expect[0], 'b--', label=r'$\langle\sigma_x\rangle$ - me lindblad')
axes[0].legend()
axes[0].set_ylim(-1, 1);

axes[1].plot(result1.times, result1.expect[1], 'r', label=r'$\langle\sigma_y\rangle$ - me')
axes[1].plot(result2.times, result2.expect[1], 'b--', label=r'$\langle\sigma_y\rangle$ - me lindblad')
axes[1].legend()
axes[1].set_ylim(-1, 1);

axes[2].plot(result1.times, result1.expect[2], 'r', label=r'$\langle\sigma_y\rangle$ - me')
axes[2].plot(result2.times, result2.expect[2], 'b--', label=r'$\langle\sigma_y\rangle$ - me lindblad')
axes[2].legend()
axes[2].set_ylim(-1, 1);
axes[2].set_xlabel("time", fontsize=16);
```

### Compare dissipation into vacuum and squeezed vacuum

```python
# for vacuum: 
r = 0
theta = 0.0
c_ops = [sqrt(gamma0) * (sm * cosh(r) + sp * sinh(r) * exp(1j*theta))]
```

```python
result1 = mesolve(H, psi0, tlist, c_ops, e_ops)
```

```python
# for squeezed vacuum: 
r = 1.0
theta = 0.0
c_ops = [sqrt(gamma0) * (sm * cosh(r) + sp * sinh(r) * exp(1j*theta))]
```

```python
result2 = mesolve(H, psi0, tlist, c_ops, e_ops)
```

```python
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))

axes[0].plot(result1.times, result1.expect[0], 'r', label=r'$\langle\sigma_x\rangle$ - vacuum')
axes[0].plot(result2.times, result2.expect[0], 'b', label=r'$\langle\sigma_x\rangle$ - squeezed vacuum')
axes[0].legend()
axes[0].set_ylim(-1, 1);

axes[1].plot(result1.times, result1.expect[1], 'r', label=r'$\langle\sigma_y\rangle$ - vacuum')
axes[1].plot(result2.times, result2.expect[1], 'b', label=r'$\langle\sigma_y\rangle$ - squeezed vacuum')
axes[1].legend()
axes[1].set_ylim(-1, 1);

axes[2].plot(result1.times, result1.expect[2], 'r', label=r'$\langle\sigma_y\rangle$ - vacuum')
axes[2].plot(result2.times, result2.expect[2], 'b', label=r'$\langle\sigma_y\rangle$ - squeezed vacuum')
axes[2].legend()
axes[2].set_ylim(-1, 1);
axes[2].set_xlabel("time", fontsize=16);
```

From this comparison it's clear that dissipation into a squeezed vacuum is faster than dissipation into vacuum.


### Software versions

```python
from qutip.ipynbtools import version_table; version_table()
```
