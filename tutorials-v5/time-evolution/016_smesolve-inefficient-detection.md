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

# Stochastic Solver: Mixing stochastic and deterministic equations


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

```python
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, coherent, destroy, fock, liouvillian,
                   mesolve, mcsolve, plot_expectation_values, smesolve)

%matplotlib inline

rcParams["font.family"] = "STIXGeneral"
rcParams["mathtext.fontset"] = "stix"
rcParams["font.size"] = "14"
```

## Direct photo-detection


Here we follow an example from Wiseman and Milburn, *Quantum measurement and control*, section. 4.8.1.

Consider cavity that leaks photons with a rate $\kappa$. The dissipated photons are detected with an inefficient photon detector,
with photon-detection efficiency $\eta$. The master equation describing this scenario, where a separate dissipation channel has been added for detections and missed detections, is

$\dot\rho = -i[H, \rho] + \mathcal{D}[\sqrt{1-\eta} \sqrt{\kappa} a] + \mathcal{D}[\sqrt{\eta} \sqrt{\kappa}a]$

To describe the photon measurement stochastically, we can unravelling only the dissipation term that corresponds to detections, and leaving the missed detections as a deterministic dissipation term, we obtain [Eq. (4.235) in W&M]

$d\rho = \mathcal{H}[-iH -\eta\frac{1}{2}a^\dagger a] \rho dt + \mathcal{D}[\sqrt{1-\eta} a] \rho dt + \mathcal{G}[\sqrt{\eta}a] \rho dN(t)$

or

$d\rho = -i[H, \rho] dt + \mathcal{D}[\sqrt{1-\eta} a] \rho dt -\mathcal{H}[\eta\frac{1}{2}a^\dagger a] \rho dt + \mathcal{G}[\sqrt{\eta}a] \rho dN(t)$

where

$\displaystyle \mathcal{G}[A] \rho = \frac{A\rho A^\dagger}{\mathrm{Tr}[A\rho A^\dagger]} - \rho$

$\displaystyle \mathcal{H}[A] \rho = A\rho + \rho A^\dagger - \mathrm{Tr}[A\rho + \rho A^\dagger] \rho $

and $dN(t)$ is a Poisson distributed increment with $E[dN(t)] = \eta \langle a^\dagger a\rangle (t)$.


### Formulation in QuTiP

The photocurrent stochastic master equation is written in the form:

$\displaystyle d\rho(t) = -i[H, \rho] dt + \mathcal{D}[B] \rho dt 
- \frac{1}{2}\mathcal{H}[A^\dagger A] \rho(t) dt 
+ \mathcal{G}[A]\rho(t) d\xi$

where the first two term gives the deterministic master equation (Lindblad form with collapse operator $B$ (c_ops)) and $A$ the stochastic collapse operator (sc_ops). 

Here $A = \sqrt{\eta\gamma} a$ and $B = \sqrt{(1-\eta)\gamma} $a.

In QuTiP, the monte carlo solver can solve this equation when the deterministic part is passed as a liouvillian.

```python
N = 15
w0 = 0.5 * 2 * np.pi
times = np.linspace(0, 15, 150)
dt = times[1] - times[0]
gamma = 0.1

a = destroy(N)

H = w0 * a.dag() * a

rho0 = fock(N, 5)

e_ops = [a.dag() * a, a + a.dag()]
```

### Highly efficient detection

```python
eta = 0.7
c_ops = [np.sqrt(1 - eta) * np.sqrt(gamma) * a]  # collapse operator B
sc_ops = [np.sqrt(eta) * np.sqrt(gamma) * a]  # stochastic collapse operator A
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
result1 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=1,
)
```

```python
result2 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=10,
)
```

```python
np.array(result2.runs_photocurrent).dtype
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(times, result1.expect[0],
                label=r"Stochastic ME (ntraj = 1)", lw=2)
axes[0, 0].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 0].set_title("Cavity photon number (ntraj = 1)")
axes[0, 0].legend()

axes[0, 1].plot(times, result2.expect[0],
                label=r"Stochatic ME (ntraj = 10)", lw=2)
axes[0, 1].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 1].set_title("Cavity photon number (ntraj = 10)")
axes[0, 1].legend()

axes[1, 0].step(times[1:], dt * np.cumsum(result1.photocurrent), lw=2)
axes[1, 0].set_title("Cummulative photon detections (ntraj = 1)")
axes[1, 1].step(times[1:], dt * np.cumsum(result2.photocurrent), lw=2)
axes[1, 1].set_title("Cummulative avg. photon detections (ntraj = 10)")

fig.tight_layout()
```

### Highly inefficient photon detection

```python
eta = 0.1
c_ops = [np.sqrt(1 - eta) * np.sqrt(gamma) * a]  # collapse operator B
sc_ops = [np.sqrt(eta) * np.sqrt(gamma) * a]  # stochastic collapse operator A
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
result1 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=1,
)
```

```python
result2 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=10,
)
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(times, result1.expect[0],
                label=r"Stochastic ME (ntraj = 1)", lw=2)
axes[0, 0].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 0].set_title("Cavity photon number (ntraj = 1)")
axes[0, 0].legend()

axes[0, 1].plot(times, result2.expect[0],
                label=r"Stochatic ME (ntraj = 10)", lw=2)
axes[0, 1].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 1].set_title("Cavity photon number (ntraj = 10)")
axes[0, 1].legend()

axes[1, 0].step(times[1:], dt * np.cumsum(result1.photocurrent[0]), lw=2)
axes[1, 0].set_title("Cummulative photon detections (ntraj = 1)")
axes[1, 1].step(times[1:], dt * np.cumsum(result2.photocurrent[0]), lw=2)
axes[1, 1].set_title("Cummulative avg. photon detections (ntraj = 10)")

fig.tight_layout()
```

## Efficient homodyne detection


The stochastic master equation for inefficient homodyne detection, when unravaling the detection part of the master equation

$\dot\rho = -i[H, \rho] + \mathcal{D}[\sqrt{1-\eta} \sqrt{\kappa} a] + \mathcal{D}[\sqrt{\eta} \sqrt{\kappa}a]$,

is given in W&M as

$d\rho = -i[H, \rho]dt + \mathcal{D}[\sqrt{1-\eta} \sqrt{\kappa} a] \rho dt 
+
\mathcal{D}[\sqrt{\eta} \sqrt{\kappa}a] \rho dt
+
\mathcal{H}[\sqrt{\eta} \sqrt{\kappa}a] \rho d\xi$

where $d\xi$ is the Wiener increment. This can be described as a standard homodyne detection with efficiency $\eta$ together with a stochastic dissipation process with collapse operator $\sqrt{(1-\eta)\kappa} a$. Alternatively we can combine the two deterministic terms on standard Lindblad for and obtain the stochastic equation (which is the form given in W&M)

$d\rho = -i[H, \rho]dt + \mathcal{D}[\sqrt{\kappa} a]\rho dt + \sqrt{\eta}\mathcal{H}[\sqrt{\kappa}a] \rho d\xi$

```python
rho0 = coherent(N, np.sqrt(5))
```

### Standard homodyne with deterministic dissipation on Lindblad form

```python
eta = 0.95
c_ops = [np.sqrt(1 - eta) * np.sqrt(gamma) * a]  # collapse operator B
sc_ops = [np.sqrt(eta) * np.sqrt(gamma) * a]  # stochastic collapse operator A
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
options = {
    "method": "platen", "store_measurement": True, "map": "parallel",
}

result = smesolve(
    H,
    rho0,
    times,
    c_ops=c_ops,
    sc_ops=sc_ops,
    e_ops=e_ops,
    ntraj=75,
    options=options
)
```

```python
plot_expectation_values([result, result_ref])
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

M = np.sqrt(eta * gamma)

for m in result.measurement:
    ax.plot(times[1:], m[0, :].real / M, "b", alpha=0.025)

ax.plot(times, result_ref.expect[1], "k", lw=2)

ax.set_ylim(-25, 25)
ax.set_xlim(0, times.max())
ax.set_xlabel("time", fontsize=12)
ax.plot(
    times[1:],
    np.mean(result.measurement, axis=0)[0, :].real / M, "b", lw=2
)
```

## Versions

```python
about()
```
