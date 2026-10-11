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

# Parameter estimation near an exceptional point: what gets hard and what does not

Author: Maicon Esteves, 2026

### Introduction

Non-Hermitian systems can have exceptional points (EPs), where two
eigenvalues and their eigenvectors coalesce. There is an active debate on
whether operating a sensor at an EP helps or hurts. Part of the answer is
that different estimation tasks behave very differently near an EP, and
some statements that sound contradictory are both true, just about
different tasks.

In this notebook we take the simplest system with an EP, a driven qubit
with postselected decay, and compute Cramer-Rao bounds for four tasks on
the same measurement record. We will see that

1. estimating a Hamiltonian parameter directly is not affected by the EP
   (in line with arXiv:1805.11760);
2. spectral tasks, where frequencies and amplitudes are extracted from
   the record as free parameters, degrade with powers 1, 2 and 3 of the
   inverse eigenvalue gap.

The task hierarchy was measured on gravitational waveforms and quantum
hardware in https://github.com/maiconburn/recoverability-criticality
(DOI 10.5281/zenodo.22156019); here we reproduce it in a clean QuTiP
example. For the sensing debate see also arXiv:2507.17961.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip import Qobj, basis

%matplotlib inline
```

### A qubit with an exceptional point

We use the standard postselected (no jump) effective Hamiltonian of a
driven qubit whose excited state decays at rate gamma:

$$H_\mathrm{eff} = \begin{pmatrix} 0 & J \\ J & -i\gamma/2 \end{pmatrix}.$$

Its eigenvalues are $\lambda_\pm = -i\gamma/4 \pm \sqrt{J^2 - \gamma^2/16}$,
which coalesce at the exceptional point $J = \gamma/4$.

```python
gamma = 1.0
J_EP = gamma / 4

def h_eff(J):
    return Qobj([[0, J], [J, -0.5j * gamma]])

J_list = np.linspace(0.05, 0.5, 200)
evals = np.array([h_eff(J).eigenenergies() for J in J_list])

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(J_list, evals.real)
axes[0].set_xlabel("J"); axes[0].set_ylabel("Re eigenvalues")
axes[1].plot(J_list, evals.imag)
axes[1].set_xlabel("J"); axes[1].set_ylabel("Im eigenvalues")
for ax in axes:
    ax.axvline(J_EP, ls=":", color="gray")
plt.tight_layout()
```

### The measurement record

Starting from the ground state, the unnormalized conditional amplitude of
the excited state is a sum of two damped complex exponentials,

$$c_1(t) = a_+ e^{-i\lambda_+ t} + a_- e^{-i\lambda_- t},$$

which is the kind of record any two mode spectroscopy produces. We model
additive Gaussian noise of size `noise` on its quadratures.

```python
ts = np.linspace(0.2, 8.0, 120)
noise = 0.01
psi0 = basis(2, 0)

def lambdas(J):
    disc = np.sqrt(complex(J**2 - gamma**2 / 16))
    return -1j * gamma / 4 + disc, -1j * gamma / 4 - disc

def record(J):
    c1 = [( (-1j * h_eff(J) * t).expm() * psi0 ).full()[1, 0] for t in ts]
    return np.array(c1)

plt.plot(ts, record(0.4).real, label="Re c1, J = 0.40")
plt.plot(ts, record(0.26).real, label="Re c1, J = 0.26 (near EP)")
plt.xlabel("t"); plt.legend();
```

### Four tasks, one record

We compute Cramer-Rao bounds by building the Fisher matrix from
derivatives of the record and inverting it. The four tasks are

- `J direct`: estimate the Hamiltonian parameter J itself, model known;
- `amplitudes, fixed frequencies`: estimate $a_\pm$ with
  $\lambda_\pm$ known;
- `splitting, free amplitudes`: estimate the eigenvalue splitting with
  all amplitudes free;
- `amplitude, free frequencies`: estimate $a_+$ with everything free.

```python
def stack(cols):
    return np.vstack([np.concatenate([c.real, c.imag]) for c in cols]).T

def crb(cols, idx=0):
    X = stack(cols)
    F = X.T @ X / noise**2
    return np.sqrt(np.linalg.inv(F)[idx, idx])

def all_tasks(J, dJ=1e-6):
    lp, lm = lambdas(J)
    gap = abs(lp - lm)
    Ep, Em = np.exp(-1j * lp * ts), np.exp(-1j * lm * ts)
    ap, am = 0.5, -0.5
    d_lp = ap * (-1j * ts) * Ep
    d_lm = am * (-1j * ts) * Em
    d_s, d_mu = 0.5 * (d_lp - d_lm), d_lp + d_lm
    amps = [Ep, 1j * Ep, Em, 1j * Em]
    freqs = [d_s, 1j * d_s, d_mu, 1j * d_mu]
    t0 = crb([(record(J + dJ) - record(J - dJ)) / (2 * dJ)])
    t1 = crb(amps)
    t2 = crb(freqs + amps)
    t3 = crb(amps + freqs)
    return gap, t0, t1, t2, t3

Js = [0.50, 0.40, 0.325, 0.29, 0.27, 0.26, 0.255, 0.2525]
results = np.array([all_tasks(J) for J in Js])
labels = ["J direct", "amplitudes, fixed freqs",
          "splitting, free amps", "amplitude, free freqs"]
for J, row in zip(Js, results):
    print(f"J={J:.4f} gap={row[0]:.4f} " +
          " ".join(f"{v:.4g}" for v in row[1:]))
```

```python
gaps = results[:, 0]
plt.figure(figsize=(6, 4))
for k, lab in enumerate(labels):
    plt.loglog(gaps, results[:, k + 1], "o-", label=lab)
plt.gca().invert_xaxis()
plt.xlabel("eigenvalue gap"); plt.ylabel("Cramer-Rao bound")
plt.legend(); plt.tight_layout()

for k, lab in enumerate(labels):
    slope = np.polyfit(np.log(gaps), np.log(results[:, k + 1]), 1)[0]
    print(f"{lab}: scaling exponent {slope:+.2f}")
```

### What this says about the EP sensing debate

The `J direct` curve is flat: knowing the model, the EP neither helps nor
hurts the estimation of the physical parameter, which is the message of
arXiv:1805.11760. The three spectral tasks degrade with exponents that
approach 1, 2 and 3 as the gap closes: each additional piece of spectral
ignorance costs one more power of the gap. Both statements hold on the
same record of the same system. Measured versions of this hierarchy, on
numerical relativity waveforms and on IBM quantum hardware, are collected
in https://github.com/maiconburn/recoverability-criticality.

### About

QuTiP version and software details:

```python
qutip.about()
```

### Testing

```python
assert np.all(results[:, 1:] > 0)
slopes = [np.polyfit(np.log(gaps), np.log(results[:, k + 1]), 1)[0]
          for k in range(4)]
assert abs(slopes[0]) < 0.5          # J direct stays flat
assert -1.5 < slopes[1] < -0.5       # amplitudes, fixed frequencies
assert -3.0 < slopes[2] < -1.7       # splitting, free amplitudes
assert slopes[3] < slopes[2]         # free amplitudes degrade fastest
```
