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

# Lecture 16 - Gallery of Wigner functions


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

This lecture series was developed by J.R. Johansson. The original lecture notebooks are available [here](https://github.com/jrjohansson/qutip-lectures).

This is a slightly modified version of the lectures, to work with the current release of QuTiP. You can find these lectures as a part of the [qutip-tutorials repository](https://github.com/qutip/qutip-tutorials). This lecture and other tutorial notebooks are indexed at the [QuTiP Tutorial webpage](https://qutip.org/tutorials.html).

```python
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from qutip import (about, basis, coherent, coherent_dm, displace, fock, ket2dm,
                   plot_wigner, squeeze, thermal_dm)


```

## Introduction


## Parameters

```python
N = 20
```

```python
def plot_wigner_2d_3d(psi):
    fig = plt.figure(figsize=(17, 8))

    ax = fig.add_subplot(1, 2, 1)
    plot_wigner(psi, fig=fig, ax=ax, alpha_max=6)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_wigner(psi, fig=fig, ax=ax, projection="3d", alpha_max=6)

    plt.close(fig)
    return fig
```

## Vacuum state: $\left|0\right>$

```python
psi = basis(N, 0)
plot_wigner_2d_3d(psi)
```

## Thermal states

```python
psi = thermal_dm(N, 2)
plot_wigner_2d_3d(psi)
```

## Coherent states: $\left|\alpha\right>$

```python
psi = coherent(N, 2.0)
plot_wigner_2d_3d(psi)
```

```python
psi = coherent(N, -1.0)
plot_wigner_2d_3d(psi)
```

## Superposition of coherent states

```python
psi = (coherent(N, -2.0) + coherent(N, 2.0)) / np.sqrt(2)
plot_wigner_2d_3d(psi)
```

```python
psi = (coherent(N, -2.0) - coherent(N, 2.0)) / np.sqrt(2)
plot_wigner_2d_3d(psi)
```

```python
psi = (coherent(N, -2.0) + coherent(N, -2j) + coherent(N, 2j)
       + coherent(N, 2.0)).unit()
plot_wigner_2d_3d(psi)
```

```python
psi = (coherent(N, -2.0) + coherent(N, -1j) + coherent(N, 1j)
       + coherent(N, 2.0)).unit()
plot_wigner_2d_3d(psi)
```

```python
NN = 8

fig, axes = plt.subplots(NN, 1, figsize=(5, 5 * NN),
                         sharex=True, sharey=True)
for n in range(NN):
    psi = sum(
        [coherent(N, 2 * np.exp(2j * np.pi * m / (n + 2)))
         for m in range(n + 2)]
    ).unit()
    plot_wigner(psi, fig=fig, ax=axes[n])

    # if n < NN - 1:
    #    axes[n].set_ylabel("")
```

### Mixture of coherent states

```python
psi = (coherent_dm(N, -2.0) + coherent_dm(N, 2.0)) / np.sqrt(2)
plot_wigner_2d_3d(psi)
```

## Fock states: $\left|n\right>$

```python

```

```python
for n in range(6):
    psi = basis(N, n)
    display(plot_wigner_2d_3d(psi))
```

## Superposition of Fock states

```python
NN = MM = 5

fig, axes = plt.subplots(NN, MM, figsize=(18, 18),
                         sharex=True, sharey=True)
for n in range(NN):
    for m in range(MM):
        psi = (fock(N, n) + fock(N, m)).unit()
        plot_wigner(psi, fig=fig, ax=axes[n, m])
        if n < NN - 1:
            axes[n, m].set_xlabel("")
        if m > 0:
            axes[n, m].set_ylabel("")
```

## Squeezed vacuum states

```python
psi = squeeze(N, 0.5) * basis(N, 0)
display(plot_wigner_2d_3d(psi))

psi = squeeze(N, 0.75j) * basis(N, 0)
display(plot_wigner_2d_3d(psi))

psi = squeeze(N, -1) * basis(N, 0)
display(plot_wigner_2d_3d(psi))
```

### Superposition of squeezed vacuum

```python
psi = (squeeze(N, 0.75j) * basis(N, 0) - squeeze(N, -0.75j)
       * basis(N, 0)).unit()
display(plot_wigner_2d_3d(psi))
```

### Mixture of squeezed vacuum

```python
psi = (
    ket2dm(squeeze(N, 0.75j) * basis(N, 0)) +
    ket2dm(squeeze(N, -0.75j) * basis(N, 0))
).unit()
display(plot_wigner_2d_3d(psi))
```

## Displaced squeezed vacuum

```python
psi = displace(N, 2) * squeeze(N, 0.75) * basis(N, 0)
display(plot_wigner_2d_3d(psi))
```

### Superposition of two displaced squeezed states

```python
psi = (
    displace(N, -1) * squeeze(N, 0.75) * basis(N, 0)
    - displace(N, 1) * squeeze(N, -0.75) * basis(N, 0)
).unit()
display(plot_wigner_2d_3d(psi))
```

## Versions

```python
about()
```
