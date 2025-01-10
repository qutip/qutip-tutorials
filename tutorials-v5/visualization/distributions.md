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

# Using qutip.distributions

Author: Mathis Beaudoin (2025)

### Introduction

This notebook shows how to use probability distributions inside QuTiP. We begin by importing the necessary packages.

```python
from qutip import fock
from qutip.distributions import HarmonicOscillatorWaveFunction, HarmonicOscillatorProbabilityFunction
import matplotlib.pyplot as plt
```

### Harmonic Oscillator Wave Function

Here, we display the spatial distribution of the wave function for the harmonic oscillator (n=0 to n=7) with the `HarmonicOscillatorWaveFunction()` class.

Optionally, define a range of values for each coordinate with the parameter called `extent`. Also, define a number of data points inside the given range with the optional parameter called `steps`. From this information, the distribution is generated and can be visualized with the `.visualize()` method.

It is also possible to calculate, along a given axis, the marginal distribution with `.marginal()` or the projection distribution with `.project()`.

```python
M=8
N=20

fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

for n in range(M):
    psi = fock(N, n)
    wf = HarmonicOscillatorWaveFunction(psi, 1.0, extent=[-10, 10])
    wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))
```

### Harmonic Oscillator Probability Function

The class `HarmonicOscillatorProbabilityFunction()` is the squared magnitude of the data that would normally be in `HarmonicOscillatorWaveFunction()`. We use the same example as before.

```python
M=8
N=20

fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

for n in range(M):
    psi = fock(N, n)
    wf = HarmonicOscillatorProbabilityFunction(psi, 1.0, extent=[-10, 10])
    wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))
```

### About

```python
qutip.about()
```
