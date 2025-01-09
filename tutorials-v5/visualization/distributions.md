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

This notebook shows how to use the different types of probability distributions inside qutip.

We begin by importing the necessary packages.

```python
from qutip import *
from qutip.distributions import *
import matplotlib.pyplot as plt
```
    
Then we define some quantum state.

```python
N = 20
alpha = 2.0 + 2j
rho = (coherent(N, alpha) + coherent(N, -alpha)).unit()
```

### Wigner Distribution

To use the Wigner distribution, simply use the `WignerDistribution` class and pass a quantum state to the constructor. Optionally, define a range of values for each coordinate with the parameter called `extent`. Also, define a number of data points inside the given range with the optional parameter called `steps`. From this information, the distribution is generated and can be visualized with the `visualize` method.

```python
#Generate the wigner distribution
wigner = WignerDistribution(rho, extent=[[-10, 10], [-10, 10]], steps=300)
wigner.visualize()
```

It is also possible to calculate the marginal and projection distribution along a given dimension. These methods are available for the other distributions below as well.

```python
alpha = 2.0
N = 20
rho = (coherent(N, alpha) + coherent(N, -alpha)).unit()
wigner = WignerDistribution(rho)
wigner.visualize()

#x axis
wigner_x = wigner.marginal(dim=0)
wigner_x.visualize()

#y axis
wigner_y = wigner.marginal(dim=1)
wigner_y.visualize()

#projection
proj = wigner.project(dim=1)
proj.visualize()
```

### Husimi-Q Distribution

To use the Husimi-Q distribution, simply use the `QDistribution` class and pass it a quantum state. Again, `extent` and `steps` are optional parameters for this distribution.

```python
hq = QDistribution(rho, extent=[[-10, 10], [-10, 10]], steps=300)
hq.visualize()
```

### Two-mode quadrature correlations
We start with a new quantum state.

```python
alpha = 1.0 
psi = (tensor(coherent(N, alpha), basis(N, 0)) + tensor(basis(N, 0), coherent(N, -alpha))).unit() 
rho = (ket2dm(tensor(coherent(N, alpha), basis(N, 0))) + ket2dm(tensor(basis(N, 0), coherent(N, -alpha)))).unit() 
```

In this case, we use the `TwoModeQuadratureCorrelation` class.

```python
two_mode_psi = TwoModeQuadratureCorrelation(psi)
two_mode_psi.visualize()

two_mode_rho = TwoModeQuadratureCorrelation(rho)
two_mode_rho.visualize()
```

### Harmonic Oscillator Wave Function
Here, we display the spatial distribution of the wave function for the harmonic oscillator (n=0 to n=7) with the `HarmonicOscillatorWaveFunction` class.

```python
M=8

fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

for n in range(M):
    psi = fock(N, n)
    wf = HarmonicOscillatorWaveFunction(psi, 1.0, extent=[-10, 10])
    wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))
```

### Harmonic Oscillator Probability Function

The class `HarmonicOscillatorProbabilityFunction` is the squared magnitude of the data that would normally be in `HarmonicOscillatorWaveFunction`. We use the same example as before.

```python
M=8

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
