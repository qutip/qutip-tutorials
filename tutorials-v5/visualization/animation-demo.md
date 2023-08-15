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

# Animation demos
For more information about QuTiP see [http://qutip.org](http://qutip.org)


## Overview
QuTiP has animation functions to visualize the time evolution of quantum dynamics.


```python
from qutip import (ket, basis, sigmaz, tensor, qeye, mesolve,
                   complex_array_to_rgb, about)
import qutip
import numpy as np
import matplotlib.pyplot as plt
```


```python
# a magic code enabling you to see animations in your jupyter notebook
%matplotlib notebook
```


# Quick Use
Consider a system composed of two qubits. Its hamiltonian is $\sigma_z \otimes \mathbf{1}$ and the initial state is an entangled state ($\left|10\right>$+$\left|01\right>$)/$\sqrt2$.
This operator acts on the first qubit and leaves the second qubit unaffected.


```python
# Hamiltonian
H = tensor(sigmaz(), qeye(2))

# initial state
psi0 = (ket('10')+ket('01')).unit()

# list of times for which the solver should store the state vector
tlist = np.linspace(0, 3*np.pi, 100)

results = mesolve(H, psi0, tlist, [], [])

fig, ani = qutip.plot_schmidt(results.states)
```


The magic code may not work in your environments. This is likely to happen if you run jupyter on Linux or use Google Colab. The code below will help you.


```python
# !pip install IPython
# from IPython.display import HTML
# HTML(ani.to_jshtml())
```


# Add other plots
You can make an animation with plots. Note that you cannot have it with other animations.


```python
compl_circ = np.array(
    [
        [(x + 1j * y) if x ** 2 + y**2 <= 1 else 0j
            for x in np.arange(-1, 1, 0.005)]
        for y in np.arange(-1, 1, 0.005)
    ]
)

fig = plt.figure(figsize=(7, 3))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.imshow(
    complex_array_to_rgb(compl_circ, rmax=1, theme='light'),
    extent=(-1, 1, -1, 1)
)
plt.tight_layout()
fig, ani = qutip.plot_schmidt(results.states, fig=fig, ax=ax0)
```


# Customize axes objects
You may want to add a title and labels to the animation. You can do it as you do to the plot.


```python
compl_circ = np.array(
    [
        [(x + 1j * y) if x ** 2 + y**2 <= 1 else 0j
            for x in np.arange(-1, 1, 0.005)]
        for y in np.arange(-1, 1, 0.005)
    ]
)

fig = plt.figure(figsize=(7, 3))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.imshow(
    complex_array_to_rgb(compl_circ, rmax=1, theme='light'),
    extent=(-1, 1, -1, 1)
)
plt.tight_layout()
fig, ani = qutip.plot_qubism(results.states, legend_iteration=1,
                          fig=fig, ax=ax0)
# add title
ax0.set_title('qubism')
ax1.set_title('color circle')
```


## Save
You can share your animations by saving them to your environment. More details in [the official doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html)


```python
# ani.save("qubism.gif")
```


## Versions


```python
about()
```