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
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from qutip import (ket, basis, sigmaz, tensor, qeye, mesolve, anim_schmidt,
                   complex_array_to_rgb, spin_q_function,
                   anim_spin_distribution, about)
```


```python
# a magic command enabling you to see animations in your jupyter notebook
%matplotlib notebook
```


## Time evolution of a qubit
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

fig, ani = anim_schmidt(results)
```


The magic code may not work in your environments. This may happen if you run jupyter on Linux or use Google Colab. The code below will help you.


```python
HTML(ani.to_jshtml())
```

## Animation with plots
You can make an animation with plots. Note that you cannot have it with other animations.


```python
compl_circ = np.array([[(x + 1j*y) if x**2 + y**2 <= 1 else 0j
                        for x in np.arange(-1, 1, 0.005)]
                       for y in np.arange(-1, 1, 0.005)])

fig = plt.figure(figsize=(7, 3))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.imshow(complex_array_to_rgb(compl_circ, rmax=1, theme='light'),
           extent=(-1, 1, -1, 1))
plt.tight_layout()
fig, ani = anim_schmidt(results, fig=fig, ax=ax0)
```


## Customize axes objects
You may want to add a title and labels to the animation. You can do it as you do to the plot.


```python
compl_circ = np.array([[(x + 1j*y) if x**2 + y**2 <= 1 else 0j
                        for x in np.arange(-1, 1, 0.005)]
                       for y in np.arange(-1, 1, 0.005)])

fig = plt.figure(figsize=(7, 3))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.imshow(complex_array_to_rgb(compl_circ, rmax=1, theme='light'),
           extent=(-1, 1, -1, 1))
plt.tight_layout()
fig, ani = anim_schmidt(results, fig=fig, ax=ax0)
# add title
ax0.set_title('schmidt')
ax1.set_title('color circle')
```


## Save
You can share your animations by saving them to your environment. Available file extensions (gif, mp4, etc.) dependes on your environment. More details in [the official doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html)


```python
# ani.save("schmidt.gif")
```


## Other animations
QuTiP has `qutip.Qobj` to store quantum states, but it also uses `np.array` to have them. For example, `qutip.spin_q_function` returns a matrix of values representing the spin Husimi Q function at the values specified by $\theta$ and $\phi$. Some animation functions are useful to visualize them. Here is one simple animation.


```python
theta = np.linspace(0, np.pi, 90)
phi = np.linspace(0, 2 * np.pi, 90)
Ps = list()
for i in range(0, 121, 2):
    spin = np.cos(np.pi/2*i/60)*basis(2, 0)+np.sin(np.pi/2*i/60)*basis(2, 1)
    # output np.array matrix
    Q, THETA, PHI = spin_q_function(spin, theta, phi)
    Ps.append(Q)

fig, ani = anim_spin_distribution(Ps, THETA, PHI, projection='3d',
                                  colorbar=True)
```


# Versions

```python
about()
```
