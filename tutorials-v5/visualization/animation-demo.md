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

```python
from qutip import (ket, basis, sigmaz, tensor, qeye, mesolve,
                   complex_array_to_rgb, about)
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
```


```python
# if this does not work on your environment, use qt.make_html_video
# to see animations.
%matplotlib notebook
```

## Time evolution of an operator


```python
# Hamiltonian
H = qt.sigmay().unit()

# initial state
psi0 = qt.sigmax().unit()

# list of times for which the solver should store the state vector
tlist = np.linspace(0, 2*np.pi, 50)

results = mesolve(H, psi0, tlist, [], [])

fig, ani = qt.hinton(results.states)

```


## Time evolution of a ket


```python
# Hamiltonian
H = qt.rand_dm(5)

# initial state
psi0 = basis(5, 0)

# list of times for which the solver should store the state vector
tlist = np.linspace(0, 10, 100)

results = mesolve(H, psi0, tlist, [], [])
```


```python
fig, ani = qt.plot_wigner(results.states, projection='3d', colorbar=True)
```


```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig, ani = qt.plot_fock_distribution(results.states, fig=fig, ax=ax)
```


```python
fig, ani = qt.matrix_histogram(results.states, bar_style='abs')
# save and show the animation
# plt.close()
# html = qt.make_html_video(ani, 'matrix.gif')
# html
```


```python
W = list()
for state in results.states:
    wig = qt.wigner_transform(state, 2, False, 50, ["x"])
    W.append(wig)

fig, ani = qt.plot_wigner_sphere(W[:30], reflections=True)
```


## Qubism animation

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
```


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
fig, ani = qt.plot_schmidt(results.states, fig=fig, ax=ax0)
```


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
fig, ani = qt.plot_qubism(results.states, legend_iteration=1,
                          fig=fig, ax=ax0)
ax0.set_title('qubism')
ax1.set_title('color circle')
```


## Spin distribution


```python
theta = np.linspace(0, np.pi, 180)
phi = np.linspace(0, 2 * np.pi, 180)

Ps = list()s
for i in range(121):
    c = np.cos(np.pi/2*i/60)
    s = np.sin(np.pi/2*i/60)
    vec = c*basis(2, 0) + s*basis(2, 1)
    Q, THETA, PHI = qt.spin_q_function(vec, theta, phi)
    Ps.append(Q)

fig, ani= qt.plot_spin_distribution(Ps, THETA, PHI,
                                    projection='3d', colorbar=True)
```


## Wave function


```python
theta = np.linspace(0, np.pi, 90)
phi = np.linspace(0, 2 * np.pi, 90)
V = list()
for i in range(61):
    c = np.cos(np.pi/2*i/60)
    s = np.sin(np.pi/2*i/60)
    vec = c*qt.basis(3, 0) + s*qt.basis(3, 2)
    values = qt.orbital(theta, phi, vec).T
    V.append(values)
fig, ani = qt.sphereplot(theta, phi, V)
```


## Versions


```python
about()
```