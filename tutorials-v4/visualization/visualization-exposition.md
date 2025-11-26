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

# Visualization demos

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import about, basis, identity, sigmax, sigmay, sigmaz

%matplotlib inline
```

## Hinton

```python
rho = qt.rand_dm(5)
```

```python
qt.hinton(rho);
```

## Sphereplot

```python
theta = np.linspace(0, np.pi, 90)
phi = np.linspace(0, 2 * np.pi, 60)
```

```python
qt.sphereplot(theta, phi, qt.orbital(theta, phi, basis(3, 0)).T);
```

```python
fig = plt.figure(figsize=(16, 4))

ax = fig.add_subplot(1, 3, 1, projection="3d")
qt.sphereplot(theta, phi, qt.orbital(theta, phi, basis(3, 0)).T, fig, ax)

ax = fig.add_subplot(1, 3, 2, projection="3d")
qt.sphereplot(theta, phi, qt.orbital(theta, phi, basis(3, 1)).T, fig, ax)

ax = fig.add_subplot(1, 3, 3, projection="3d")
qt.sphereplot(theta, phi, qt.orbital(theta, phi, basis(3, 2)).T, fig, ax);
```

# Matrix histogram

```python
qt.matrix_histogram(rho.full().real);
```

```python
qt.matrix_histogram_complex(rho.full());
```

# Plot energy levels

```python
H0 = qt.tensor(sigmaz(), identity(2)) + qt.tensor(identity(2), sigmaz())
Hint = 0.1 * qt.tensor(sigmax(), sigmax())

qt.plot_energy_levels([H0, Hint], figsize=(8, 4));
```

# Plot Fock distribution

```python
rho = (qt.coherent(15, 1.5) + qt.coherent(15, -1.5)).unit()
```

```python
qt.plot_fock_distribution(rho);
```

# Plot Wigner function and Fock distribution

```python
qt.plot_wigner_fock_distribution(rho);
```

# Plot winger function

```python
qt.plot_wigner(rho, figsize=(6, 6));
```

# Plot expectation values

```python
H = sigmaz() + 0.3 * sigmay()
e_ops = [sigmax(), sigmay(), sigmaz()]
times = np.linspace(0, 10, 100)
psi0 = (basis(2, 0) + basis(2, 1)).unit()
result = qt.mesolve(H, psi0, times, [], e_ops)
```

```python
qt.plot_expectation_values(result);
```

# Bloch sphere

```python
b = qt.Bloch()
b.add_vectors(qt.expect(H.unit(), e_ops))
b.add_points(result.expect, meth="l")
b.make_sphere()
```

# Plot spin Q-functions

```python
j = 5
psi = qt.spin_state(j, -j)
psi = qt.spin_coherent(j, np.random.rand() * np.pi,
                       np.random.rand() * 2 * np.pi)
rho = qt.ket2dm(psi)
```

```python
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
```

```python
Q, THETA, PHI = qt.spin_q_function(psi, theta, phi)
```

## 2D

```python
qt.plot_spin_distribution_2d(Q, THETA, PHI);
```

## 3D

```python
fig, ax = qt.plot_spin_distribution_3d(Q, THETA, PHI)

ax.view_init(15, 30)
```

## Combined 2D and 3D

```python
fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(1, 2, 1)
f1, a1 = qt.plot_spin_distribution_2d(Q, THETA, PHI, fig=fig, ax=ax)

ax = fig.add_subplot(1, 2, 2, projection="3d")
f2, a2 = qt.plot_spin_distribution_3d(Q, THETA, PHI, fig=fig, ax=ax)
```

# Plot spin-Wigner functions

```python
W, THETA, PHI = qt.spin_wigner(psi, theta, phi)
```

```python
fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(1, 2, 1)
f1, a1 = qt.plot_spin_distribution_2d(W.real, THETA, PHI, fig=fig, ax=ax)

ax = fig.add_subplot(1, 2, 2, projection="3d")
f2, a2 = qt.plot_spin_distribution_3d(W.real, THETA, PHI, fig=fig, ax=ax)
```

# Versions

```python
about()
```
