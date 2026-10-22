---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: esp26
    language: python
    name: python3
---

# Interacting Qubits

Welcome to the first part of this tutorial at Euroscipy 2026.
In this notebook we will show how to visualize a qubit state and how to simulate the dynamics of interacting qubits with and without an environment.

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
```

# Visualizing a Qubit

```python
def bloch_state(theta, phi):
    """
    Return a single-qubit state |psi> on the Bloch sphere.
    theta, phi in radians.
    """
    return np.cos(theta / 2) * qt.basis(2, 0) + np.exp(1j * phi) * np.sin(
        theta / 2
    ) * qt.basis(2, 1)


def plot_on_bloch(states, labels=None):
    """
    Plot one or more states on the Bloch sphere.

    states: list of Qobj kets or a single Qobj.
    labels: optional list of labels.
    """
    if isinstance(states, qt.Qobj):
        states = [states]
    b = qt.Bloch()
    for i, s in enumerate(states):
        b.add_states(s)
        if labels is not None:
            b.add_annotation(s, labels[i])
    b.show()
```

```python
# Example: |0>, |1>, |+>, |-> on the Bloch sphere
zero = qt.basis(2, 0)
one = qt.basis(2, 1)
plus = (zero + one) / np.sqrt(2)
minus = (zero - one) / np.sqrt(2)

plot_on_bloch(
    [zero, one, plus, minus],
    labels=[r"$|0\rangle$", r"$|1\rangle$", r"$|+\rangle$", r"$|-\rangle$"],
)
```

# Two Interacting Qubits

```python
# Building the Hamiltonian
epsilonA = 0.5
epsilonB = 0.5
g = 0.2

szA = qt.sigmaz() & qt.qeye(2)  # sigma_z
szB = qt.qeye(2) & qt.sigmaz()
sxA = qt.sigmax() & qt.qeye(2)  # sigma_x
sxB = qt.qeye(2) & qt.sigmax()

spA = qt.sigmap() & qt.qeye(2)  # sigma_+
spB = qt.qeye(2) & qt.sigmap()
smA = qt.sigmam() & qt.qeye(2)  # sigma_-
smB = qt.qeye(2) & qt.sigmam()

H = epsilonA * szA + epsilonB * szB + g * (spA * smB + smA * spB)

print(H)
```

```python
psi0 = qt.basis(2, 0) & qt.basis(2, 1)  # initial state
tlist = np.linspace(0, 40, 100)  # simulation time
```

```python
se_result = qt.sesolve(H, psi0, tlist, e_ops=[szA, szB])
```

```python
lw = 3
fs = 14
plt.plot(tlist, se_result.expect[0], label="A", c="firebrick", linewidth=lw)
plt.plot(tlist, se_result.expect[1], label="B", c="steelblue", linewidth=lw)
plt.xlabel("Time", fontsize=fs)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=fs)
plt.legend(fontsize=fs)
plt.show()
```

# Two Qubits and an Environment

```python
# dissipation rate
gA = np.sqrt(4e-2)
gB = np.sqrt(4e-2)

c_ops = [gA * smA, gB * smB]
```

```python
me_result = qt.mesolve(H, psi0, tlist, c_ops, e_ops=[szA, szB])
```

```python
plt.plot(tlist, me_result.expect[0], label="A", c="firebrick", linewidth=lw)
plt.plot(tlist, me_result.expect[1], label="B", c="steelblue", linewidth=lw)
plt.xlabel("Time", fontsize=fs)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=fs)
plt.legend(fontsize=fs)
plt.show()
```

```python
qt.about()
```
