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

# Lecture 2A - Simulation of a two-qubit gate using a resonator as coupler

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

This lecture series was developed by J.R. Johansson. The original lecture notebooks are available [here](https://github.com/jrjohansson/qutip-lectures).

This is a slightly modified version of the lectures, to work with the current release of QuTiP. You can find these lectures as a part of the [qutip-tutorials repository](https://github.com/qutip/qutip-tutorials). This lecture and other tutorial notebooks are indexed at the [QuTiP Tutorial webpage](https://qutip.org/tutorials.html).

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, concurrence, destroy, expect, fidelity,
                   ket2dm, mesolve, phasegate, ptrace, qeye, sigmaz, sqrtiswap,
                   tensor)
from scipy.special import sici


```

## Parameters

```python
N = 10

wc = 5.0 * 2 * np.pi
w1 = 3.0 * 2 * np.pi
w2 = 2.0 * 2 * np.pi

g1 = 0.01 * 2 * np.pi
g2 = 0.0125 * 2 * np.pi

tlist = np.linspace(0, 100, 500)

width = 0.5

# resonant SQRT iSWAP gate
T0_1 = 20
T_gate_1 = (1 * np.pi) / (4 * g1)

# resonant iSWAP gate
T0_2 = 60
T_gate_2 = (2 * np.pi) / (4 * g2)
```

### Operators, Hamiltonian and initial state 

```python
# cavity operators
a = tensor(destroy(N), qeye(2), qeye(2))
n = a.dag() * a

# operators for qubit 1
sm1 = tensor(qeye(N), destroy(2), qeye(2))
sz1 = tensor(qeye(N), sigmaz(), qeye(2))
n1 = sm1.dag() * sm1

# operators for qubit 2
sm2 = tensor(qeye(N), qeye(2), destroy(2))
sz2 = tensor(qeye(N), qeye(2), sigmaz())
n2 = sm2.dag() * sm2
```

```python
# Hamiltonian using QuTiP
Hc = a.dag() * a
H1 = -0.5 * sz1
H2 = -0.5 * sz2
Hc1 = g1 * (a.dag() * sm1 + a * sm1.dag())
Hc2 = g2 * (a.dag() * sm2 + a * sm2.dag())

H = wc * Hc + w1 * H1 + w2 * H2 + Hc1 + Hc2
```

```python
H
```

```python
# initial state: start with one of the qubits in its excited state
psi0 = tensor(basis(N, 0), basis(2, 1), basis(2, 0))
```

# Ideal two-qubit iSWAP gate

```python
def step_t(w1, w2, t0, width, t):
    """
    Step function that goes from w1 to w2 at time t0
    as a function of t.
    """
    return w1 + (w2 - w1) * (t > t0)


fig, axes = plt.subplots(1, 1, figsize=(8, 2))
axes.plot(tlist, [step_t(0.5, 1.5, 50, 0.0, t) for t in tlist], "k")
axes.set_ylim(0, 2)
fig.tight_layout()
```

```python
def wc_t(t, args=None):
    return wc


def w1_t(t, args=None):
    return (
        w1
        + step_t(0.0, wc - w1, T0_1, width, t)
        - step_t(0.0, wc - w1, T0_1 + T_gate_1, width, t)
    )


def w2_t(t, args=None):
    return (
        w2
        + step_t(0.0, wc - w2, T0_2, width, t)
        - step_t(0.0, wc - w2, T0_2 + T_gate_2, width, t)
    )


H_t = [[Hc, wc_t], [H1, w1_t], [H2, w2_t], Hc1 + Hc2]
```

### Evolve the system

```python
res = mesolve(H_t, psi0, tlist, [], [])
```

### Plot the results

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r",
             linewidth=2, label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b",
             linewidth=2, label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g",
             linewidth=2, label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
```

### Inspect the final state

```python
# extract the final state from the result of the simulation
rho_final = res.states[-1]
```

```python
# trace out the resonator mode and print the two-qubit density matrix
rho_qubits = ptrace(rho_final, [1, 2])
rho_qubits
```

```python
# compare to the ideal result of the sqrtiswap gate (plus phase correction)
# for the current initial state
rho_qubits_ideal = ket2dm(
    tensor(phasegate(0), phasegate(-np.pi / 2))
    * sqrtiswap()
    * tensor(basis(2, 1), basis(2, 0))
)
rho_qubits_ideal
```

### Fidelity and concurrence

```python
fidelity(rho_qubits, rho_qubits_ideal)
```

```python
concurrence(rho_qubits)
```

# Dissipative two-qubit iSWAP gate


### Define collapse operators that describe dissipation



```python
kappa = 0.0001
gamma1 = 0.005
gamma2 = 0.005

c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma1) * sm1, np.sqrt(gamma2) * sm2]
```

### Evolve the system

```python
res = mesolve(H_t, psi0, tlist, c_ops, [])
```

### Plot the results

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r", linewidth=2,
             label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b", linewidth=2,
             label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g", linewidth=2,
             label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
```

### Fidelity and concurrence

```python
rho_final = res.states[-1]
rho_qubits = ptrace(rho_final, [1, 2])
```

```python
fidelity(rho_qubits, rho_qubits_ideal)
```

```python
concurrence(rho_qubits)
```

# Two-qubit iSWAP gate: Finite pulse rise time

```python
def step_t(w1, w2, t0, width, t):
    """
    Step function that goes from w1 to w2 at time t0
    as a function of t, with finite rise time defined
    by the parameter width.
    """
    return w1 + (w2 - w1) / (1 + np.exp(-(t - t0) / width))


fig, axes = plt.subplots(1, 1, figsize=(8, 2))
axes.plot(tlist, [step_t(0.5, 1.5, 50, width, t) for t in tlist], "k")
axes.set_ylim(0, 2)
fig.tight_layout()
```

### Evolve the system

```python
res = mesolve(H_t, psi0, tlist, [], [])
```

### Plot the results

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r", linewidth=2,
             label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b", linewidth=2,
             label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g", linewidth=2,
             label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
```

### Fidelity and concurrence

```python
rho_final = res.states[-1]
rho_qubits = ptrace(rho_final, [1, 2])
```

```python
fidelity(rho_qubits, rho_qubits_ideal)
```

```python
concurrence(rho_qubits)
```

# Two-qubit iSWAP gate: Finite rise time with overshoot

```python
def step_t(w1, w2, t0, width, t):
    """
    Step function that goes from w1 to w2 at time t0
    as a function of t, with finite rise time and
    and overshoot defined by the parameter width.
    """

    return w1 + (w2 - w1) * (0.5 + sici((t - t0) / width)[0] / (np.pi))


fig, axes = plt.subplots(1, 1, figsize=(8, 2))
axes.plot(tlist, [step_t(0.5, 1.5, 50, width, t) for t in tlist], "k")
axes.set_ylim(0, 2)
fig.tight_layout()
```

### Evolve the system

```python
res = mesolve(H_t, psi0, tlist, [], [])
```

### Plot the results

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r", linewidth=2,
             label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b", linewidth=2,
             label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g", linewidth=2,
             label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
```

### Fidelity and concurrence

```python
rho_final = res.states[-1]
rho_qubits = ptrace(rho_final, [1, 2])
```

```python
fidelity(rho_qubits, rho_qubits_ideal)
```

```python
concurrence(rho_qubits)
```

# Two-qubit iSWAP gate: Finite pulse rise time and dissipation

```python
# increase the pulse rise time a bit
width = 0.6

# high-Q resonator but dissipative qubits
kappa = 0.00001
gamma1 = 0.005
gamma2 = 0.005

c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma1) * sm1, np.sqrt(gamma2) * sm2]
```

### Evolve the system

```python
res = mesolve(H_t, psi0, tlist, c_ops, [])
```

### Plot results

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r", linewidth=2,
             label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b", linewidth=2,
             label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g", linewidth=2,
             label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
```

### Fidelity and concurrence

```python
rho_final = res.states[-1]
rho_qubits = ptrace(rho_final, [1, 2])
```

```python
fidelity(rho_qubits, rho_qubits_ideal)
```

```python
concurrence(rho_qubits)
```

# Two-qubit iSWAP gate: Using tunable resonator and fixed-frequency qubits

```python
# reduce the rise time
width = 0.25


def wc_t(t, args=None):
    return (
        wc
        - step_t(0.0, wc - w1, T0_1, width, t)
        + step_t(0.0, wc - w1, T0_1 + T_gate_1, width, t)
        - step_t(0.0, wc - w2, T0_2, width, t)
        + step_t(0.0, wc - w2, T0_2 + T_gate_2, width, t)
    )


H_t = [[Hc, wc_t], H1 * w1 + H2 * w2 + Hc1 + Hc2]
```

### Evolve the system

```python
res = mesolve(H_t, psi0, tlist, c_ops, [])
```

### Plot the results

```python
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

axes[0].plot(
    tlist,
    np.array(list(map(wc_t, tlist))) / (2 * np.pi),
    "r",
    linewidth=2,
    label="cavity",
)
axes[0].plot(
    tlist,
    np.array(list(map(w1_t, tlist))) / (2 * np.pi),
    "b",
    linewidth=2,
    label="qubit 1",
)
axes[0].plot(
    tlist,
    np.array(list(map(w2_t, tlist))) / (2 * np.pi),
    "g",
    linewidth=2,
    label="qubit 2",
)
axes[0].set_ylim(1, 6)
axes[0].set_ylabel("Energy (GHz)", fontsize=16)
axes[0].legend()

axes[1].plot(tlist, np.real(expect(n, res.states)), "r", linewidth=2,
             label="cavity")
axes[1].plot(tlist, np.real(expect(n1, res.states)), "b", linewidth=2,
             label="qubit 1")
axes[1].plot(tlist, np.real(expect(n2, res.states)), "g", linewidth=2,
             label="qubit 2")
axes[1].set_ylim(0, 1)

axes[1].set_xlabel("Time (ns)", fontsize=16)
axes[1].set_ylabel("Occupation probability", fontsize=16)
axes[1].legend()

fig.tight_layout()
```

### Fidelity and concurrence

```python
rho_final = res.states[-1]
rho_qubits = ptrace(rho_final, [1, 2])
```

```python
fidelity(rho_qubits, rho_qubits_ideal)
```

```python
concurrence(rho_qubits)
```

### Software versions

```python
about()
```
