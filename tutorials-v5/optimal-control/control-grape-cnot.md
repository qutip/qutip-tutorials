---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: qiskit-stable8
    language: python
    name: python3
---

### GRAPE calculation of control fields for cnot implementation

[This is an updated implementation based on the deprecated notebook of GRAPE CNOT implementation by Robert Johansson](https://nbviewer.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-cnot.ipynb)

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
# the library for quantum control
import qutip_qtrl.pulseoptim as qtrl
from qutip.ipynbtools import version_table
```

```python
# total duration
T = 2 * np.pi
# number of time steps
times = np.linspace(0, T, 500)
```

```python
U_0 = qt.operators.identity(4)
U_target = qt.core.gates.cnot()
```

### Starting Point

```python
U_0
```

### Target Operator

```python
U_target
```

```python
# Drift Hamiltonian
g = 0
H_drift = g * (
    qt.tensor(qt.sigmax(), qt.sigmax()) + qt.tensor(qt.sigmay(), qt.sigmay())
)
```

```python
H_drift
```

```python
H_ctrl = [
    qt.tensor(qt.sigmax(), qt.identity(2)),
    qt.tensor(qt.sigmay(), qt.identity(2)),
    qt.tensor(qt.sigmaz(), qt.identity(2)),
    qt.tensor(qt.identity(2), qt.sigmax()),
    qt.tensor(qt.identity(2), qt.sigmay()),
    qt.tensor(qt.identity(2), qt.sigmaz()),
    qt.tensor(qt.sigmax(), qt.sigmax()),
    qt.tensor(qt.sigmay(), qt.sigmay()),
    qt.tensor(qt.sigmaz(), qt.sigmaz()),
]
```

```python
H_ctrl
```

```python
H_labels = [
    r"$u_{1x}$",
    r"$u_{1y}$",
    r"$u_{1z}$",
    r"$u_{2x}$",
    r"$u_{2y}$",
    r"$u_{2z}$",
    r"$u_{xx}$",
    r"$u_{yy}$",
    r"$u_{zz}$",
]
```

## GRAPE

```python
result = qtrl.optimize_pulse_unitary(
    H_drift,
    H_ctrl,
    U_0,
    U_target,
    num_tslots=500,
    evo_time=(2 * np.pi),
    # this attribute is crucial for convergence!!
    amp_lbound=-(2 * np.pi * 0.05),
    amp_ubound=(2 * np.pi * 0.05),
    fid_err_targ=1e-9,
    max_iter=500,
    max_wall_time=60,
    alg="GRAPE",
    optim_method="FMIN_L_BFGS_B",
    method_params={
        "disp": True,
        "maxiter": 1000,
    },
)
```

```python
for attr in dir(result):
    if not attr.startswith("_"):
        print(f"{attr}: {getattr(result, attr)}")

# --> array[num_tslots, n_ctrls]
print(np.shape(result.final_amps))
```

## Plot control fields for cnot gate in the presense of single-qubit tunnelling

```python
def plot_control_amplitudes(times, final_amps, labels):
    num_controls = final_amps.shape[1]

    y_max = 0.1  # Fixed y-axis scale
    y_min = -0.1

    for i in range(num_controls):
        fig, ax = plt.subplots(figsize=(8, 3))

        for j in range(num_controls):
            # Highlight the current control
            color = "black" if i == j else "gray"
            alpha = 1.0 if i == j else 0.1
            ax.plot(
                times,
                final_amps[:, j],
                label=labels[j],
                color=color,
                alpha=alpha
            )
        ax.set_title(f"Control Fields Highlighting: {labels[i]}")
        ax.set_xlabel("Time")
        ax.set_ylabel(labels[i])
        ax.set_ylim(y_min, y_max)  # Set fixed y-axis limits
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

plot_control_amplitudes(times, result.final_amps / (2 * np.pi), H_labels)
```

## Fidelity/overlap

```python

U_target
```

```python
U_f=result.evo_full_final
U_f.dims = [[2,2], [2, 2]]
```

```python
U_f
```

```python
def overlap(U_target, U_f):
    """
    Calculate the overlap between the target unitary U_target and
    the final unitary U_f.

    Parameters:
    U_target (qutip.Qobj): Target unitary operator.
    U_f (qutip.Qobj): Final unitary operator.

    Returns:
    float: Real part of the overlap value.
    float: Fidelity (absolute square of the overlap).
    """
    # dividing over U_target.shape[0] is for normalization
    overlap_value = (U_target.dag() * U_f).tr() / U_target.shape[0]
    fidelity = abs(overlap_value) ** 2
    return overlap_value.real, fidelity


# Example usage
overlap_real, fidelity = overlap(U_target, U_f)
print(f"Overlap (real part): {overlap_real}")
print(f"Fidelity: {fidelity}")
```

```python
np.shape(U_f)
```

## Proceess tomography


Quantum Process Tomography (QPT) is a technique used to characterize an unknown quantum operation by reconstructing its process matrix (also called the χ (chi) matrix). This matrix describes how an input quantum state is transformed by the operation.


Defines the basis operators 
{
𝐼
,
𝑋
,
𝑌
,
𝑍
}
for the two-qubit system.

These operators form a complete basis to describe any quantum operation in the Pauli basis.


### Ideal cnot gate

```python
op_basis = [[qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]] * 2
op_label = [["i", "x", "y", "z"]] * 2
```

U_target is the ideal CNOT gate.

qt.to_super(U_target) converts it into superoperator form, which is necessary for QPT.

qt.qpt(U_i_s, op_basis) computes the χ matrix for the ideal gate.

```python
fig = plt.figure(figsize=(12, 6))

U_i_s = qt.to_super(U_target)

chi = qt.qpt(U_i_s, op_basis)

fig = qt.qpt_plot_combined(chi, op_label, fig=fig, threshold=0.001)
```

```python
op_basis = [[qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]] * 2
op_label = [["i", "x", "y", "z"]] * 2
```

```python
fig = plt.figure(figsize=(12, 6))

U_f_s = qt.to_super(U_f)

chi = qt.qpt(U_f_s, op_basis)

fig = qt.qpt_plot_combined(chi, op_label, fig=fig, threshold=0.01)
```

## Versions


```python
version_table()
```
