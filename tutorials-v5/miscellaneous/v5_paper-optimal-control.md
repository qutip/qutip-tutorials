---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: qutip-tutorials-v5
    language: python
    name: python3
---

# QuTiPv5 Paper Example: The Quantum Optimal Control Package

Quantum systems are sensitive to external perturbations, which on the one hand can be used to perform
precise measurements or operations, but on the other hand also introduce noise and errors. Therefore,
finding the optimal control fields that achieve a desired quantum operation under various objectives (e.g.
minimum energy or maximum robustness) is a challenging and important problem. In practice, there are
often constraints and limitations on the control fields, such as bandwidth, amplitude, duration, and noise.
These factors make quantum optimal control a complex and rich field of research.

To find optimal control parameters, several methods have been developed.
Here, we look at three algorithms: *gradient ascent pulse engineering* (GRAPE), *chopped random basis* (CRAB) and *gradient optimization af analytic controls* (GOAT).
Whereas the former two have been part of the `QuTiP-QTRL` package of QuTiPv4, the latter is a new addition in version 5.
Allthogether, these algorithms are now included in the new `QuTiP-QOC` package that also adds additional functionalities such as integration with JAX via the JAX optimization technique (JOPT).

```python
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, numpy
from qutip import (about, gates, liouvillian, qeye, sigmam, sigmax, sigmay,
                   sigmaz)
from qutip_qoc import Objective, optimize_pulses

%matplotlib inline
```

## Introduction

In this example we want to implement a Hadamard gate on a single qubit.
In general, a qubit might be subject to decoherence which can be captured using the Lindblad formalism with the jump operator $\sigma_{-}$.

For simplicity, we consider a control Hamiltonian parametrized by $\sigma_x$, $\sigma_y$ and $\sigma_z$:

$H_c(t) = c_x(t) \sigma_x + c_y(t) \sigma_y + c_z(t) \sigma_z$

with $c_x(t)$, $c_y(t)$ and $c_z(t)$ as independent control parameters.
Additionally, we model a constant drift Hamiltonian

$H_d = \dfrac{1}{2} (\omega \sigma_z + \delta \sigma_x)$,

with associated energy splitting $\omega$ and tunneling rate $\delta$.
The amplitude damping rate for the collapse operator $C = \sqrt{\gamma} \sigma_-$ is denoted as $\gamma$.

```python
# energy splitting, tunneling, amplitude damping
omega = 0.1  # energy splitting
delta = 1.0  # tunneling
gamma = 0.1  # amplitude damping
sx, sy, sz = sigmax(), sigmay(), sigmaz()

Hc = [sx, sy, sz]  # control operator
Hc = [liouvillian(H) for H in Hc]

Hd = 1 / 2 * (omega * sz + delta * sx)  # drift term
Hd = liouvillian(H=Hd, c_ops=[np.sqrt(gamma) * sigmam()])

# combined operator list
H = [Hd, Hc[0], Hc[1], Hc[2]]
```

```python
# objective for optimization
initial = qeye(2)
target = gates.hadamard_transform()
```

```python
# pulse time interval
times = np.linspace(0, np.pi / 2, 100)
```

## Implementation

### GRAPE Algorithm

The GRAPE algorithm works by minimizing an infidelity loss function that measures how close the final state or unitary tranformation is to the desired target.
Starting from the provided `guess` control pulse, it optimizes evenly spaced piecewise constant pulse amplitudes.
In the end, it strives to achieve the desired target infidelity, sepcified by the `fid_err_targ` keyword.

```python
res_grape = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters={
        "ctrl_x": {"guess": np.sin(times), "bounds": [-1, 1]},
        "ctrl_y": {"guess": np.cos(times), "bounds": [-1, 1]},
        "ctrl_z": {"guess": np.tanh(times), "bounds": [-1, 1]},
    },
    tlist=times,
    algorithm_kwargs={"alg": "GRAPE", "fid_err_targ": 0.01},
)
```

### CRAB Algorithm

This algorithm is based on the idea of expanding the control fields in a random basis and optimizing the expansion coefficients $\vec{\alpha}$.
This has the advantage of using analytical control functions $c(\vec{\alpha}, t)$ on a continuous time interval, and is by default a Fourier expansion.
This reduces the search space to the function parameters.
Typically, these parameters can efficiently be calculated through direct search algorithms (like Nelder-Mead).
The basis function is only expanded for some finite number of summands and the initial basis coefficients are usually picked at random.

```python
n_params = 3  # adjust in steps of 3
alg_args = {"alg": "CRAB", "fid_err_targ": 0.01, "fix_frequency": False}
```

```python
res_crab = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters={
        "ctrl_x": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
        "ctrl_y": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
        "ctrl_z": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
    },
    tlist=times,
    algorithm_kwargs=alg_args,
)
```

### GOAT Algorithm

Similar to CRAB, this method also works with analytical control functions.
By constructing a coupled system of equations of motion, the derivative of the (time ordered) evolution operator with respect to the control parameters can be calculated after numerical forward integration.
In unconstrained settings, GOAT was found to outperform the previous described methods in terms of convergence and fidelity achievement.
The QuTiP implementation allows for arbitrary control functions provided together with their respective derivatives in a common python manner.

```python
def sin(t, c):
    return c[0] * np.sin(c[1] * t)


# derivatives
def grad_sin(t, c, idx):
    if idx == 0:  # w.r.t. c0
        return np.sin(c[1] * t)
    if idx == 1:  # w.r.t. c1
        return c[0] * np.cos(c[1] * t) * t
    if idx == 2:  # w.r.t. time
        return c[0] * np.cos(c[1] * t) * c[1]
```

```python
H = [Hd] + [[hc, sin, {"grad": grad_sin}] for hc in Hc]

ctrl_parameters = {
    id: {"guess": [1, 0], "bounds": [(-1, 1), (0, 2 * np.pi)]}  # c0 and c1
    for id in ["x", "y", "z"]
}
```

For even faster convergence QuTiP extends to original algorithm with the option to optimize controls with
respect to the overall time evolution, which can be enabled by specifying the additional time keyword
argument:

```python
# treats time as optimization variable
ctrl_parameters["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}
```

```python
# run the optimization
res_goat = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": 0.01,
    },
)
```

### JOT Algorithm - JAX integration

QuTiP's new JAX backend provides automatic differentiation capabilities that can be directly be used with the new control framework.
As with QuTiP’s GOAT implementation, any analytically defined control function can be handed to the algorithm.
However, in this method, JAX automatic differentiation abilities take care of calculating the derivative throughout the whole system evolution.
Therefore we don't have to provide any derivatives manually.
Compared to the previous example, this simply means to swap the control functions with their just-in-time compiled version.

```python
@jit
def sin_y(t, d, **kwargs):
    return d[0] * numpy.sin(d[1] * t)


@jit
def sin_z(t, e, **kwargs):
    return e[0] * numpy.sin(e[1] * t)


@jit
def sin_x(t, c, **kwargs):
    return c[0] * numpy.sin(c[1] * t)
```

```python
H = [Hd] + [[Hc[0], sin_x], [Hc[1], sin_y], [Hc[2], sin_z]]
```

```python
res_jopt = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_parameters,
    tlist=times,
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": 0.01,
    },
)
```

## Comparison of Results

After running the global and local optimization, one can compare the results obtained by the various
algorithms through a `qoc.Result` object, which provides common optimization metrics along with the `optimized_controls`.

### Pulse Amplitudes

```python
fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.54))

goat_range = times < res_goat.optimized_params[-1]
jopt_range = times < res_jopt.optimized_params[-1]

for i in range(3):
    ax[i].plot(times, res_grape.optimized_controls[i], ":", label="GRAPE")
    ax[i].plot(times, res_crab.optimized_controls[i], "-.", label="CRAB")
    ax[i].plot(
        times[goat_range],
        np.array(res_goat.optimized_controls[i])[goat_range],
        "-",
        label="GOAT",
    )
    ax[i].plot(
        times[jopt_range],
        np.array(res_jopt.optimized_controls[i])[jopt_range],
        "--",
        label="JOPT",
    )

    ax[i].set_xlabel(r"Time $t$")

ax[0].legend(loc=0)
ax[0].set_ylabel(r"Pulse amplitude $c_x(t)$", labelpad=-5)
ax[1].set_ylabel(r"Pulse amplitude $c_y(t)$", labelpad=-5)
ax[2].set_ylabel(r"Pulse amplitude $c_z(t)$", labelpad=-5)
ax[2].set_ylim(-0.2, 1.1)  # ensure equal spacing between subplots

plt.show()
```

### Infidelities and Processing Time

```python
print("GRAPE: ", res_grape.fid_err)
print(res_grape.total_seconds, " seconds")
print()
print("CRAB : ", res_crab.fid_err)
print(res_crab.total_seconds, " seconds")
print()
print("GOAT : ", res_goat.fid_err)
print(res_goat.total_seconds, " seconds")
print()
print("JOPT : ", res_jopt.fid_err)
print(res_jopt.total_seconds, " seconds")
```

## About

```python
about()
```

## Testing
