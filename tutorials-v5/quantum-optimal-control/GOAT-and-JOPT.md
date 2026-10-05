---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    name: python3
---

```python
import jax
import numpy as np
import qutip as qt
import qutip_qoc as qoc
from matplotlib import pyplot as plt
```

# QuTiP - Quantum Optimal Control - GOAT and JOPT


This tutorial notebook will guide you through the implementation of a **Hadamard** gate using a python defined control (pulse) functions.

We will optimize the control parameters using the **GOAT** and **JOPT** methods available in `qutip-qoc`.



The first step is to set up our quantum system in familiar QuTiP fashion.

Find out how you can define your system using QuTiP in our [example notebooks](https://qutip.org/qutip-tutorials/).

```python
# Define quantum system
σx, σy, σz = qt.sigmax(), qt.sigmay(), qt.sigmaz()

# Energy splitting, tunnelling, amplitude damping
ω, Δ, γ, π = 0.1, 1.0, 0.1, np.pi

# Time independent drift Hamiltonian
H_d = 1 / 2 * (ω * σz + Δ * σx)

# And bake into Liouvillian
L = qt.liouvillian(H=H_d, c_ops=[np.sqrt(γ) * qt.sigmam()])
```

In addition to the system at hand, we need to specify the control Hamiltonian and the pulse function to be optimized.

```python
def pulse(t, α):
    """Parameterized pulse function."""
    return α[0] * np.sin(α[1] * t + α[2])


def pulse_grad(t, α, i):
    """Derivative with respect to α and t (only required for GOAT)."""
    if i == 0:
        return np.sin(α[1] * t + α[2])
    elif i == 1:
        return α[0] * t * np.cos(α[1] * t + α[2])
    elif i == 2:
        return α[0] * np.cos(α[1] * t + α[2])
    elif i == 3:
        return α[0] * α[1] * np.cos(α[1] * t)
```

In the same way, we would construct a `quip.QuobjEvo`, we merge the constant Louivillian with our time dependent control Hamiltonians and their parameterized pulse functions.

```python
# Define the control Hamiltonian
H_c = [qt.liouvillian(H) for H in [σx, σy]]

# And merge it together with the associated pulse functions.
H = [
    L,  # Note the additional dictionary specifying our gradient function.
    [H_c[0], lambda t, p: pulse(t, p), {"grad": pulse_grad}],
    [H_c[1], lambda t, q: pulse(t, q), {"grad": pulse_grad}],
]

# Define the optimization goal
initial = qt.qeye(2)  # Identity
target = qt.gates.hadamard_transform()

# Super-operator form for open systems
initial = qt.sprepost(initial, initial.dag())
target = qt.sprepost(target, target.dag())
```

When defining initial and target state make sure the dimensions match.

Finally we specify the time interval in which we want to optimize the pulses and the overall objective.

```python
objective = qoc.Objective(initial, H, target)
tlist = np.linspace(0, 2 * π, 100)
```

We can now run the local optimization using the **GOAT** algorithm. The global optimizer can be enabled by an additional keyword.

```python
res_goat = qoc.optimize_pulses(
    objectives=objective,
    control_parameters={
        "p": {
            "guess": [1.0, 1.0, 1.0],
            "bounds": [(-1, 1), (0, 1), (0, 2 * π)],
        },
        "q": {
            "guess": [1.0, 1.0, 1.0],
            "bounds": [(-1, 1), (0, 1), (0, 2 * π)],
        },
    },
    tlist=tlist,
    algorithm_kwargs={
        "fid_err_targ": 0.01,
        "alg": "GOAT",
    },
    # uncomment for global optimization
    # optimizer_kwargs={"max_iter": 10}
)
```

```python
res_goat
```

Unfortunately the desired target fidelity could not yet be achieved within the given constraints.

To compare the result with the target visually we can take a quick look at the hinton plot.

```python
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
ax0.set_title("Initial")
ax1.set_title("Final")
ax2.set_title("Target")

qt.hinton(initial, ax=ax0)
qt.hinton(res_goat.final_states[0], ax=ax1)
qt.hinton(target, ax=ax2)
```

If we don't have the derivative of our pulse function available, we can use JAX defined pulse functions instead and optimize them with the **JOPT** algorithm.

```python
def sin_jax(t, α):
    return α[0] * jax.numpy.sin(α[1] * t + α[2])
```

To use the function with `qutip-jax` we need to jit compile them.

```python
@jax.jit
def sin_x_jax(t, p, **kwargs):
    return sin_jax(t, p)


@jax.jit
def sin_y_jax(t, q, **kwargs):
    return sin_jax(t, q)
```

Since JAX comes with automatic differentiation, we can drop the `"grad"` dictionary and functions.

```python
H_jax = [L, [H_c[0], sin_x_jax], [H_c[1], sin_y_jax]]
```

We simply have to change the ``algorithm_kwargs`` to run the **JOPT** algorithm.

```python
res_jopt = qoc.optimize_pulses(
    objectives=qoc.Objective(initial, H_jax, target),
    tlist=tlist,
    control_parameters={
        "p": {
            "guess": [1.0, 1.0, 1.0],
            "bounds": [(-1, 1), (0, 1), (0, 2 * π)],
        },
        "q": {
            "guess": [1.0, 1.0, 1.0],
            "bounds": [(-1, 1), (0, 1), (0, 2 * π)],
        },
    },
    algorithm_kwargs={
        "alg": "JOPT",  # Use the JAX optimizer
        "fid_err_targ": 0.01,
    },
)
```

We end up with the same result.

```python
res_jopt
```

# Multi-objective and time parameter


Both algorithms provide the option for multiple objectives, e.g. to account for variations in the Hamiltonian.

```python
H_low = [0.95 * L, [0.95 * H_c[0], sin_x_jax], [0.95 * H_c[1], sin_y_jax]]

H_high = [1.05 * L, [1.05 * H_c[0], sin_x_jax], [1.05 * H_c[1], sin_y_jax]]
```

```python
objectives = [
    qoc.Objective(initial, H_low, target, weight=0.25),
    qoc.Objective(initial, H_jax, target, weight=0.50),
    qoc.Objective(initial, H_high, target, weight=0.25),
]
```

Additionally we can loosen the fixed time constraint to reach the desired target fidelity.

```python
# same control parameters as before
control_parameters = {
    k: {
        "guess": [1.0, 1.0, 1.0],
        "bounds": [(-1, 1), (0, 1), (0, 2 * π)],
    }
    for k in ["ctrl_x", "ctrl_y"]
}

# add magic time parameter
control_parameters["__time__"] = {
    "guess": tlist[len(tlist) // 3],
    "bounds": [tlist[0], tlist[-1]],
}
```

Again we run the optimization with the **JOPT** algorithm.

```python
# run optimization
result = qoc.optimize_pulses(
    objectives,
    control_parameters,
    tlist,
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": 0.01,
    },
)
```

Finaly we manage to achieve the desired target fidelity by loosening the time constraint.

```python
result
```

```python
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
ax0.set_title("Initial")
ax1.set_title("Final")
ax2.set_title("Target")

qt.hinton(initial, ax=ax0)
qt.hinton(result.final_states[0], ax=ax1)
qt.hinton(target, ax=ax2)
```
