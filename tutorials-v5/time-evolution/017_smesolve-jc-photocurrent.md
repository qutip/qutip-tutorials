---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Stochastic Solver: Photo-current detection in a JC model


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from qutip import about, destroy, fock, liouvillian, identity, mcsolve, mesolve

%matplotlib inline
rcParams["font.family"] = "STIXGeneral"
rcParams["mathtext.fontset"] = "stix"
rcParams["font.size"] = "14"
```

```python
N = 15
w0 = 1.0 * 2 * np.pi
g = 0.2 * 2 * np.pi
times = np.linspace(0, 15, 151)
dt = times[1] - times[0]
gamma = 0.01
kappa = 0.1
ntraj = 150
```

```python
a = destroy(N) & identity(2)
sm = identity(N) & destroy(2)
```

```python
H = w0 * a.dag() * a + w0 * sm.dag() * sm + g * (sm * a.dag() + sm.dag() * a)
```

```python
rho0 = fock(N, 5) & fock(2, 0)
```

```python
e_ops = [a.dag() * a, a + a.dag(), sm.dag() * sm]
```

### Highly efficient detection

```python
c_ops = [np.sqrt(gamma) * sm]  # collapse operator for qubit
sc_ops = [np.sqrt(kappa) * a]  # stochastic collapse for resonator
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
result1 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=1,
    options={"store_states": True},
)
```

Run the `smesolve` solver in parallel by passing the keyword argument `map_func=parallel_map`:

```python
result2 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=ntraj,
    options={"store_states": True, "map": "parallel"},
)
```

```python
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

axes[0, 0].plot(
    times, result1.expect[0], label=r"Stochastic ME (ntraj = 1)", lw=2
)
axes[0, 0].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 0].set_title("Cavity photon number (ntraj = 1)")
axes[0, 0].legend()

axes[1, 0].plot(
    times, result2.expect[0], label=r"Stochatic ME (ntraj = %d)" % ntraj, lw=2
)
axes[1, 0].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[1, 0].set_title("Cavity photon number (ntraj = 10)")
axes[1, 0].legend()


axes[0, 1].plot(times,
                result1.expect[2], label=r"Stochastic ME (ntraj = 1)", lw=2)
axes[0, 1].plot(times, result_ref.expect[2], label=r"Lindblad ME", lw=2)
axes[0, 1].set_title("Qubit excition probability (ntraj = 1)")
axes[0, 1].legend()

axes[1, 1].plot(
    times, result2.expect[2], label=r"Stochatic ME (ntraj = %d)" % ntraj, lw=2
)
axes[1, 1].plot(times, result_ref.expect[2], label=r"Lindblad ME", lw=2)
axes[1, 1].set_title("Qubit excition probability (ntraj = %d)" % ntraj)
axes[1, 1].legend()


axes[0, 2].step(times[1:], dt * np.cumsum(result1.photocurrent[0]), lw=2)
axes[0, 2].set_title("Cummulative photon detections (ntraj = 1)")
axes[1, 2].step(times[1:], dt * np.cumsum(result2.photocurrent[0]), lw=2)
axes[1, 2].set_title("Cummulative avg. photon detections (ntraj = %d)" % ntraj)

fig.tight_layout()
```

## Versions

```python
about()
```
