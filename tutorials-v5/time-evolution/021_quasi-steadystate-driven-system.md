---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: dev
  language: python
  name: dev
---

# Steady-State: Time-dependent (periodic) quantum system

Authors: J.R. Johansson and P.D. Nation

Updated by: M. Gobbo (2024)

### Introduction
In this notebook, we will find the steady state of a driven qubit using the `steadystate()`, `propagator_steadystate()`, and `steadystate_floquet()` solver methods. The results will be compared with the master equation solver `mesolve()` implemented in QuTiP.

You can also find more on solving for steady-state solutions with QuTiP [here](https://qutip.readthedocs.io/en/latest/guide/guide-steady.html).

### Imports
Here we import the required modules for this example.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from qutip import (
    about,
    basis,
    destroy,
    expect,
    mesolve,
    steadystate,
    propagator,
    propagator_steadystate,
    sigmax,
    sigmaz,
    steadystate_floquet,
)
```

### System setup
We consider the usual driven system with the following Hamiltonian: 

$$ H = - \frac{\Delta}{2} \sigma_x - \frac{\epsilon_0}{2} \sigma_z + \frac{A}{2} \sigma_z \sin(\omega t) $$

We also assume a coupling with the external heat bath described by a coupling constant $\kappa_1$, the temperature of the heat bath is defined via the average photon number $\langle n \rangle$. In addition, we assume a variation in the phase of the qubit described by a collapse operator with a constant $\kappa_2$.

```{code-cell} ipython3
# Parameters
delta = (2 * np.pi) * 0.3
eps_0 = (2 * np.pi) * 1.0
A = (2 * np.pi) * 0.05
w = (2 * np.pi) * 1.0
kappa_1 = 0.15
kappa_2 = 0.05

# Operators
sx = sigmax()
sz = sigmaz()
sm = destroy(2)

# Non-driving Hamiltonian
H0 = -delta / 2.0 * sx - eps_0 / 2.0 * sz

# Driving Hamiltonian
H1 = A / 2.0 * sz
args = {"w": w}

# Total Hamiltonian
H = [H0, [H1, "np.sin(w*t)"]]

# Collapse operators
c_op_list = []

# Thermal population
n_th = 0.5

# Relaxation
rate = kappa_1 * (1 + n_th)

if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sm)

# Excitation
rate = kappa_1 * n_th

if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sm.dag())

# Dephasing
rate = kappa_2
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sz)
```

### Time evolution

```{code-cell} ipython3
# Period
T = 2 * np.pi / w

# Simulation time
t_list = np.linspace(0, 50, 500)

# Initial state
psi_0 = basis(2, 0)
psi_1 = basis(2, 1)
```

### Master equation

```{code-cell} ipython3
# Solve with the Master equation
output = mesolve(H, psi_0, t_list, c_op_list, [psi_1 * psi_1.dag()], args)
prob_me = output.expect[0]
```

### Steady state method

```{code-cell} ipython3
# Evaluate the steady state using the steadystate method
rho_ss = steadystate(H0, c_op_list, method="power", solver="spsolve")
prob_ss = expect(psi_1 * psi_1.dag(), rho_ss)
```

### Propagator method

```{code-cell} ipython3
# Evaluate the steady state using the propagator method
U = propagator(H, T, c_op_list, args)
rho_pss = propagator_steadystate(U)
prob_pss = expect(psi_1 * psi_1.dag(), rho_pss)
```

### Floquet method

```{code-cell} ipython3
# Evaluate the steady state using the Floquet method
rho_fss = steadystate_floquet(H0, c_op_list, H1, w)
prob_fss = expect(psi_1 * psi_1.dag(), rho_fss)
```

```{code-cell} ipython3
# Figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot
ax.plot(t_list, prob_me, label='Master equation')
ax.plot(t_list, prob_ss * np.ones(t_list.shape[0]), label='Steady state')
ax.plot(t_list, prob_pss * np.ones(t_list.shape[0]), label='Propagator steady state')
ax.plot(t_list, prob_fss * np.ones(t_list.shape[0]), label='Floquet steady state')
ax.set_ylim(0, 1)

# Inset
ax_inset = inset_axes(ax, width="60%", height="80%", loc='center', bbox_to_anchor=(0.2, 0.45, 0.5, 0.45), bbox_transform=ax.transAxes)
ax_inset.plot(t_list, prob_me, label='Master Equation')
ax_inset.plot(t_list, prob_ss * np.ones(t_list.shape[0]), label='Steady state')
ax_inset.plot(t_list, prob_pss * np.ones(t_list.shape[0]), label='Propagator steady state')
ax_inset.plot(t_list, prob_fss * np.ones(t_list.shape[0]), label='Floquet steady state')
ax_inset.set_xlim(40, 50)
ax_inset.set_ylim(0.25, 0.3)
ax_inset.set_xticks([40, 45, 50])
ax_inset.set_yticks([0.25, 0.27, 0.3])
mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

# Labels
ax.set_xlabel("Time")
ax.set_ylabel("$P\,(|1\\rangle)$")
ax.set_title("Excitation probabilty of qubit")
ax.legend()
plt.show()
```

### About

```{code-cell} ipython3
about()
```

### Testing

```{code-cell} ipython3
np.testing.assert_allclose(prob_ss, np.mean(prob_me[200:]), atol=1e-2)
np.testing.assert_allclose(prob_pss, np.mean(prob_me[200:]), atol=1e-2)
np.testing.assert_allclose(prob_fss, np.mean(prob_me[200:]), atol=1e-2)
np.testing.assert_allclose(prob_ss, prob_pss, atol=1e-2)
np.testing.assert_allclose(prob_ss, prob_fss, atol=1e-2)
np.testing.assert_allclose(prob_pss, prob_fss, atol=1e-2)
```
