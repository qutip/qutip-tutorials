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

# Steady-State: Time-dependent (periodic) quantum system


J.R. Johansson and P.D. Nation

For more information about QuTiP see [http://qutip.org](http://qutip.org)


Find the steady state of a driven qubit, by finding the eigenstates of the propagator for one driving period

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, destroy, expect, mesolve, propagator,
                   propagator_steadystate, sigmax, sigmaz)

%matplotlib inline
```

```python
def hamiltonian_t(t, args):
    #
    # evaluate the hamiltonian at time t.
    #
    H0 = args["H0"]
    H1 = args["H1"]
    w = args["w"]

    return H0 + H1 * np.sin(w * t)
```

```python
def sd_qubit_integrate(delta, eps0, A, w, gamma1, gamma2, psi0, tlist):

    # Hamiltonian
    sx = sigmax()
    sz = sigmaz()
    sm = destroy(2)

    H0 = -delta / 2.0 * sx - eps0 / 2.0 * sz
    H1 = -A * sx

    H_args = {"H0": H0, "H1": H1, "w": w}
    # collapse operators
    c_op_list = []

    n_th = 0.5  # zero temperature

    # relaxation
    rate = gamma1 * (1 + n_th)
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm)

    # excitation
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm.dag())

    # dephasing
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sz)

    # evolve and calculate expectation values
    output = mesolve(hamiltonian_t, psi0, tlist,
                     c_op_list, [sm.dag() * sm], H_args)

    T = 2 * np.pi / w

    U = propagator(hamiltonian_t, T, c_op_list, H_args)

    rho_ss = propagator_steadystate(U)

    return output.expect[0], expect(sm.dag() * sm, rho_ss)
```

```python
delta = 0.3 * 2 * np.pi  # qubit sigma_x coefficient
eps0 = 1.0 * 2 * np.pi  # qubit sigma_z coefficient
A = 0.05 * 2 * np.pi  # driving amplitude (sigma_x coupled)
w = 1.0 * 2 * np.pi  # driving frequency

gamma1 = 0.15  # relaxation rate
gamma2 = 0.05  # dephasing  rate

# intial state
psi0 = basis(2, 0)
tlist = np.linspace(0, 50, 500)
```

```python
p_ex, p_ex_ss = sd_qubit_integrate(delta, eps0, A, w,
                                   gamma1, gamma2, psi0, tlist)
```

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(tlist, np.real(p_ex))
ax.plot(tlist, np.real(p_ex_ss) * np.ones(tlist.shape[0]))
ax.set_xlabel("Time")
ax.set_ylabel("P_ex")
ax.set_ylim(0, 1)
ax.set_title("Excitation probabilty of qubit");
```

## Software version:

```python
about()
```
