---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
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
                   propagator_steadystate, sigmax, sigmaz, steadystate_floquet)

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

    n_th = 0.1  # zero temperature

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

    # Calculate steadystate using the propagator
    U = propagator(hamiltonian_t, T, c_op_list, H_args)
    rho_ss = propagator_steadystate(U)

    # Calculate steadystate using floquet formalism
    rho_ss_f = steadystate_floquet(H0, c_op_list, H1, w)

    return output.expect[0], expect(sm.dag() * sm, rho_ss), \
        expect(sm.dag() * sm, rho_ss_f)
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
p_ex, p_ex_ss, p_ex_ss_f = sd_qubit_integrate(delta,
                                              eps0,
                                              A,
                                              w,
                                              gamma1,
                                              gamma2,
                                              psi0,
                                              tlist)
```

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(tlist, np.real(p_ex), label='Mesolve')
ax.plot(tlist, np.real(p_ex_ss) * np.ones(tlist.shape[0]),
        label='Propagator Steadystate')
ax.plot(tlist, np.real(p_ex_ss_f) * np.ones(tlist.shape[0]),
        label='Floquet Steadystate')
ax.set_xlabel("Time")
ax.set_ylabel("P_ex")
ax.set_ylim(0, 1)
ax.set_title("Excitation probabilty of qubit")
ax.legend()
```

```python
assert np.all((p_ex >= 0) & (p_ex <= 1)), "p_ex out of range"
assert np.all((p_ex_ss >= 0) & (p_ex_ss <= 1)), "p_ex_ss out of range"
assert np.all((p_ex_ss_f >= 0) & (p_ex_ss_f <= 1)), "p_ex_ss_f out of range"

# Check that the two steady-state methods give similar results
tolerance = 1e-1
assert np.isclose(p_ex_ss, p_ex_ss_f, atol=tolerance), \
    f"Steady state values differ: {p_ex_ss} vs {p_ex_ss_f}"
```

## Software version:

```python
about()
```
