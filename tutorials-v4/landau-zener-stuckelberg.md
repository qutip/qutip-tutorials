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

# QuTiP example: Landau-Zener-Stuckelberg inteferometry


J.R. Johansson and P.D. Nation

For more information about QuTiP see [http://qutip.org](http://qutip.org)

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (Options, about, destroy, expect, num, propagator,
                   propagator_steadystate, sigmax, sigmaz)
from qutip.ui.progressbar import TextProgressBar as ProgressBar

%matplotlib inline
```

Landau-Zener-Stuckelberg interferometry: Steady state of a strongly driven two-level system, using the one-period propagator. 

```python
# set up the parameters and start calculation
delta = 1.0 * 2 * np.pi  # qubit sigma_x coefficient
w = 2.0 * 2 * np.pi  # driving frequency
T = 2 * np.pi / w  # driving period
gamma1 = 0.00001  # relaxation rate
gamma2 = 0.005  # dephasing  rate

eps_list = np.linspace(-20.0, 20.0, 51) * 2 * np.pi
A_list = np.linspace(0.0, 20.0, 51) * 2 * np.pi

# pre-calculate the necessary operators
sx = sigmax()
sz = sigmaz()
sm = destroy(2)
sn = num(2)

# collapse operators: relaxation and dephasing
c_op_list = [np.sqrt(gamma1) * sm, np.sqrt(gamma2) * sz]
```

```python
# ODE settings (for list-str format)
options = Options()
options.atol = 1e-6  # reduce accuracy to speed
options.rtol = 1e-5  # up the calculation a bit
options.rhs_reuse = True  # Compile Hamiltonian only the first time.
```

```python
# perform the calculation for each combination of eps and A, store the result
# in a matrix
def calculate():

    p_mat = np.zeros((len(eps_list), len(A_list)))

    H0 = -delta / 2.0 * sx

    # Define H1 (first time-dependent term)
    # String method:
    H1 = [-sz / 2, "eps"]
    # Function method:
    # H1 = [- sz / 2, lambda t, args: args['eps'] ]

    # Define H2 (second time-dependent term)
    # String method:
    H2 = [sz / 2, "A * sin(w * t)"]
    # Function method:
    # H2 = [sz / 2, lambda t, args: args['A']*np.sin(args['w'] * t) ]

    H = [H0, H1, H2]

    pbar = ProgressBar(len(eps_list))
    for m, eps in enumerate(eps_list):
        pbar.update(m)
        for n, A in enumerate(A_list):
            args = {"w": w, "A": A, "eps": eps}

            U = propagator(H, T, c_op_list, args, options=options)
            rho_ss = propagator_steadystate(U)

            p_mat[m, n] = np.real(expect(sn, rho_ss))

    return p_mat
```

```python
p_mat = calculate()
```

```python
fig, ax = plt.subplots(figsize=(8, 8))

A_mat, eps_mat = np.meshgrid(A_list / (2 * np.pi), eps_list / (2 * np.pi))

ax.pcolor(eps_mat, A_mat, p_mat, shading="auto")
ax.set_xlabel(r"Bias point $\epsilon$")
ax.set_ylabel(r"Amplitude $A$")
ax.set_title(
    "Steadystate excitation probability\n"
    + r"$H = -\frac{1}{2}\Delta\sigma_x -\frac{1}{2}\epsilon\sigma_z -"
    + r"\frac{1}{2}A\sin(\omega t)$"
    + "\n"
);
```

## Versions

```python
about()
```
