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

# Quantum Process Tomography


J.R. Johansson and P.D. Nation

For more information about QuTiP see [http://qutip.org](http://qutip.org)

```python
import numpy as np
from qutip import (about, qeye, qpt, qpt_plot_combined, sigmax, sigmay, sigmaz,
                   spost, spre)
from qutip_qip.operations import (cnot, fredkin, iswap, phasegate, snot,
                                  sqrtiswap, swap, toffoli)

%matplotlib inline
```

```python
"""
Plot the process tomography matrices for some 1, 2, and 3-qubit qubit gates.
"""
gates = [
    ["C-NOT", cnot()],
    ["SWAP", swap()],
    ["$i$SWAP", iswap()],
    [r"$\sqrt{i\mathrm{SWAP}}$", sqrtiswap()],
    ["S-NOT", snot()],
    [r"$\pi/2$ phase gate", phasegate(np.pi / 2)],
    ["Toffoli", toffoli()],
    ["Fredkin", fredkin()],
]
```

```python
def plt_qpt_gate(gate, figsize=(8, 6)):

    name = gate[0]
    U_psi = gate[1]

    N = len(U_psi.dims[0])  # number of qubits

    # create a superoperator for the density matrix
    # transformation rho = U_psi * rho_0 * U_psi.dag()
    U_rho = spre(U_psi) * spost(U_psi.dag())

    # operator basis for the process tomography
    op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()] for i in range(N)]

    # labels for operator basis
    op_label = [["$i$", "$x$", "$y$", "$z$"] for i in range(N)]

    # calculate the chi matrix
    chi = qpt(U_rho, op_basis)

    # visualize the chi matrix
    fig, ax = qpt_plot_combined(chi, op_label, name, figsize=figsize)

    ax.set_title(name)

    return fig, ax
```

```python
plt_qpt_gate(gates[0]);
```

```python
plt_qpt_gate(gates[1]);
```

```python
plt_qpt_gate(gates[2]);
```

```python
plt_qpt_gate(gates[3]);
```

```python
plt_qpt_gate(gates[4]);
```

```python
plt_qpt_gate(gates[5]);
```

```python
fig, ax = plt_qpt_gate(gates[6], figsize=(16, 12))
ax.axis("tight");
```

```python
fig, ax = plt_qpt_gate(gates[7], figsize=(16, 12))
ax.axis("tight");
```

## Versions

```python
about()
```
