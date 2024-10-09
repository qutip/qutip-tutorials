---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: qutip-dev
    language: python
    name: python3
---

# Plotting and Customizing Quantum Circuits using TextRenderer
Author: Rushiraj Gadhvi (gadhvirushiraj@gmail.com)

This notebook serves as a comprehensive guide to plotting quantum circuits using QuTiP-QIP's TextRenderer. It explores the various customization options available to users for creating and modifying plots.

- **Circuit-Level Customization**
- **Gate-Level Customization**

```python
import qutip
import numpy as np
from qutip_qip.circuit import QubitCircuit
```

```python
qc = QubitCircuit(2, num_cbits=1)
qc.add_gate("CNOT", controls=0, targets=1)
qc.add_gate("SNOT", targets=1)
qc.add_gate("ISWAP", targets=[0, 1])
qc.add_measurement("M0", targets=1, classical_store=0)
```

```python
qc.draw("text")
```

---
### Circuit Level Customization Options
---


##### Custom Wire Labels

```python
qc.draw("text", wire_label=["some_name", "some_long_name", "long_long_name"])
```

##### Control extra wire extension at end of circuit

```python
qc.draw("text", end_wire_ext=0)
```

```python
qc.draw("text", end_wire_ext=5)
```

##### Adjust Gate Padding

```python
qc.draw("text", gate_pad=3)
```

---
### Gate Level Customization Options
---

```python
qc = QubitCircuit(2)
qc.add_gate("H", targets=[0])
qc.add_gate("H", targets=[1], arg_label="hadamard gate")
```

```python
qc.draw("text")
```

#### With User Custom Gates

```python
from qutip import Qobj
from qutip_qip.operations import rx
```

```python
def user_gate1(arg_value):
    # controlled rotation X
    mat = np.zeros((4, 4), dtype=np.complex)
    mat[0, 0] = mat[1, 1] = 1.0
    mat[2:4, 2:4] = rx(arg_value).full()
    return Qobj(mat, dims=[[2, 2], [2, 2]])


def user_gate2():
    # S gate
    mat = np.array([[1.0, 0], [0.0, 1.0j]])
    return Qobj(mat, dims=[[2], [2]])
```

```python
qc = QubitCircuit(3)
qc.user_gates = {"CTRLRX": user_gate1, "S": user_gate2}

# qubit 1 controls qubit 0
qc.add_gate("CTRLRX", targets=[1, 0], arg_value=np.pi / 2)
# qubit 2 is target of S gate
qc.add_gate("S", targets=[2])
```

```python
qc.draw("text")
```

---

```python
qutip.about()
```
