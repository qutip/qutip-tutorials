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

# QuTiP example: Quantum Gates and their usage


Author: Anubhav Vardhan (anubhavvardhan@gmail.com)

User-defined gate added by: Boxi Li (etamin1201@gmail.com)

For more information about QuTiP see [http://qutip.org](http://qutip.org)

#### Installation: 
The circuit image visualization requires LaTeX and [ImageMagick](https://imagemagick.org/index.php) for display. The module automatically process the LaTeX code for plotting the circuit, generate the pdf and convert it to the png format.
On Mac and Linux, ImageMagick can be easily installed with the command `conda install imagemagick` if you have conda installed.
Otherwise, please follow the installation instructions on the ImageMagick documentation.

On windows, you need to download and install ImageMagick installer. In addition, you also need [perl](https://www.perl.org/get.html) (for pdfcrop) and [Ghostscript](https://ghostscript.com/releases/index.html) (additional dependency of ImageMagick for png conversion).

To test if the installation is complete, try the following three commands working correctly in Command Prompt: `pdflatex`, `pdfcrop` and `magick anypdf.pdf antpdf.png`, where `anypdf.pdf` is any pdf file you have.

```python
import numpy as np
from numpy import pi
from qutip import Qobj, about
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import (
    expand_operator,
    gate_sequence_product,
    get_controlled_gate,
    get_unitary_gate,
)
from qutip_qip.operations.gates import (
    BERKELEY,
    CNOT,
    CPHASE,
    CSIGN,
    FREDKIN,
    GLOBALPHASE,
    ISWAP,
    MS,
    PHASE,
    R,
    RX,
    RY,
    RZ,
    SNOT,
    SQRTISWAP,
    SQRTNOT,
    SQRTSWAP,
    SWAP,
    SWAPALPHA,
    TOFFOLI,
)
from qutip_qip.transpiler import to_chain_structure

%matplotlib inline
```

## Introduction


http://en.wikipedia.org/wiki/Quantum_gate



## Gates in QuTiP and their representation


### Controlled-PHASE

```python
CPHASE(pi / 2).get_qobj()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate(CSIGN, controls=[0], targets=[1])
q.draw()
```

### Rotation about X-axis

```python
RX(pi / 2).get_qobj()
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate(RX(pi / 2), 0, style={"showarg": True})
q.draw()
```

### Rotation about Y-axis

```python
RY(pi / 2).get_qobj()
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate(RY(pi / 2), 0, style={"showarg": True})
q.draw()
```

### Rotation about Z-axis

```python
RZ(pi / 2).get_qobj()
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate(RZ(pi / 2), 0, style={"showarg": True})
q.draw()
```

### CNOT

```python
CNOT.get_qobj()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate(CNOT, controls=[0], targets=[1])
q.draw()
```

### CSIGN

```python
CSIGN.get_qobj()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate(CSIGN, controls=[0], targets=[1])
q.draw()
```

### Berkeley

```python
BERKELEY.get_qobj()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate(BERKELEY, [0, 1])
q.draw()
```

### SWAPalpha

```python
SWAPALPHA(pi / 2).get_qobj()
```

### FREDKIN

```python
FREDKIN.get_qobj()
```

### TOFFOLI

```python
TOFFOLI.get_qobj()
```

### SWAP

```python
SWAP.get_qobj()
q = QubitCircuit(2, reverse_states=False)
q.add_gate(SWAP, [0, 1])
q.draw()
```

### ISWAP

```python
ISWAP.get_qobj()
q = QubitCircuit(2, reverse_states=False)
q.add_gate(ISWAP, [0, 1])
q.draw()
```

### SQRTiSWAP

```python
SQRTISWAP.get_qobj()
```

### SQRTSWAP

```python
SQRTSWAP.get_qobj()
```

### SQRTNOT

```python
SQRTNOT.get_qobj()
```

### HADAMARD

```python
SNOT.get_qobj()
```

### PHASEGATE

```python
PHASE(pi / 2).get_qobj()
```

### GLOBALPHASE

```python
GLOBALPHASE(pi / 2).get_qobj()
```

### Mølmer–Sørensen gate

```python
MS(pi / 2, 0).get_qobj()
```

### Qubit rotation gate

```python
R(pi / 4, pi / 2).get_qobj()
```

### Expanding gates to larger qubit registers


The example above show how to generate matrix representations of gates in their minimal qubit requirements. To represent a gate in a larger qubit register, use `expand_operator`. For example, to generate a CNOT matrix on a 3-qubit register:

```python
expand_operator(CNOT.get_qobj(), dims=[2, 2, 2], targets=[0, 1])
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate(CNOT, controls=[1], targets=[2])
q.draw()
```

You can also choose different control-target positions with `targets=[control, target]` in the `expand_operator` function:

```python
expand_operator(CNOT.get_qobj(), dims=[2, 2, 2], targets=[2, 0])
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate(CNOT, controls=[0], targets=[2])
q.draw()
```

## Setup of a Qubit Circuit


The gates implemented in QuTiP can be used to build any qubit circuit using the class QubitCircuit. The output can be obtained in the form of a unitary matrix or a latex representation.


In the following example, we take a SWAP gate. It is known that a swap gate is equivalent to three CNOT gates applied in the given format.

```python
N = 2
qc0 = QubitCircuit(N)
qc0.add_gate(ISWAP, [0, 1])
qc0.draw()
```

```python
U_list0 = qc0.propagators()
U0 = gate_sequence_product(U_list0)
U0
```

```python
qc1 = QubitCircuit(N)
qc1.add_gate(CNOT, controls=[0], targets=[1])
qc1.add_gate(CNOT, controls=[1], targets=[0])
qc1.add_gate(CNOT, controls=[0], targets=[1])
qc1.draw()
```

```python
U_list1 = qc1.propagators()
U1 = gate_sequence_product(U_list1)
U1
```

In place of manually converting the SWAP gate to CNOTs, it can be automatically converted using an inbuilt function in QubitCircuit

```python
qc2 = qc0.resolve_gates("CNOT")
qc2.draw()
```

```python
U_list2 = qc2.propagators()
U2 = gate_sequence_product(U_list2)
U2
```

## Example of basis transformation

```python
qc3 = QubitCircuit(3)
qc3.add_gate(CNOT, controls=[1], targets=[0])
qc3.add_gate(RX(pi / 2), 0)
qc3.add_gate(RY(pi / 2), 1)
qc3.add_gate(RZ(pi / 2), 2)
qc3.add_gate(ISWAP, [1, 2])
qc3.draw()
```

```python
U3 = gate_sequence_product(qc3.propagators())
U3
```

### The transformation can either be only in terms of 2-qubit gates:

```python
qc4 = qc3.resolve_gates("CNOT")
qc4.draw()
```

```python
U4 = gate_sequence_product(qc4.propagators())
U4
```

```python
qc5 = qc3.resolve_gates("ISWAP")
qc5.draw()
```

```python
U5 = gate_sequence_product(qc5.propagators())
U5
```

### Or the transformation can be in terms of any 2 single qubit rotation gates along with the 2-qubit gate.

```python
qc6 = qc3.resolve_gates(["ISWAP", "RX", "RY"])
qc6.draw()
```

```python
U6 = gate_sequence_product(qc6.propagators())
U6
```

```python
qc7 = qc3.resolve_gates(["CNOT", "RZ", "RX"])
qc7.draw()
```

```python
U7 = gate_sequence_product(qc7.propagators())
U7
```

## Resolving non-adjacent interactions


Interactions between non-adjacent qubits can be resolved by QubitCircuit to a series of adjacent interactions, which is useful for systems such as spin chain models.

```python
qc8 = QubitCircuit(3)
qc8.add_gate(CNOT, controls=[2], targets=[0])
qc8.draw()
```

```python
U8 = gate_sequence_product(qc8.propagators())
U8
```

```python
qc9 = to_chain_structure(qc8, setup="linear")
qc9.draw()
```

```python
U9 = gate_sequence_product(qc9.propagators())
U9
```

```python
qc10 = qc9.resolve_gates("CNOT")
qc10.draw()
```

```python
U10 = gate_sequence_product(qc10.propagators())
U10
```

`QubitCircuit.gates` has been replaced by `QubitCircuit.instructions`.

```python
qc.instructions
```

## User defined gates
In `qutip-qip`v0.5, define parameterized controlled gates with `get_controlled_gate`, and fixed custom unitaries with `get_unitary_gate`.

```python
def user_gate2():
    # S gate
    mat = np.array([[1.0, 0], [0.0, 1.0j]])
    return Qobj(mat, dims=[[2], [2]])
```

Then create gate classes:

```python
# parameterized controlled-RX gate class
CTRLRX = get_controlled_gate(RX, n_ctrl_qubits=1, gate_name="CTRLRX")
# fixed one-qubit custom gate class
S_USER = get_unitary_gate("S_USER", user_gate2())
```

Use these gate classes in `QubitCircuit.add_gate(...)`:

```python
qc = QubitCircuit(2)
# qubit 0 controls qubit 1
qc.add_gate(CTRLRX(pi / 2), controls=0, targets=1)
# qubit 1 controls qubit 0
qc.add_gate(CTRLRX(pi / 2), controls=1, targets=0)
# one-qubit custom gate
qc.add_gate(S_USER, 1)
props = qc.propagators()
```

```python
props[0]  # qubit 0 controls qubit 1
```

```python
props[1]  # qubit 1 controls qubit 0
```

```python
props[2]  # S  gate acts on qubit 1
```

## Software versions

```python
about()
```
