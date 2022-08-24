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
from qutip_qip.operations import (Gate, berkeley, cnot, cphase, csign, fredkin,
                                  gate_sequence_product, globalphase, iswap,
                                  molmer_sorensen, phasegate, qrot, rx, ry, rz,
                                  snot, sqrtiswap, sqrtnot, sqrtswap, swap,
                                  swapalpha, toffoli)

%matplotlib inline
```

## Introduction


http://en.wikipedia.org/wiki/Quantum_gate



## Gates in QuTiP and their representation


### Controlled-PHASE

```python
cphase(pi / 2)
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("CSIGN", controls=[0], targets=[1])
q.png
```

### Rotation about X-axis

```python
rx(pi / 2)
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate("RX", targets=[0], arg_value=pi / 2, arg_label=r"\frac{\pi}{2}")
q.png
```

### Rotation about Y-axis

```python
ry(pi / 2)
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate("RY", targets=[0], arg_value=pi / 2, arg_label=r"\frac{\pi}{2}")
q.png
```

### Rotation about Z-axis

```python
rz(pi / 2)
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate("RZ", targets=[0], arg_value=pi / 2, arg_label=r"\frac{\pi}{2}")
q.png
```

### CNOT

```python
cnot()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("CNOT", controls=[0], targets=[1])
q.png
```

### CSIGN

```python
csign()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("CSIGN", controls=[0], targets=[1])
q.png
```

### Berkeley

```python
berkeley()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("BERKELEY", targets=[0, 1])
q.png
```

### SWAPalpha

```python
swapalpha(pi / 2)
```

### FREDKIN

```python
fredkin()
```

### TOFFOLI

```python
toffoli()
```

### SWAP

```python
swap()
q = QubitCircuit(2, reverse_states=False)
q.add_gate("SWAP", targets=[0, 1])
q.png
```

### ISWAP

```python
iswap()
q = QubitCircuit(2, reverse_states=False)
q.add_gate("ISWAP", targets=[0, 1])
q.png
```

### SQRTiSWAP

```python
sqrtiswap()
```

### SQRTSWAP

```python
sqrtswap()
```

### SQRTNOT

```python
sqrtnot()
```

### HADAMARD

```python
snot()
```

### PHASEGATE

```python
phasegate(pi / 2)
```

### GLOBALPHASE

```python
globalphase(pi / 2)
```

### Mølmer–Sørensen gate

```python
molmer_sorensen(pi / 2)
```

### Qubit rotation gate

```python
qrot(pi / 2, pi / 4)
```

### Expanding gates to larger qubit registers


The example above show how to generate matrice representations of the gates implemented in QuTiP, in their minimal qubit requirements. If the same gates is to be represented in a qubit register of size $N$, the optional keywork argument `N` can be specified when calling the gate function. For example, to generate the matrix for the CNOT gate for a $N=3$ bit register:

```python
cnot(N=3)
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate("CNOT", controls=[1], targets=[2])
q.png
```

Furthermore, the control and target qubits (when applicable) can also be similarly specified using keyword arguments `control` and `target` (or in some cases `controls` or `targets`):

```python
cnot(N=3, control=2, target=0)
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate("CNOT", controls=[0], targets=[2])
q.png
```

## Setup of a Qubit Circuit


The gates implemented in QuTiP can be used to build any qubit circuit using the class QubitCircuit. The output can be obtained in the form of a unitary matrix or a latex representation.


In the following example, we take a SWAP gate. It is known that a swap gate is equivalent to three CNOT gates applied in the given format.

```python
N = 2
qc0 = QubitCircuit(N)
qc0.add_gate("ISWAP", [0, 1], None)
qc0.png
```

```python
U_list0 = qc0.propagators()
U0 = gate_sequence_product(U_list0)
U0
```

```python
qc1 = QubitCircuit(N)
qc1.add_gate("CNOT", 0, 1)
qc1.add_gate("CNOT", 1, 0)
qc1.add_gate("CNOT", 0, 1)
qc1.png
```

```python
U_list1 = qc1.propagators()
U1 = gate_sequence_product(U_list1)
U1
```

In place of manually converting the SWAP gate to CNOTs, it can be automatically converted using an inbuilt function in QubitCircuit

```python
qc2 = qc0.resolve_gates("CNOT")
qc2.png
```

```python
U_list2 = qc2.propagators()
U2 = gate_sequence_product(U_list2)
U2
```

From QuTiP 4.4, we can also add gate at arbitrary position in a circuit.

```python
qc1.add_gate("CSIGN", index=[1], targets=[0], controls=[1])
qc1.png
```

## Example of basis transformation

```python
qc3 = QubitCircuit(3)
qc3.add_gate("CNOT", 1, 0)
qc3.add_gate("RX", 0, None, pi / 2, r"\pi/2")
qc3.add_gate("RY", 1, None, pi / 2, r"\pi/2")
qc3.add_gate("RZ", 2, None, pi / 2, r"\pi/2")
qc3.add_gate("ISWAP", [1, 2])
qc3.png
```

```python
U3 = gate_sequence_product(qc3.propagators())
U3
```

### The transformation can either be only in terms of 2-qubit gates:

```python
qc4 = qc3.resolve_gates("CNOT")
qc4.png
```

```python
U4 = gate_sequence_product(qc4.propagators())
U4
```

```python
qc5 = qc3.resolve_gates("ISWAP")
qc5.png
```

```python
U5 = gate_sequence_product(qc5.propagators())
U5
```

### Or the transformation can be in terms of any 2 single qubit rotation gates along with the 2-qubit gate.

```python
qc6 = qc3.resolve_gates(["ISWAP", "RX", "RY"])
qc6.png
```

```python
U6 = gate_sequence_product(qc6.propagators())
U6
```

```python
qc7 = qc3.resolve_gates(["CNOT", "RZ", "RX"])
qc7.png
```

```python
U7 = gate_sequence_product(qc7.propagators())
U7
```

## Resolving non-adjacent interactions


Interactions between non-adjacent qubits can be resolved by QubitCircuit to a series of adjacent interactions, which is useful for systems such as spin chain models.

```python
qc8 = QubitCircuit(3)
qc8.add_gate("CNOT", 2, 0)
qc8.png
```

```python
U8 = gate_sequence_product(qc8.propagators())
U8
```

```python
qc9 = qc8.adjacent_gates()
qc9.gates
```

```python
U9 = gate_sequence_product(qc9.propagators())
U9
```

```python
qc10 = qc9.resolve_gates("CNOT")
qc10.png
```

```python
U10 = gate_sequence_product(qc10.propagators())
U10
```

## Adding gate in the middle of a circuit
From QuTiP 4.4 one can add a gate at an arbitrary position of a circuit. All one needs to do is to specify the parameter index. With this, we can also add the same gate at multiple positions at the same time.

```python
qc = QubitCircuit(1)
qc.add_gate("RX", targets=1, arg_value=np.pi / 2)
qc.add_gate("RX", targets=1, arg_value=np.pi / 2)
qc.add_gate("RY", targets=1, arg_value=np.pi / 2, index=[0])
qc.gates
```

## User defined gates
From QuTiP 4.4 on, user defined gates can be defined by a python function that takes at most one parameter and return a `Qobj`, the dimension of the `Qobj` has to match the qubit system.

```python
def user_gate1(arg_value):
    # controlled rotation X
    mat = np.zeros((4, 4), dtype=np.complex)
    mat[0, 0] = mat[1, 1] = 1.0
    mat[2:4, 2:4] = rx(arg_value)
    return Qobj(mat, dims=[[2, 2], [2, 2]])


def user_gate2():
    # S gate
    mat = np.array([[1.0, 0], [0.0, 1.0j]])
    return Qobj(mat, dims=[[2], [2]])
```

To let the `QubitCircuit` process those gates, we need to modify its attribute `QubitCircuit.user_gates`, which is a python dictionary in the form `{name: gate_function}`.

```python
qc = QubitCircuit(2)
qc.user_gates = {"CTRLRX": user_gate1, "S": user_gate2}
```

When calling the `add_gate` method, the target qubits and the argument need to be given.

```python
# qubit 0 controls qubit 1
qc.add_gate("CTRLRX", targets=[0, 1], arg_value=pi / 2)
# qubit 1 controls qubit 0
qc.add_gate("CTRLRX", targets=[1, 0], arg_value=pi / 2)
# a gate can also be added using the Gate class
g_T = Gate("S", targets=[1])
qc.add_gate("S", targets=[1])
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
