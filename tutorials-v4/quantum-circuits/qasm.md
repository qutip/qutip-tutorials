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

# Imports and Exports QASM circuit

Notebook Author: Sidhant Saraogi(sid1397@gmail.com)


This notebook introduces the [OpenQASM](https://github.com/Qiskit/openqasm) import and export functions. It can also serve as a short introduction to the QASM format. The Quantum Assembly Language(QASM) acts as an intermediate representation for Quantum Circuits. This is one way to export/import from/to with QuTiP. In this way, we can make the QIP module of QuTiP compatible with Qiskit and Cirq.

```python
import numpy as np
from qutip import about, basis, rand_ket, tensor
from qutip_qip.operations import Measurement
from qutip_qip.operations.gates import gate_sequence_product
from qutip_qip.qasm import read_qasm, print_qasm
```

The process is quite simple and only requires the user to store the `.qasm` file in an appropriate location and maintain the absolute path of the file. This will reading the file simpler. For this demonstration, we already saved a few qasm circuit examples in the directory qasm_files. You can find more examples at [OpenQASM repository](https://github.com/Qiskit/openqasm) Let's start off by reading one of the examples: 

```python
path = "../../qasm_files/swap.qasm"
qasm_file = open(path, "r")
print(qasm_file.read())
```

## Qasm Import


This QASM file imitates the SWAP gate native to QuTiP in the QASM format. To import it, we use the `read_qasm` function with the arguments being the file path, the `mode` which defaults to "qiskit" and the `version` which defaults to "2.0".  

We can check that the circuit indeed implements the swap gate by checking the unitary matrix corresponding
to the circuit. This can be done by using the `gate_sequence_product` function and the `propagators` function of the 
`QubitCircuit` class. 

```python
qc = read_qasm(path, mode="qiskit", version="2.0")
gate_sequence_product(qc.propagators())
```

The `mode` refers to the internal way in which QuTiP processes the QASM files. 
With "qiskit" mode, QASM skips the include command for the file qelib1.inc and maps all custom gates defined in it to QuTiP gates without parsing the gate definitions. 

**Note**: "qelib1.inc" is a "header" file that contains some QASM gate definitions. It is available in the OpenQASM repository (as a standard file) and is included with QASM exports by QuTiP (and also by Qiskit/Cirq).

The `version` refers to the version of the OpenQASM standard being processed. The documentation for the same can be found in the [OpenQASM](https://github.com/Qiskit/openqasm) repository. Currently, only OpenQASM 2.0 is supported which is the most popular QASM standard. 


### QASM Export

We can also convert a `QubitCircuit` to the QASM format. This can be particularly useful when we are trying to export quantum circuits to other quantum packages such as Qiskit and Cirq. There are three different ways to output QASM files, `print_qasm`, `str_qasm` and `write_qasm`.  

```python
print_qasm(qc)
```

### Custom Gates

QASM also offers the option to define custom gates in terms of already defined gates using the "gate" keyword. In "qiskit" mode, our QASM interpreter can be assumed to already allow for all the gates defined in the file [qelib1.inc](https://github.com/Qiskit/openqasm/blob/master/examples/generic/qelib1.inc) provided by the OpenQASM repository.

In the file `swap_custom.qasm`, we define the `swap` gate in terms of the pre-defined `cx` gates.

```python
path = "../../qasm_files/swap_custom.qasm"
qasm_file = open(path, "r")
print(qasm_file.read())
```

Furthermore, the circuit also measures the two qubits q[0] and q[1] and stores the results in the classical registers c[0] and c[1]

```python
qc = read_qasm(path)
```

We can now run the circuit to confirm that the circuit is correctly loaded and performs the correct operations. To do this, we can use the `QubitCircuit.run` function with the appropriate input state. In our case, we can take the state `|01⟩`. 

```python
qc.run(tensor(basis(2, 0), basis(2, 1)))
```

As predicted the output is the state after swapping which is `|10⟩`


### Measurements and Classical Control

The QASM format also allows for other circuit features such as measurement and control of gates by classical bits. 
This is also supported by QuTiP. For an example, we can refer to the example of quantum teleportation. A more complete explanation of teleportation can be found in the [notebook](teleportation.ipynb) on quantum teleportation.

```python
path = "../../qasm_files/teleportation.qasm"
qasm_file = open(path, "r")
qasm_str = qasm_file.read()
print(qasm_str)
```

 We can also read in a QASM file from a string by specifying `strmode=True` to `read_qasm`

```python
teleportation = read_qasm(qasm_str, strmode=True)
```

**Note**: 
The above warning is expected to inform the user that the import from QASM to QuTiP does not retain any information about the different qubit/classical bit register names. This could potentially be an issue when the circuit is exported if the user wants to maintain the consistency. 


We can quickly check that the teleportation circuit works properly by teleporting the first qubit into the third qubit. 

```python
state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))

initial_measurement = Measurement("start", targets=[0])
_, initial_probabilities = initial_measurement.measurement_comp_basis(state)

state_final = teleportation.run(state)

final_measurement = Measurement("start", targets=[2])
_, final_probabilities = final_measurement.measurement_comp_basis(state_final)

np.testing.assert_allclose(initial_probabilities, final_probabilities)
```

**Note**: Custom gates imported in the QASM format cannot be easily exported. Currently, only gates that are defined native to QuTiP can be exported. QuTiP also produces custom gate definitions for gates not provided in the `qelib1.inc` "header" file. In these cases, QuTiP will add it's own gate definitions directly to the the exported `.qasm` file but this is restricted only to already gates native to QuTiP. 
Export from QuTiP handles both gates and measurements. However, it does not allow for export of controlled gates. 

```python
about()
```
