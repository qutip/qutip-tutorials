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

# Quantum Teleportation Circuit

```python
from math import sqrt

from qutip import about, basis, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Measurement
```

## Introduction 

This notebook introduces the basic quantum teleportation circuit (http://www.physics.udel.edu/~msafrono/425-2011/Lecture%2025-1.pdf), complete with measurements and classical control. This notebook also serves as an example on how to add measurement gates and classical controls to a circuit.

We will describe the circuit that enables quantum teleportation. We will use two classical wires and three qubit wires. The first qubit wire represents the quantum state $| q0 ⟩ = | \psi ⟩$ that needs to be transferred from Alice to Bob (so the first qubit is in the possession of Alice). 

```python
teleportation = QubitCircuit(
    3, num_cbits=2, input_states=[r"\psi", "0", "0", "c0", "c1"]
)
```

First, Alice and Bob need to create the shared EPR pair ($\frac{| 00 ⟩ + | 11 ⟩} {2}$) between the second and third qubit by using the hadamard gate on Alice's qubit followed by an entangling CNOT gate.  

```python
teleportation.add_gate("SNOT", targets=[1])
teleportation.add_gate("CNOT", targets=[2], controls=[1])
```

Following this, Alice makes the qubit $| q0 ⟩$ interact with Alice's EPR qubit, followed by measuring on the two qubits belonging to Alice. The measurement results for the first qubit is stored in classical register $c1$ and the second qubit is stored in classical register $c0$.

```python
teleportation.add_gate("CNOT", targets=[1], controls=[0])
teleportation.add_gate("SNOT", targets=[0])

teleportation.add_measurement("M0", targets=[0], classical_store=1)
teleportation.add_measurement("M1", targets=[1], classical_store=0)
```

Now, we apply the $X$ gate on Bob's qubit based on the classical control $c0$ and $Z$ gate based on classical control $c1$. These operations correspond to the following operations based on the state of Alice's measurement. 

$|00⟩ \rightarrow $ no operation \
$|01⟩ \rightarrow Z$ \
$|10⟩ \rightarrow X$ \
$|11⟩ \rightarrow ZX$ 

The final circuit mathematically must result in the third qubit taking the state $|\psi⟩$. 

```python
teleportation.add_gate("X", targets=[2], classical_controls=[0])
teleportation.add_gate("Z", targets=[2], classical_controls=[1])
```

Finally, our teleportation circuit is ready to run, we can view the circuit structure using the following command. 

```python
teleportation.gates
```

The circuit can also be visualized:

```python
teleportation
```

The first qubit is user-specified $|\psi ⟩$ state and the other two must be $|0⟩$. 

### Example 1 
#### $|\psi⟩ = |+ ⟩$  

```python
a = 1 / sqrt(2) * basis(2, 0) + 1 / sqrt(2) * basis(2, 1)
state = tensor(a, basis(2, 0), basis(2, 0))
```

We can confirm our state is initialized correctly by observing the measurment statistics on the first qubit, followed by which we run the circuit.

```python
initial_measurement = Measurement("start", targets=[0])
initial_measurement.measurement_comp_basis(state)
```

We can run the circuit using the `QubitCircuit.run()` function which provided the initial state-vector (or density matrix) initiates one run on the circuit (including sampling any intermediate measurements) and providing the final results (any classical bits can also be explicitly set using the argument `cbits`). The results are returned as a `Result` object. The result states can be accessed through the `get_states()` function where the argument `index=0` specifies the first(only) result should be returned

```python
state_final = teleportation.run(state)
print(state_final)
```

After running, we can see the measurement statistics on the last qubit to see that the qubit is teleported correctly. 

```python
final_measurement = Measurement("start", targets=[2])
final_measurement.measurement_comp_basis(state_final)
```

### Example 2 
#### $|\psi⟩ = |1 ⟩$  

```python
state = tensor(basis(2, 1), basis(2, 0), basis(2, 0))
initial_measurement = Measurement("start", targets=[0])
initial_measurement.measurement_comp_basis(state)
```

```python
state_final = teleportation.run(state)
print(state_final)
```

```python
final_measurement = Measurement("start", targets=[2])
final_measurement.measurement_comp_basis(state_final)
```

Another useful feature of the circuit module is the **QubitCircuit.run_statistics()** feature which provides the opportunity to gather all the possible output states of the circuit along with their output probabilities. Again, the results are returned as a `Result` object. The result states and respective probabilites can be accessed through the `get_results()` function. 

```python
results = teleportation.run_statistics(state)
results.probabilities
```

```python
results.final_states
```

```python
about()
```
