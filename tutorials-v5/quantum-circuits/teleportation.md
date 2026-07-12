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
```

## Introduction 

This notebook introduces the basic quantum teleportation circuit (https://en.wikipedia.org/wiki/Quantum_teleportation), complete with measurements and classical control. This notebook also serves as an example on how to add measurement gates and classical controls to a circuit.

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

The final circuit mathematically must result in the third qubit taking the state $|\psi⟩$. To inspect that output, we append a measurement of the third qubit. It reuses `c0` only after the classical controls have been applied, so it does not affect the teleportation protocol.

```python
teleportation.add_gate("X", targets=[2], classical_controls=[0])
teleportation.add_gate("Z", targets=[2], classical_controls=[1])
teleportation.add_measurement("M2", targets=[2], classical_store=0)
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

To inspect the input without changing the teleportation circuit, we use a separate circuit that measures its first qubit. The result contains the collapsed states and their probabilities.

```python
initial_measurement = QubitCircuit(3, num_cbits=1)
initial_measurement.add_measurement("M", targets=[0], classical_store=0)
initial_results = initial_measurement.run_statistics(state)
initial_results.final_states, initial_results.probabilities
```

We can run the circuit using `QubitCircuit.run()`. It evolves the supplied state vector (or density matrix), samples intermediate measurements, and returns the final state. Classical bits can be set explicitly with the `cbits` argument.

```python
state_final = teleportation.run(state)
print(state_final)
```

The final measurement is part of the teleportation circuit. Running the circuit statistics shows the collapsed output states and their probabilities.

```python
final_results = teleportation.run_statistics(state)
final_results.final_states, final_results.probabilities
```

### Example 2 
#### $|\psi⟩ = |1 ⟩$  

```python
state = tensor(basis(2, 1), basis(2, 0), basis(2, 0))

# The same measurement circuit gives the input statistics for |1>.
initial_results = initial_measurement.run_statistics(state)
initial_results.final_states, initial_results.probabilities
```

```python
state_final = teleportation.run(state)
print(state_final)
```

Another useful feature of the circuit module is **`QubitCircuit.run_statistics()`**, which gathers all possible output states and their probabilities. The returned result exposes these through its `final_states` and `probabilities` attributes.

```python
final_results = teleportation.run_statistics(state)
final_results.final_states, final_results.probabilities
```

```python
about()
```
