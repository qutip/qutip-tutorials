---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: qutip-tutorials-v5
    language: python
    name: python3
---

# QuTiPv5 Paper - Quantum Circuits with QIP

Authors: Maximilian Meyer-Mölleringhof (m.meyermoelleringhof@gmail.com), Neill Lambert (nwlambert@gmail.com)

Quantum circuits serve as a standard framework for representing and manipulating quantum algorithms visually and conceptually.
As a member of the QuTiP family, the QuTiP-QIP package add this framework and enables several distinctive capabilities.
It allows seamless incorporation of circuit-representing unitaries into QuTiP's ecosystem using the `Qobj` class.
Moreover, it links QuTiP-QOC and the open-system solvers, enabling pulse-level simulations of circuits with realistic noise effects.

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, destroy, expect, ket2dm, mesolve, qeye,
                   sesolve, sigmax, sigmay, sigmaz, tensor)
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits

%matplotlib inline
```

## Introduction

In this example, we show how to

- Construct a simple quantum circuit which simulates the dynamics of a two-qubit Hamiltonian
- Simulate the dynamics of an open system, using ancillas to induce noise
- Run both simulations on hardware backend (i.e., processor) that simulates itself the intrinsic noisy dynamics

As for the Hamiltonian, we consider two interacting qubits:

$H = \dfrac{\epsilon_1}{2} \sigma_z^{(1)} + \dfrac{\epsilon_2}{2} \sigma_z^{(2)} + g \sigma_{x}^{(1)} \sigma_{x}^{(2)}$.

The Pauli matrices $\sigma_z^{(1/2)}$ describe the respective two-level system of the qubits.
The qubits are coupled via $\sigma_{x}^{(1/2)}$ and their interaction strength is given by $g$.

```python
# Parameters
epsilon1 = 0.7
epsilon2 = 1.0
g = 0.3

sx1 = sigmax() & qeye(2)
sx2 = qeye(2) & sigmax()

sy1 = sigmay() & qeye(2)
sy2 = qeye(2) & sigmay()

sz1 = sigmaz() & qeye(2)
sz2 = qeye(2) & sigmaz()

H = 0.5 * epsilon1 * sz1 + 0.5 * epsilon2 * sz2 + g * sx1 * sx2

init_state = basis(2, 0) & basis(2, 1)
```

## Circuit Visualization

Before jumping into the example, we want to quickly look at how the circuits we build can be visualized.
In QuTiP-QIP, quantum circuits can be drawn in three different ways: as text, as matplotlib plot and in latex format.

```python
qc = QubitCircuit(2, num_cbits=1)
qc.add_gate("H", 0)
qc.add_gate("H", 1)
qc.add_gate("CNOT", 1, 0)
qc.add_measurement("M", targets=[0], classical_store=0)
```

After this, we can draw the circuit by defining the various renderers.

```python
qc.draw("text")
qc.draw("matplotlib")
qc.draw("latex")
```

## Simulating Hamiltonian Dynamics

A very common method in quantum simulations is to reduce the propagation of the Schrödinger equation into several short time steps in order to finally arrive at the desired solution.
The propagator for one-time such step is well approximated by using *Trotterization*:

$\psi(t_f) = e^{-i (H_A + H_B) t_f} \psi(0) \approx [e^{-i H_A dt} e^{-i H_B dt}]^d \psi(0)$,

given that the time steps $dt = t_f / d$ is sufficiently small.
The idea then is that the Hamiltonians $H_A$ and $H_B$ are chosen such that $e^{-i H_{A,B} dt}$ can be mapped to quantum gates.

For our example, we express

$H_A = \dfrac{\epsilon_1}{2} \sigma_z^{(1)} + \dfrac{\epsilon_2}{2} \sigma_z^{(2)}$ and

$H_B = g \sigma_x^{(1)} \sigma_x^{(2)}$.

We can then construct the circuit with two qubits and a set of gates:

$A_1 = e^{-i \epsilon_1 \sigma_z^{(1)} dt / 2}$,

$A_2 = e^{-i \epsilon_2 \sigma_z^{(2)} dt / 2}$ and

$B = e^{-i g \sigma_x^{(1)} \sigma_x^{(2)} dt}$.

We apply them to an initial state and then repeat them $d$ times.

Depending on the hardware, the type of available physical gates changes.
Since we will be using a superconducting qubit hardware backend, we will express these gates in terms of RZ (rotation around Z axis), RZX (combined rotation around XZ) and Hadamard gates.
In general, QuTiP-QIP supports a great variety of gates and also the option for custom gates exists.
More information on this is presented in the original paper for QuTiP-QIP [\[1\]](#References).

```python
# simulation parameters
tf = 20.0  # total time
dt = 2.0  # Trotter step size
num_steps = int(tf / dt)
times_circ = np.arange(0, tf + dt, dt)
```

```python
# initialization of two qubit circuit
trotter_simulation = QubitCircuit(2)

# gates for trotterization with small timesteps dt
trotter_simulation.add_gate("RZ", targets=[0], arg_value=(epsilon1 * dt))
trotter_simulation.add_gate("RZ", targets=[1], arg_value=(epsilon2 * dt))

trotter_simulation.add_gate("H", targets=[0])
trotter_simulation.add_gate("RZX", targets=[0, 1], arg_value=g * dt * 2)
trotter_simulation.add_gate("H", targets=[0])
trotter_simulation.compute_unitary()
```

```python
trotter_simulation.draw("matplotlib")
```

```python
# Evaluate multiple iteration of a circuit
result_circ = init_state
state_trotter_circ = [init_state]

for dd in range(num_steps):
    result_circ = trotter_simulation.run(state=result_circ)
    state_trotter_circ.append(result_circ)
```

### Noisy Hardware

We can load our quantum circuit into a hardware backed to simulate the execution on various types of hardware.
In our case, we are interested in a superconducting circuit:

```python
processor = SCQubits(num_qubits=2, t1=2e5, t2=2e5)
processor.load_circuit(trotter_simulation)
# Since SCQubit is modelled as a qutrit, we need three-level systems here
init_state_trit = tensor(basis(3, 0), basis(3, 1))
```

Now we can run the simulation in a similar fashion as we did before.
In this case however, the results is a `results` object from the QuTiP solver being used.
For our example, `mesolve()` is used as we specified finite $T_1$ and $T_2$ upon initialization.
The processor itself defines an internal Hamiltonian as well as available control operations, pulse shapes, $T_1$ and $T_2$, etc..

```python
state_proc = init_state_trit
state_list_proc = [init_state_trit]

for dd in range(num_steps):
    result = processor.run_state(state_proc)
    state_proc = result.final_state
    state_list_proc.append(result.final_state)
```

We can see the pulse shapes used in the solver:

```python
processor.plot_pulses()
```

### Comparison of Results

To evaluate the quantum simulation, we compare the final results with the standard `sesolve` method.

```python
# Exact Schrodinger equation
tlist = np.linspace(0, tf, 100)
states_sesolve = sesolve(H, init_state, tlist).states
```

```python
sz_qutrit = basis(3, 0) * basis(3, 0).dag() - basis(3, 1) * basis(3, 1).dag()

expec_sesolve = expect(sz1, states_sesolve)
expec_trotter = expect(sz1, state_trotter_circ)
expec_supcond = expect(sz_qutrit & qeye(3), state_list_proc)

plt.plot(tlist, expec_sesolve, "-", label="Ideal")
plt.plot(times_circ, expec_trotter, "--d", label="Trotter circuit")
plt.plot(times_circ, expec_supcond, "-.o", label="noisy hardware")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

## Lindblad Simulation

To simulate the Lindblad master equation, we consider two recent proposals [\[2, 3\]](#References) where a sequence of unitaries is used to approximate Lindblad dynamics to arbitrary order.
This is realized by employing a ancilla qubits, and measurements / resets, for the Lindblad collapse operators.
The initial state is a dilated state $\ket{\psi_D(t=0)} = \ket{\psi(t = 0)} \otimes \ket{0}^{\otimes K}$ where $K$ referes to the number of ancillas.
Every time step, the system and the ancillas interact for a time span $\sqrt{dt}$.
Thereafter, the ancillas are reset to their ground state.

We can find the Trotter approximation for the unitary describing the interaction between system $i$ and its associated ancilla for collapse operator $k$:

$U(\sqrt{dt}) \approx e^{-\frac{1}{2} i \sigma_{x}^{(i)}\sigma_{x}^{(k)} \sqrt{\gamma_k dt}} \cdot e^{\frac{1}{2} i \sigma_{y}^{(i)}\sigma_{y}^{(k)}\sqrt{\gamma_k dt}}$,

with the dissipation rate $\gamma_k$.
Like in the example above, we can decompose this unitary into Hadamard, `RZ`, `RX` and `RZX` gates which are the native gates for the superconducting processor hardware in QuTiP.

```python
gam = 0.03  # dissipation rate
```

```python
trotter_simulation_noisy = QubitCircuit(4)

sqrt_term = np.sqrt(gam * dt)

# Coherent dynamics
trotter_simulation_noisy.add_gate("RZ", targets=[1], arg_value=epsilon1 * dt)
trotter_simulation_noisy.add_gate("RZ", targets=[2], arg_value=epsilon2 * dt)

trotter_simulation_noisy.add_gate("H", targets=[1])
trotter_simulation_noisy.add_gate("RZX", targets=[1, 2], arg_value=g * dt * 2)
trotter_simulation_noisy.add_gate("H", targets=[1])

# Decoherence
# exp(-i XX t)
trotter_simulation_noisy.add_gate("H", targets=[0])
trotter_simulation_noisy.add_gate("RZX", targets=[0, 1], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("H", targets=[0])

# exp(-i YY t)
trotter_simulation_noisy.add_gate("RZ", 1, arg_value=np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 0, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RZX", [0, 1], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("RZ", 1, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 0, arg_value=np.pi / 2)

# exp(-i XX t)
trotter_simulation_noisy.add_gate("H", targets=[2])
trotter_simulation_noisy.add_gate("RZX", targets=[2, 3], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("H", targets=[2])

# exp(-i YY t)
trotter_simulation_noisy.add_gate("RZ", 3, arg_value=np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 2, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RZX", [2, 3], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("RZ", 3, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 2, arg_value=np.pi / 2)

trotter_simulation_noisy.draw("matplotlib")
```

```python
state_system = ket2dm(init_state)
state_trotter_circ = [init_state]
ancilla = basis(2, 1) * basis(2, 1).dag()
for dd in range(num_steps):
    state_full = tensor(ancilla, state_system, ancilla)
    state_full = trotter_simulation_noisy.run(state=state_full)
    state_system = state_full.ptrace([1, 2])
    state_trotter_circ.append(state_system)
```

Again, we want to run this trotterized evolution on the suuperconducting hardware backend. Be aware that, due of the increased complexity, this computation can take several minutes depending on your hardware.

```python
processor = SCQubits(num_qubits=4, t1=3.0e4, t2=3.0e4)
processor.load_circuit(trotter_simulation_noisy)
```

```python
state_system = ket2dm(init_state_trit)
state_list_proc = [state_system]
for dd in range(num_steps):
    state_full = tensor(
        basis(3, 1) * basis(3, 1).dag(),
        state_system,
        basis(3, 1) * basis(3, 1).dag(),
    )
    result_noisey = processor.run_state(
        state_full,
        solver="mesolve",
        options={
            "store_states": False,
            "store_final_state": True,
        },
    )
    state_full = result_noisey.final_state
    state_system = state_full.ptrace([1, 2])
    state_list_proc.append(state_system)
    print(f"Step {dd+1}/{num_steps} finished.")
```

```python
processor.plot_pulses()
```

### Comparison of Results

```python
# Standard mesolve solution
sm1 = tensor(destroy(2).dag(), qeye(2))
sm2 = tensor(qeye(2), destroy(2).dag())
c_ops = [np.sqrt(gam) * sm1, np.sqrt(gam) * sm2]

result_me = mesolve(H, init_state, tlist, c_ops, e_ops=[sz1, sz2])
```

```python
plt.plot(tlist, result_me.expect[0], "-", label=r"Ideal")
plt.plot(times_circ, expect(sz1, state_trotter_circ), "--d", label="trotter")
plt.plot(
    times_circ,
    expect(sz_qutrit & qeye(3), state_list_proc),
    "-.o",
    label=r"noisy hardware",
)
plt.xlabel("Time")
plt.ylabel("Expectation values")
plt.legend()
plt.show()
```

## References

\[1\] [Li, et. al, Quantum (2022)](http://dx.doi.org/10.22331/q-2022-01-24-630)

\[2\] [Ding, et. al, PRX Quantum (2024)](https://doi.org/10.1103/PRXQuantum.5.020332)

\[3\] [Cleve and Lang, ICALP (2017)](https://doi.org/10.48550/arXiv.1612.09512)


## About

```python
about()
```

## Testing

```python
# TODO
```
