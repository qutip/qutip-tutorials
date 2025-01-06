---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Quantum Optimal Control with Reinforcement Learning

In this notebook, we will demonstrate how to use the `_RL` module to solve a quantum optimal control problem using reinforcement learning (RL). We will define a simple state transfer problem with a single qubit, where the goal is to transfer a quantum system from one state to another, and we will use the RL agent to optimize the control pulses to achieve this task.
After we will also see the same problem but using unitary operators


## State to State Transfer




### Setup and Import Required Libraries

```python
# If you are running this in an environment where some packages are missing,
# use this cell to install them:
# !pip install qutip stable-baselines3 gymnasium

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
```

### Define the Quantum Control Problem


We define the problem of transferring a quantum system from the initial state |0⟩ to the target state |+⟩. The system is controlled via three control Hamiltonians corresponding to the Pauli matrices, and a drift Hamiltonian for natural evolution of the qubit.

```python
# Define the initial and target states
initial_state = qt.basis(2, 0)  # |0⟩
target_state = (qt.basis(2, 0) + qt.basis(2, 1)).unit()  # |+⟩
# target_state = qt.basis(2, 1)   # |1⟩

# Define the control Hamiltonians (Pauli matrices)
H_c = [qt.sigmax(), qt.sigmay()]  # , qt.sigmaz()]

# Define the drift Hamiltonian
w, d = 0.1, 1.0
H_d = 1 / 2 * (w * qt.sigmaz() + d * qt.sigmax())

# Combine the Hamiltonians into a single list
H = [H_d] + H_c

# Define the objective
objectives = [Objective(initial=initial_state, H=H, target=target_state)]

# Define the control parameters with bounds
control_parameters = {
    "p": {"bounds": [(-1, +1)]},
}

# Define the time interval
tlist = np.linspace(0, 2 * np.pi, 100)

# Define algorithm-specific settings
algorithm_kwargs = {
    "fid_err_targ": 0.01,
    "alg": "RL",
    "max_iter": 23000,
    "shorter_pulses": False,
}
optimizer_kwargs = {}
```

Note that `max_iter` defines the number of episodes the algorithm can execute, the 100 in `tlist` defines the maximum number of steps per episode.  
If `shorter_pulses` is True, the training will be longer as the algorithm will try to optimize the episodes using as few steps as possible in addition to checking if the target infidelity is reached.
If False, the algorithm takes less time and stops as soon as it finds an episode with infidelity <= target infidelity.


### Initialize and Train the RL Environment


Now we will call the `optimize_pulses()` method, passing it the control problem we defined.
The method will create an instance of the `_RL` class, which will set up the reinforcement learning environment and start training.
Finally it returns the optimization results through an object of the `Result` class.

```python
# Initialize the RL environment and start training
rl_result = optimize_pulses(
    objectives, control_parameters, tlist, algorithm_kwargs, optimizer_kwargs
)
```

### Analyze the Results


After the training is complete, we can analyze the results obtained by the RL agent. 
In the above window showing the output produced by Gymansium, you can observe how during training the number of steps per episode (ep_len_mean) decreases and the average reward of the episodes (ep_rew_mean) increases.


We can now see the fields of the `Result` class, this includes the final infidelity, the optimized control parameters and more.

```python
print(rl_result)
```

```python
# We can also visualize the initial and final states on the Bloch sphere
bloch_sp = qt.Bloch()
bloch_sp.add_states(initial_state)  # green
bloch_sp.add_states(target_state)  # orange
bloch_sp.add_states(rl_result._final_states[0])  # blue
bloch_sp.show()
```

## Unitary Operators

Now we will show how to tackle a problem similar to the previous one, but this time, instead of reaching a specific target state, the goal is to start from the identity operator and evolve it in a controlled way until we obtain a specific unitary operator, such as the Hadamard gate.


The control problem is similar to the previous one, we just need to change the initial state, the target state (now they are matrices) and update the objective.  
We can also change the number of episodes for this task by changing `max_iter`  
By setting `shorter_pulses` to False, the algorithm will stop as soon as it finds an episode that satisfies the target infidelity.

```python
initial = qt.qeye(2)  # Identity
target = qt.gates.hadamard_transform()

# Define the control Hamiltonians (Pauli matrices)
H_c = [qt.sigmax(), qt.sigmay()]  # , qt.sigmaz()]

# Define the drift Hamiltonian
w, d = 0.1, 1.0
H_d = 1 / 2 * (w * qt.sigmaz() + d * qt.sigmax())

# Combine the Hamiltonians into a single list
H = [H_d] + H_c

objectives = [Objective(initial, H, target)]

# Define the control parameters with bounds
control_parameters = {
    "p": {"bounds": [(-1, +1)]},  # -15, -11
}

# Define the time interval
tlist = np.linspace(0, 2 * np.pi, 100)

algorithm_kwargs = {
    "fid_err_targ": 0.01,
    "alg": "RL",
    "max_iter": 300,
    "shorter_pulses": False,
}
```

```python
# Initialize the RL environment and start training
rl_result = optimize_pulses(
    objectives, control_parameters, tlist, algorithm_kwargs, optimizer_kwargs
)
```

```python
print(rl_result)
```

```python
# We can show in this case the hinton matrix

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
ax0.set_title("Initial")
ax1.set_title("Final")
ax2.set_title("Target")

qt.hinton(initial, ax=ax0)
qt.hinton(rl_result.final_states[0], ax=ax1)
qt.hinton(target, ax=ax2)
```

We are using PSU norm in the infidelity calculation, so the found transformation is correct, independently of the global phase.


## Open system

```python

```

Here we introduce the decoherence and dissipation effects considering the open system using the Liouvillian and collapse operators.

```python
# Define the initial and target states
initial = qt.qeye(2)  # Identity
target = qt.gates.hadamard_transform()

initial = qt.sprepost(initial, initial.dag())
target = qt.sprepost(target, target.dag())

# Define the control Hamiltonians (Pauli matrices)
H_c = [qt.sigmax(), qt.sigmay()]  # , qt.sigmaz()]

# Define the drift Hamiltonian
w, d, gamma = 0.1, 1.0, 0.1
H_d = 1 / 2 * (w * qt.sigmaz() + d * qt.sigmax())

# Liouvillian
L = qt.liouvillian(H=H_d, c_ops=[np.sqrt(gamma) * qt.sigmam()])
H_c = [qt.liouvillian(H) for H in H_c]
# Combine the Hamiltonians into a single list
H = [L] + H_c

# Define the objective
objectives = [Objective(initial, H, target)]

# Define the control parameters with bounds
control_parameters = {
    "p": {"bounds": [(-1, +1)]},
}

# Define the time interval
tlist = np.linspace(0, 2 * np.pi, 100)

# Define algorithm-specific settings
algorithm_kwargs = {
    "fid_err_targ": 0.01,
    "alg": "RL",
    "max_iter": 400,
    "shorter_pulses": False,
}
optimizer_kwargs = {}
```

```python
# Initialize the RL environment and start training
rl_result = optimize_pulses(
    objectives, control_parameters, tlist, algorithm_kwargs, optimizer_kwargs
)
```

```python
print(rl_result)
```

```python
# We can show in this case the hinton matrix

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
ax0.set_title("Initial")
ax1.set_title("Final")
ax2.set_title("Target")

qt.hinton(initial, ax=ax0)
qt.hinton(rl_result.final_states[0], ax=ax1)
qt.hinton(target, ax=ax2)
```

```python

```
