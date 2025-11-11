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

In this notebook, we will demonstrate how to use the `_RL` module to solve a quantum optimal control problem using reinforcement learning (RL).
The goal is to use 2 Qubits to realize CNOT gate. In practice there is a control qubit and a target qubit, if the control qubit is in the state |0⟩ the target qubit remains unchanged, if the control qubit is in the state |1⟩ the CNOT gate flips the state of the target qubit.



### Setup and Import Required Libraries

```python
# If you are running this in an environment where some packages are missing, use this cell to install them:
# !pip install qutip stable-baselines3 gymnasium

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip_qoc import Objective, optimize_pulses
from stable_baselines3 import PPO
```

### Define the Quantum Control Problem


The system starts from an initial state represented by the identity on two qubits, with the goal of achieving a CNOT gate as the target state. To accomplish this, control operators based on the Pauli matrices are defined to act on individual qubits and pairs of qubits. Additionally, a drift Hamiltonian is introduced to account for interactions between the qubits and noise, thereby modeling the dynamics of the open quantum system.

```python
# Define the initial and target states
initial = qt.tensor(qt.qeye(2), qt.qeye(2))
target = qt.gates.cnot()

# convert to superoperator (for open system)
initial = qt.sprepost(initial, initial.dag())
target = qt.sprepost(target, target.dag())

# single qubit control operators
sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
identity = qt.qeye(2)

# two qubit control operators
i_sx, sx_i = qt.tensor(sx, identity), qt.tensor(identity, sx)
i_sy, sy_i = qt.tensor(sy, identity), qt.tensor(identity, sy)
i_sz, sz_i = qt.tensor(sz, identity), qt.tensor(identity, sz)

# Define the control Hamiltonians
Hc = [i_sx, i_sy, i_sz, sx_i, sy_i, sz_i]
Hc = [qt.liouvillian(H) for H in Hc]

# drift and noise term for a two-qubit system
omega, delta, gamma = 0.1, 1.0, 0.1
i_sm, sm_i = qt.tensor(qt.sigmam(), identity), qt.tensor(identity, qt.sigmam())

# energy levels and interaction
Hd = omega * (i_sz + sz_i) + delta * i_sz * sz_i
Hd = qt.liouvillian(H=Hd, c_ops=[gamma * (i_sm + sm_i)])

# combined operator list
H = [Hd, Hc[0], Hc[1], Hc[2], Hc[3], Hc[4], Hc[5]]

# Define the objective
objectives = [Objective(initial, H, target)]

# Define the control parameters with bounds
control_parameters = {"p": {"bounds": [(-30, 30)]}}

# Define the time interval
tlist = np.linspace(0, np.pi, 100)

# Define algorithm-specific settings
algorithm_kwargs = {
    "fid_err_targ": 0.01,
    "alg": "RL",
    "max_iter": 400,
    "shorter_pulses": False,
}
optimizer_kwargs = {}
```

Note that `max_iter` defines the number of episodes, the 100 in `tlist` defines the maximum number of steps per episode.  
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
# We can show the hinton matrix
fig, ax = qt.hinton(rl_result._final_states[0])
ax.set_title("hinton")
```

```python

```
