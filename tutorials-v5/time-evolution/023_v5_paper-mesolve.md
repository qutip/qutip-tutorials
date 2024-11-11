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

# QuTiPv5 Paper Example: `sesolve` and `mesolve` with the new solver class

In QuTiP v5 a unified interface for interacting with solvers is introduced.
This can be useful when the same Hamiltonian data is reused with different initial conditions, time steps or other options.
A significant speed-up can therefore be achieved if the solver is reused many times.

When the solver is instantiated, one first supplies only the Hamiltonian and the collapse operators (e.g., collapse operators for a Lindabladian master equation).
Initial conditions, time steps, etc. are passed to the `Solver.run()` method which then performs the simulation.

In this notebook we will consider several examples illustrating the usage of the new solver classes.

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (SESolver, about, basis, brmesolve, fidelity, mesolve, qeye,
                   sigmam, sigmax, sigmaz)
```

## Example 1:

In our first example, we want to look at two interacting qubits that are (for now) decoupled from an environment.
Such a system is described by the Hamiltonian

$H = \dfrac{\epsilon_1}{2} \sigma_z^{(1)} + \dfrac{\epsilon_2}{2} \sigma_z^{(2)} + g \sigma_{x}^{(1)} \sigma_{x}^{(2)}$.

The Pauli matrices $\sigma_z^{(1/2)}$ describe the respective two-level system of the qubits.
The qubits are coupled via $\sigma_{x}^{(1/2)}$ and their interaction strength is given by $g$.

```python
epsilon1 = 1.0
epsilon2 = 1.0
g = 0.1

sx1 = sigmax() & qeye(2)
sx2 = qeye(2) & sigmax()
sz1 = sigmaz() & qeye(2)
sz2 = qeye(2) & sigmaz()

H = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2

print(H)
```

The dynamics of such a system is described by the Schrödinger equation

$i \hbar \dfrac{d}{dt} \ket{\psi} = H \ket{\psi}$.
Therefore, we can use `SESovler` to calculate the dynamics.

```python
se_solver = SESolver(H)
psi0 = basis(2, 0) & basis(2, 1)
tlist = np.linspace(0, 40, 100)
```

```python
se_res = se_solver.run(psi0, tlist, e_ops=[sz1, sz2])
```

```python
plt.plot(tlist, se_res.expect[0], label="i=1")
plt.plot(tlist, se_res.expect[1], label="i=2")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(i)} \rangle$")
plt.legend()
plt.show()
```

### Manual Stepping Interface

A new feature in QuTiP v5 is that time steps can be manually controlled.
This is specficially useful if the Hamiltonian depends on external control parameters such as field strength.
Such parameters can be updated in euch step using the optional paramter `args`.
In paractice, this can look like this:

```python
t = 0
dt = 40 / 100
se_solver2 = SESolver(H)
se_solver2.start(psi0, t)
while t < 40:
    t = t + dt
    psi = se_solver2.step(t)
    # process the result psi and calculate next time step
```

### Solver and Integrator Options

Another change in QuTiP v5 is that the `options` argument takes a standard Python dictionary.
This should increase future felxibility and allow different solvers to provide individual sets of options more easily.
The complete list of options can be found in the online documentation for each solver.

As an example of frequently used options, we show `store_states`, determining whether the system state at each time step should be included in the output and
`store_final_state` telling the solver if the final state of the evolution should be included.
These states in addition to the requested observables are then included in the final result.
Another common option is `method`, specifying the ODE integration method as well as specific options for it.
Also shown here are `atol` to control the precision (absolute tolerance).
`nsteps` controls the maximum number of steps between two time steps and `max_step` refers to the maximum allowed integration step of the default Adams ODE.

```python
options = {"store_states": True, "atol": 1e-12, "nsteps": 1e3, "max_step": 0.1}
se_solver.options = options
```

```python
se_res = se_solver.run(psi0, tlist)
print(se_res)
```

## Example 2: Lindblad Dynamics and Beyond

In principle, the Schrödinger equation describes the dynamics of any quantum system.
However, it is often impossible to solve once larger or even continious systems are explored.
To solve this issue, master equations were developed and have now become the most common way to describe the dynamics of finite (open) quantum systems.
Generally, a master equation refers to a first-order linear differential equation for $\rho(t)$, the reduced density operator describing the quantum state.
Although `mesolve` supports master equations of various forms, the Lindbladian type is implemented by default in QuTiP.
Such an equation has the following form:

$\dot{\rho}(t) = - \dfrac{i}{\hbar} [H(t), \rho(t)] + \sum_n \dfrac{1}{2}[ 2 C_n \rho(t) C_n^\dag - \rho(t) C_n^\dag C_n - C^\dag_n C_n \rho(t) ]$.

Next to the desnity operator $\rho(t)$ and the Hamiltonian $H(t)$, this equation includes the so-called collapse (or jump) operators $C_n = \sqrt{\gamma_n} A_n$.
They define the dissipation due to contanct with and environment.
$\gamma_n$ can hereby be understood as rates describing the frequency of transitions between the states connected by the operator $A_n$.

To continue our example of the two qubits, we now connect them to an evironment using the collapse operators $C_1 = \sqrt{\gamma} \sigma_{-}^{(1)}$ and $C_2 = \sqrt{\gamma} \sigma_{-}^{(2)}$ where $\sigma_{-}^{i}$ takes qubit (i) from its excited state to its ground state.
This time of course we will be using the `mesolve` solver.

```python
sm1 = sigmam() & qeye(2)
sm2 = qeye(2) & sigmam()
gam = 0.1  # dissipation rate
c_ops = [np.sqrt(gam) * sm1, np.sqrt(gam) * sm2]
```

```python
me_local_res = mesolve(H, psi0, tlist, c_ops, e_ops=[sz1, sz2])
```

```python
plt.plot(tlist, me_local_res.expect[0], label="i=1")
plt.plot(tlist, me_local_res.expect[1], label="i=2")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(i)} \rangle$")
plt.legend()
plt.show()
```

### Global Master Equation - Born-Markoc-secular approximation

In the previous example, the collapse operators act locally on each qubit.
However, different kinds of collapse operators can be found under certain approximations.
For example, if the qubits interact more strongly with each other than with the bath, on arrives at the global master equation under the standard Born-Markox approximations.
Here, the collapse operators act like annihilation and creation operators on the total coupled eigenstates of the interacting two-qubit system

$A_{ij} = \ket{\psi_i}\bra{\psi_i}$

and rates

$\gamma_{ij} = | \bra{\psi_i} d \ket{psi_j} |^2 S(\Delta_{ij})$.

The states $\ket{\psi_i}$ are the eigenstates of $H$ and $\Delta_{ij} = E_j - E_i$ are the difference of eigenenergies.
Furthermore, $d$ is the coupling operator of the system to the environment.
The power spectrum

$S(\omega) = 2 J(\omega) [n_{th} (\omega) + 1] \theta(\omega) + 2J(-\omega)[n_{th}(-\omega)]\theta(-\omega)$,

depends on details of the environment like its spectral density $J(\omega)$ and its temperature through the Bose-Einstein distribution $n_{th} (\omega)$.
Here, $\theta$ is the Heaviside function.

Assuming a flat spectral density $J(\omega) = \gamma / 2$ and zero temperature gives $S(\omega) = \gamma \theta(\omega)$.
For this example, we manually implement this zero temperature environment for our two-qubit system using `mesolve()`.

```python
def power_spectrum(w):
    if w >= 0:
        return gam
    else:
        return 0
```

```python
all_energy, all_state = H.eigenstates()
Nmax = len(all_state)
collapse_list = []
for i in range(Nmax):
    for j in range(Nmax):
        delE = all_energy[j] - all_energy[i]
        absolute = (
            np.abs(sx1.matrix_element(all_state[i].dag(), all_state[j])) ** 2
            + np.abs(sx2.matrix_element(all_state[i].dag(), all_state[j])) ** 2
        )
        rate = power_spectrum(delE) * absolute
        if rate > 0:
            outer = all_state[i] * all_state[j].dag()
            collapse_list.append(np.sqrt(rate) * outer)

tlist_long = np.linspace(0, 1000, 100)
```

```python
opt = {"store_states": True}
me_global_res = mesolve(
    H, psi0, tlist_long, collapse_list, e_ops=[sz1, sz2], options=opt
)
```

```python
grnd_state = all_state[0] @ all_state[0].dag()
fidelity = fidelity(me_global_res.states[-1], grnd_state)
print(f"Fidelity with ground-state: {fidelity:.6f}")
```

It is interesting to note that the long-time evolution leads to a state that is close to the coupled ground state of the two qubit system.


### Solver comparison

In the following, we compare the results of the local and dressed (global) Lindblad simulations from above with the Bloch-Redfield solver.
The Bloch-Redfield solver is explained in more detail in other tutorials but we use it here to solve the weak-coupling master equation from a given bath power spectrum.
When the qubit-qubit coupling is small, the results from the local and global master equations both agree with the Bloch-Redfield solver.
For large coupling, however, the local master equation deviates from the global and the Bloch-Redfield approach.

```python
# weak coupling
g = 0.1 * epsilon1
H_weak = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2
```

```python
me_local_res = mesolve(H_weak, psi0, tlist, c_ops, e_ops=[sz1, sz2])
me_global_res = mesolve(H_weak, psi0, tlist, collapse_list, e_ops=[sz1, sz2])
br_res = brmesolve(
    H_weak,
    psi0,
    tlist,
    e_ops=[sz1, sz2],
    a_ops=[[sx1, power_spectrum], [sx2, power_spectrum]],
)
```

```python
plt.plot(tlist, me_local_res.expect[0], label=r"Local Lindblad")
plt.plot(tlist, me_global_res.expect[0], "--", label=r"Dressed Lindblad")
plt.plot(tlist, br_res.expect[0], ":", label=r"Bloch-Redfield")
plt.title("Weak Coupling")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

```python
# strong coupling
g = 2 * epsilon1
H_strong = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2
```

```python
me_local_res = mesolve(H_strong, psi0, tlist, c_ops, e_ops=[sz1, sz2])
me_global_res = mesolve(H_strong, psi0, tlist, collapse_list, e_ops=[sz1, sz2])
br_res = brmesolve(
    H_strong,
    psi0,
    tlist,
    e_ops=[sz1, sz2],
    a_ops=[[sx1, power_spectrum], [sx2, power_spectrum]],
)
```

```python
plt.plot(tlist, me_local_res.expect[0], label=r"Local Lindblad")
plt.plot(tlist, me_global_res.expect[0], "--", label=r"Dressed Lindblad")
plt.plot(tlist, br_res.expect[0], ":", label=r"Bloch-Redfield")
plt.title("Weak Coupling")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

## About

```python
about()
```

## Testing

```python
# test sesolve gives the same result as SESolver
```
