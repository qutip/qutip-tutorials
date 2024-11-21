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

# QuTiPv5 Paper Example: `sesolve` and `mesolve` and the new solver class

Authors: Maximilian Meyer-Mölleringhof (m.meyermoelleringhof@gmail.com), Neill Lambert (nwlambert@gmail.com)

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
                   sigmam, sigmax, sigmaz, spost, spre, sprepost)

# from qutip.solver.heom import HEOMSolver, BosonicBath

%matplotlib inline
```

## Part 0: Introduction to the New Solver Class

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

## Part 1: Lindblad Dynamics and Beyond

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

### Global Master Equation - Born-Markov-secular approximation

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


def make_co_list(energies, eigenstates):
    Nmax = len(eigenstates)
    collapse_list = []
    for i in range(Nmax):
        for j in range(Nmax):
            delE = energies[j] - energies[i]
            m1 = sx1.matrix_element(eigenstates[i].dag(), eigenstates[j])
            m2 = sx2.matrix_element(eigenstates[i].dag(), eigenstates[j])
            absolute = np.abs(m1) ** 2 + np.abs(m2) ** 2
            rate = power_spectrum(delE) * absolute
            if rate > 0:
                outer = eigenstates[i] * eigenstates[j].dag()
                collapse_list.append(np.sqrt(rate) * outer)
    return collapse_list
```

```python
all_energy, all_state = H.eigenstates()
collapse_list = make_co_list(all_energy, all_state)
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
# generate new collapse operators for weak coupling Hamiltonian
all_energy, all_state = H_weak.eigenstates()
co_list = make_co_list(all_energy, all_state)
```

```python
me_local_res = mesolve(H_weak, psi0, tlist, c_ops, e_ops=[sz1, sz2])
me_global_res = mesolve(H_weak, psi0, tlist, co_list, e_ops=[sz1, sz2])
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
# generate new collapse operators for weak coupling Hamiltonian
all_energy, all_state = H_strong.eigenstates()
co_list = make_co_list(all_energy, all_state)

# time list with smaller steps
tlist_fine = np.linspace(0, 40, 1000)
```

```python
me_local_res = mesolve(H_strong, psi0, tlist_fine, c_ops, e_ops=[sz1, sz2])
me_global_res = mesolve(H_strong, psi0, tlist_fine, co_list, e_ops=[sz1, sz2])
br_res = brmesolve(
    H_strong,
    psi0,
    tlist_fine,
    e_ops=[sz1, sz2],
    a_ops=[[sx1, power_spectrum], [sx2, power_spectrum]],
)
```

```python
plt.plot(tlist_fine, me_local_res.expect[0], label=r"Local Lindblad")
plt.plot(tlist_fine, me_global_res.expect[0], "--", label=r"Dressed Lindblad")
plt.plot(tlist_fine, br_res.expect[0], ":", label=r"Bloch-Redfield")
plt.title("Strong Coupling")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

### Manual Liouvillian Superoperator

As mentioned before, QuTiP's master equation solver can be used to solve any other master equation.
By using `spre()`, `spost()` and `sprepost()`, we can manually construct such equations.
These functions specifically convert the operators from the original Hilbert space to operators in the double space which is internally used by QuTiP to optimize computations (see the paper for more details).

For example, the Lindbladian corresponding to the master equation of the previous example can be constructed manually via:


```python
lindbladian = -1.0j * (spre(H) - spost(H))
for c in c_ops:
    lindbladian += sprepost(c, c.dag())
    lindbladian -= 0.5 * (spre(c.dag() * c) + spost(c.dag() * c))
```

```python
manual_res = mesolve(lindbladian, psi0, tlist_fine, [], e_ops=[sz1, sz2])
```

```python
plt.plot(tlist_fine, me_local_res.expect[0], label="i=1")
plt.plot(tlist_fine, me_local_res.expect[1], label="i=2")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(i)} \rangle$")
plt.legend()
plt.show()
```

<!-- #region -->
## Part 2: Time-Dependent Systems


Finally, we compare the results with another `mesolve` considering the rotating-wave approximation, the Bloch-Redfield solver and the HEOMSolver.
<!-- #endregion -->

```python
# Hamiltonian parameters
Delta = 2 * np.pi  # qubit splitting
omega_d = Delta  # drive frequency
A = 0.01 * Delta  # drive amplitude

# Bath parameters
gamma = 0.005 * Delta / (2 * np.pi)  # dissipation strength
temp = 0  # temperature

# Simulation parameters
psi0 = basis(2, 0)  # initial state
e_ops = [sigmaz()]
T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 1000 * T, 500)
```

### With `mesolve`

```python
# driving field
def f(t):
    return np.sin(omega_d * t)
```

```python
H0 = Delta / 2.0 * sigmaz()
H1 = [A / 2.0 * sigmax(), f]
H = [H0, H1]
```

```python
c_ops_me = [np.sqrt(gamma) * sigmam()]
```

```python
driv_res = mesolve(H, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)
```

### With Rotating-Wave-Approximated `mesolve`

```python
H_RWA = (Delta - omega_d) * 0.5 * sigmaz() + A / 4 * sigmax()
c_ops_me_RWA = [np.sqrt(gamma) * sigmam()]
```

```python
driv_RWA_res = mesolve(H_RWA, psi0, tlist, c_ops=c_ops_me_RWA, e_ops=e_ops)
```

### With `brmesolve`

```python
# Bose einstein distribution
def nth(w):
    if temp > 0:
        return 1 / (np.exp(w / temp) - 1)
    else:
        return 0


# Power spectrum
def power_spectrum(w):
    if w > 0:
        return gamma * (nth(w) + 1)
    elif w == 0:
        return 0
    else:
        return gamma * nth(-w)
```

```python
a_ops = [[sigmax(), power_spectrum]]
```

```python
driv_br_res = brmesolve(H, psi0, tlist, a_ops, e_ops, sec_cutoff=-1)
```

### With `HEOMSolver`

```python
wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi

gamma_heom = 1.9 * w0


lambd = np.sqrt(
    0.5
    * gamma
    * ((w0**2 - wsamp**2) ** 2 + (gamma_heom**2) * ((wsamp) ** 2))
    / (gamma_heom * wsamp)
)
```

```python
# TODO UnderDampedEnvironment is in a big separate on Overleaf, how to handle?

# Create Environment
# bath = UnderDampedEnvironment(lam=lambd, w0=w0, gamma=gamma_heom, T=1e-30)
fit_times = np.linspace(0, 5, 1000)  # range for correlation function fit
```

### Comparison of Solver Results

```python
plt.figure()

plt.plot(tlist, driv_res.expect[0], "-", label="mesolve (time-dep)")
plt.plot(tlist, driv_RWA_res.expect[0], "-.", label="mesolve (rwa)")
# plt.plot(tlist, results_corr_fit.expect[0], '--', label=r'heomsolve')
plt.plot(tlist, driv_br_res.expect[0], ":", linewidth=3, label="brmesolve")

plt.xlabel(r"$t\, /\, \Delta^{-1}$")
plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.legend()
plt.show()
```

### Adiabatic Energy Switching

To illustrate where using naive local-basis collapse operators can fail, we look at a single qubit whose energies are adiabatically switched between positive and negative values.

$H = \dfrac{\Delta}{2} \sin{(\omega_d t)} \sigma_z$.

For slow drives, we expect the bath to respond to this change.
Therefore, transitions from higher to lower energy levels should be induced.

```python
# Hamiltonian
omega_d = 0.05 * Delta  # drive frequency
A = Delta  # drive amplitude
H_adi = [[A / 2.0 * sigmaz(), f]]
# H = [H0]

# Bath parameters
gamma = 0.05 * Delta / (2 * np.pi)

# Simulation parameters
tlist = np.linspace(0, 2 * T, 400)
```

```python
# Simple mesolve
adi_me_res = mesolve(H_adi, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)
```

```python
# bath = UnderDampedEnvironment(lam=lambd, w0=w0, gamma=gamma_heom, T=1e-30)
# fit_times = np.linspace(0, 5, 1000)  # range for correlation function fit

# cfit, fit_info = bath.approx_by_cf_fit(fit_times,
#                                        Ni_max=1, Nr_max=2, target_rsme=None)
# print(fit_info["summary"])
# Convert to a HEOM bath
# heombath = cfit.to_bath(sigmax())


# HEOM_corr_fit = HEOMSolver(
# qt.QobjEvo(H),
# heombath,
# max_depth=max_depth,
# options={"nsteps": 15000, "rtol": 1e-12, "atol": 1e-12},
# )
# results_corr_fit = HEOM_corr_fit.run(psi0 * psi0.dag(), tlist, e_ops=e_ops)
```

```python
# BRSolve
brme_result2 = brmesolve(H, psi0, tlist, a_ops=a_ops, e_ops=e_ops)
```

```python
# BRSolve non-flat power spectrum
# a_ops_non_flat = [[sigmax(), lambda w: cfit.power_spectrum(w).item()]]
# brme_result = brmesolve(H, psi0, tlist, a_ops=a_ops_non_flat, e_ops=e_ops)
```

```python
plt.plot(tlist, adi_me_res.expect[0], "-", label="mesolve")
# plt.plot(tlist, results_corr_fit.expect[0], '--', label=r'heomsolve')
# plt.plot(tlist, brme_result.expect[0], ":", linewidth=6,
#                                           label="brmesolve non-flat")
plt.plot(tlist, brme_result2.expect[0], ":", linewidth=6, label="brmesolve")

plt.xlabel(r"$t\, /\, \Delta^{-1}$", fontsize=18)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=18)
plt.legend()
```

## About

```python
about()
```

## Testing

```python
# test sesolve gives the same result as SESolver
```
