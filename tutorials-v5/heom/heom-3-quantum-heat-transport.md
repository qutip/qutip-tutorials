---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: qutip-dev
  language: python
  name: python3
---

# HEOM 3: Quantum Heat Transport

+++

## Introduction

In this notebook, we apply the QuTiP HEOM solver to a quantum system coupled to two bosonic baths and demonstrate how to extract information about the system-bath heat currents from the auxiliary density operators (ADOs).
We consider the setup described in Ref. \[1\], which consists of two coupled qubits, each connected to its own heat bath.
The Hamiltonian of the qubits is given by

$$ \begin{aligned} H_{\text{S}} &= H_1 + H_2 + H_{12} , \quad\text{ where }\\
H_K &= \frac{\epsilon}{2} \bigl(\sigma_z^K + 1\bigr) \quad  (K=1,2) \quad\text{ and }\quad H_{12} = J_{12} \bigl( \sigma_+^1 \sigma_-^2 + \sigma_-^1 \sigma_+^2 \bigr) . \end{aligned} $$

Here, $\sigma^K_{x,y,z,\pm}$ denotes the usual Pauli matrices for the K-th qubit, $\epsilon$ is the eigenfrequency of the qubits and $J_{12}$ the coupling constant.

Each qubit is coupled to its own bath; therefore, the total Hamiltonian is

$$ H_{\text{tot}} = H_{\text{S}} + \sum_{K=1,2} \bigl( H_{\text{B}}^K + Q_K \otimes X_{\text{B}}^K \bigr) , $$

where $H_{\text{B}}^K$ is the free Hamiltonian of the K-th bath and $X_{\text{B}}^K$ its coupling operator, and $Q_K = \sigma_x^K$ are the system coupling operators.
We assume that the bath spectral densities are given by Drude distributions

$$ J_K(\omega) = \frac{2 \lambda_K \gamma_K \omega}{\omega^2 + \gamma_K^2} , $$

where $\lambda_K$ is the free coupling strength and $\gamma_K$ the cutoff frequency.

We begin by defining the system and bath parameters.
We use the parameter values from Fig. 3(a) of Ref. \[1\].
Note that we set $\hbar$ and $k_B$ to one and we will measure all frequencies and energies in units of $\epsilon$.

References:

&nbsp;&nbsp; \[1\] Kato and Tanimura, [J. Chem. Phys. **143**, 064107](https://doi.org/10.1063/1.4928192) (2015).

+++

## Setup

```{code-cell} ipython3
import dataclasses

import numpy as np
import matplotlib.pyplot as plt

import qutip as qt
from qutip.solver.heom import (
    HEOMSolver,
    DrudeLorentzPadeBath
)
from qutip.core.environment import (
    CFExponent,
    DrudeLorentzEnvironment,
    system_terminator,
)

from ipywidgets import IntProgress
from IPython.display import display

%matplotlib inline
```

## Helpers

```{code-cell} ipython3
# Solver options:

options = {
    "nsteps": 15000,
    "store_states": True,
    "rtol": 1e-12,
    "atol": 1e-12,
    "min_step": 1e-18,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## System and bath definition

```{code-cell} ipython3
@dataclasses.dataclass
class SystemParams:
    """ System parameters and Hamiltonian. """
    epsilon: float = 1.0
    J12: float = 0.1

    def H(self):
        """ Return the Hamiltonian for the system.

            The system consists of two qubits with Hamiltonians (H1 and H2)
            and an interaction term (H12).
        """
        H1 = self.epsilon / 2 * (
            qt.tensor(qt.sigmaz() + qt.identity(2), qt.identity(2))
        )
        H2 = self.epsilon / 2 * (
            qt.tensor(qt.identity(2), qt.sigmaz() + qt.identity(2))
        )
        H12 = self.J12 * (
            qt.tensor(qt.sigmap(), qt.sigmam()) +
            qt.tensor(qt.sigmam(), qt.sigmap())
        )
        return H1 + H2 + H12

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)
```

```{code-cell} ipython3
@dataclasses.dataclass
class BathParams:
    """ Bath parameters. """
    sign: str  # + or -
    qubit: int  # 0 or 1

    gamma: float = 2.0
    lam: float = 0.05
    Tbar: float = 2.0
    Tdelta: float = 0.01

    def __post_init__(self):
        # T = Tbar +- Tdelta * Tbar:
        assert self.sign in ("+", "-")
        sign = +1 if self.sign == "+" else -1
        self.T = self.Tbar + sign * self.Tdelta * self.Tbar
        # qubit
        assert self.qubit in (0, 1)

    def Q(self):
        """ Coupling operator for the bath. """
        Q = [qt.identity(2), qt.identity(2)]
        Q[self.qubit] = qt.sigmax()
        return qt.tensor(Q)

    def bath(self, Nk, tag=None):
        env=DrudeLorentzEnvironment(
            lam=self.lam, gamma=self.gamma, T=self.T, tag=tag
        )
        env_approx,delta=env.approx_by_pade(Nk=Nk,compute_delta=True,tag=tag)
        return (env_approx,self.Q()),system_terminator(self.Q(),delta),delta

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)
```

## Heat currents

Following Ref. \[2\], we consider two possible definitions of the heat currents from the qubits into the baths.
The so-called bath heat currents are $j_{\text{B}}^K = \partial_t \langle H_{\text{B}}^K \rangle$ and the system heat currents are $j_{\text{S}}^K = \mathrm i\, \langle [H_{\text{S}}, Q_K] X_{\text{B}}^K \rangle$.
As shown in Ref. \[2\], they can be expressed in terms of the HEOM ADOs as follows:
$$ \begin{aligned} \mbox{} \\
    j_{\text{B}}^K &= \!\!\sum_{\substack{\mathbf n\\ \text{Level 1}\\ \text{Bath $K$}}}\!\! \nu[\mathbf n] \operatorname{tr}\bigl[ Q_K \rho_{\mathbf n} \bigr] - 2 C_I^K(0) \operatorname{tr}\bigl[ Q_k^2 \rho \bigr] + \Gamma_{\text{T}}^K \operatorname{tr}\bigl[ [[H_{\text{S}}, Q_K], Q_K]\, \rho \bigr] , \\[.5em]
    j_{\text{S}}^K &= \mathrm i\!\! \sum_{\substack{\mathbf n\\ \text{Level 1}\\ \text{Bath $k$}}}\!\! \operatorname{tr}\bigl[ [H_{\text{S}}, Q_K]\, \rho_{\mathbf n} \bigr] + \Gamma_{\text{T}}^K \operatorname{tr}\bigl[ [[H_{\text{S}}, Q_K], Q_K]\, \rho \bigr] . \\ \mbox{}
\end{aligned} $$
The sums run over all level-$1$ multi-indices $\mathbf n$ with one excitation corresponding to the K-th bath, $\nu[\mathbf n]$ is the corresponding (negative) exponent of the bath auto-correlation function $C^K(t)$, and $\Gamma_{\text{T}}^K$ is the Ishizaki-Tanimura terminator (i.e., a correction term accounting for the error introduced by approximating the correlation function with a finite sum of exponential terms).
In the expression for the bath heat currents, we left out terms involving $[Q_1, Q_2]$, which is zero in this example.

&nbsp;&nbsp; \[2\] Kato and Tanimura, [J. Chem. Phys. **145**, 224105](https://doi.org/10.1063/1.4971370) (2016).

+++

In QuTiP, these currents can be conveniently calculated as follows:

```{code-cell} ipython3
def bath_heat_current(bath_tag, ado_state, hamiltonian, coupling_op, delta=0):
    """
    Bath heat current from the system into the heat bath with the given tag.

    Parameters
    ----------
    bath_tag : str, tuple or any other object
        Tag of the heat bath corresponding to the current of interest.

    ado_state : HierarchyADOsState
        Current state of the system and the environment (encoded in the ADOs).

    hamiltonian : Qobj
        System Hamiltonian at the current time.

    coupling_op : Qobj
        System coupling operator at the current time.

    delta : float
        The prefactor of the \\delta(t) term in the correlation function (the
        Ishizaki-Tanimura terminator).
    """
    l1_labels = ado_state.filter(level=1, tags=[bath_tag])
    a_op = 1j * (hamiltonian * coupling_op - coupling_op * hamiltonian)

    result = 0
    cI0 = 0  # imaginary part of bath auto-correlation function (t=0)
    for label in l1_labels:
        [exp] = ado_state.exps(label)
        result += exp.vk * (coupling_op * ado_state.extract(label)).tr()

        if exp.type == CFExponent.types['I']:
            cI0 += exp.ck
        elif exp.type == CFExponent.types['RI']:
            cI0 += exp.ck2

    result -= 2 * cI0 * (coupling_op * coupling_op * ado_state.rho).tr()
    if delta != 0:
        result -= (
            1j * delta *
            ((a_op * coupling_op - coupling_op * a_op) * ado_state.rho).tr()
        )
    return result


def system_heat_current(
    bath_tag, ado_state, hamiltonian, coupling_op, delta=0,
):
    """
    System heat current from the system into the heat bath with the given tag.

    Parameters
    ----------
    bath_tag : str, tuple or any other object
        Tag of the heat bath corresponding to the current of interest.

    ado_state : HierarchyADOsState
        Current state of the system and the environment (encoded in the ADOs).

    hamiltonian : Qobj
        System Hamiltonian at the current time.

    coupling_op : Qobj
        System coupling operator at the current time.

    delta : float
        The prefactor of the \\delta(t) term in the correlation function (the
        Ishizaki-Tanimura terminator).
    """
    l1_labels = ado_state.filter(level=1, tags=[bath_tag])
    a_op = 1j * (hamiltonian * coupling_op - coupling_op * hamiltonian)

    result = 0
    for label in l1_labels:
        result += (a_op * ado_state.extract(label)).tr()

    if delta != 0:
        result -= (
            1j * delta *
            ((a_op * coupling_op - coupling_op * a_op) * ado_state.rho).tr()
        )
    return result
```

Note that at long times, we expect $j_{\text{B}}^1 = -j_{\text{B}}^2$ and $j_{\text{S}}^1 = -j_{\text{S}}^2$ due to energy conservation. At long times, we also expect $j_{\text{B}}^1 = j_{\text{S}}^1$ and $j_{\text{B}}^2 = j_{\text{S}}^2$ since the coupling operators commute, $[Q_1, Q_2] = 0$. Hence, all four currents should agree in the long-time limit (up to a sign). This long-time value is what was analyzed in Ref. \[2\].

+++

## Simulations

+++

For our simulations, we will represent the bath spectral densities using the first term of their Padé decompositions, and we will use $7$ levels of the HEOM hierarchy.

```{code-cell} ipython3
Nk = 1
NC = 7
```

### Time Evolution

We fix $J_{12} = 0.1 \epsilon$ (as in Fig. 3(a-ii) of Ref. \[2\]) and choose the fixed coupling strength $\lambda_1 = \lambda_2 = J_{12}\, /\, (2\epsilon)$ (corresponding to $\bar\zeta = 1$ in Ref. \[2\]).
Using these values, we will study the time evolution of the system state and the heat currents.

```{code-cell} ipython3
# fix qubit-qubit and qubit-bath coupling strengths
sys = SystemParams(J12=0.1)
bath_p1 = BathParams(qubit=0, sign="+", lam=sys.J12 / 2)
bath_p2 = BathParams(qubit=1, sign="-", lam=sys.J12 / 2)

# choose arbitrary initial state
rho0 = qt.tensor(qt.identity(2), qt.identity(2)) / 4

# simulation time span
tlist = np.linspace(0, 50, 250)
```

```{code-cell} ipython3
H = sys.H()

bath1,b1term,b1delta = bath_p1.bath(Nk, tag='bath 1')
Q1 = bath_p1.Q()

bath2,b2term,b2delta = bath_p2.bath(Nk, tag='bath 2')
Q2 = bath_p2.Q()


solver = HEOMSolver(
    qt.liouvillian(H) + b1term + b2term,
    [bath1, bath2],
    max_depth=NC,
    options=options,
)

result = solver.run(rho0, tlist, e_ops=[
    qt.tensor(qt.sigmaz(), qt.identity(2)),
    lambda t, ado: bath_heat_current('bath 1', ado, H, Q1, b1delta),
    lambda t, ado: bath_heat_current('bath 2', ado, H, Q2, b2delta),
    lambda t, ado: system_heat_current('bath 1', ado, H, Q1, b1delta),
    lambda t, ado: system_heat_current('bath 2', ado, H, Q2, b2delta),
])
```

We first plot $\langle \sigma_z^1 \rangle$ to see the time evolution of the system state:

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(8, 8))
axes.plot(tlist, result.expect[0], 'r', linewidth=2)
axes.set_xlabel('t', fontsize=28)
axes.set_ylabel(r"$\langle \sigma_z^1 \rangle$", fontsize=28);
```

We find a rather quick thermalization of the system state. For the heat currents, however, it takes a somewhat longer time until they converge to their long-time values:

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

ax1.plot(
    tlist, -np.real(result.expect[1]),
    color='darkorange', label='BHC (bath 1 -> system)',
)
ax1.plot(
    tlist, np.real(result.expect[2]),
    '--', color='darkorange', label='BHC (system -> bath 2)',
)
ax1.plot(
    tlist, -np.real(result.expect[3]),
    color='dodgerblue', label='SHC (bath 1 -> system)',
)
ax1.plot(
    tlist, np.real(result.expect[4]),
    '--', color='dodgerblue', label='SHC (system -> bath 2)',
)

ax1.set_xlabel('t', fontsize=28)
ax1.set_ylabel('j', fontsize=28)
ax1.set_ylim((-0.05, 0.05))
ax1.legend(loc=0, fontsize=12)

ax2.plot(
    tlist, -np.real(result.expect[1]),
    color='darkorange', label='BHC (bath 1 -> system)',
)
ax2.plot(
    tlist, np.real(result.expect[2]),
    '--', color='darkorange', label='BHC (system -> bath 2)',
)
ax2.plot(
    tlist, -np.real(result.expect[3]),
    color='dodgerblue', label='SHC (bath 1 -> system)',
)
ax2.plot(
    tlist, np.real(result.expect[4]),
    '--', color='dodgerblue', label='SHC (system -> bath 2)',
)

ax2.set_xlabel('t', fontsize=28)
ax2.set_xlim((20, 50))
ax2.set_ylim((0, 0.0002))
ax2.legend(loc=0, fontsize=12);
```

### Steady-state currents

Here, we try to reproduce the HEOM curves in Fig. 3(a) of Ref. \[1\] by varying the coupling strength and finding the steady state for each coupling strength.

```{code-cell} ipython3
def heat_currents(sys, bath_p1, bath_p2, Nk, NC, options):
    """ Calculate the steady sate heat currents for the given system and
        bath.
    """

    bath1,b1term,b1delta = bath_p1.bath(Nk, tag='bath 1')
    Q1 = bath_p1.Q()

    bath2,b2term,b2delta = bath_p2.bath(Nk, tag='bath 2')
    Q2 = bath_p2.Q()

    solver = HEOMSolver(
        qt.liouvillian(sys.H()) + b1term + b2term,
        [bath1, bath2],
        max_depth=NC,
        options=options
    )

    _, steady_ados = solver.steady_state()

    return (
        bath_heat_current('bath 1', steady_ados, sys.H(), Q1, b1delta),
        bath_heat_current('bath 2', steady_ados, sys.H(), Q2, b2delta),
        system_heat_current('bath 1', steady_ados, sys.H(), Q1, b1delta),
        system_heat_current('bath 2', steady_ados, sys.H(), Q2, b2delta),
    )
```

```{code-cell} ipython3
# Define number of points to use for the plot
plot_points = 10  # use 100 for a smoother curve

# Range of relative coupling strengths
# Chosen so that zb_max is maximum, centered around 1 on a log scale
zb_max = 4  # use 20 to see more of the current curve
zeta_bars = np.logspace(
    -np.log(zb_max),
    np.log(zb_max),
    plot_points,
    base=np.e,
)

# Setup a progress bar
progress = IntProgress(min=0, max=(3 * plot_points))
display(progress)


def calculate_heat_current(J12, zb, Nk, progress=progress):
    """ Calculate a single heat current and update the progress bar. """
    # Estimate appropriate HEOM max_depth from coupling strength
    NC = 7 + int(max(zb * J12 - 1, 0) * 2)
    NC = min(NC, 20)
    # the four currents are identical in the steady state
    j, _, _, _ = heat_currents(
        sys.replace(J12=J12),
        bath_p1.replace(lam=zb * J12 / 2),
        bath_p2.replace(lam=zb * J12 / 2),
        Nk, NC, options=options,
    )
    progress.value += 1
    return j


# Calculate steady state currents for range of zeta_bars
# for J12 = 0.01, 0.1 and 0.5:
j1s = [
    calculate_heat_current(0.01, zb, Nk)
    for zb in zeta_bars
]
j2s = [
    calculate_heat_current(0.1, zb, Nk)
    for zb in zeta_bars
]
j3s = [
    calculate_heat_current(0.5, zb, Nk)
    for zb in zeta_bars
]
```

## Create Plot

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(12, 7))

axes.plot(
    zeta_bars, -1000 * 100 * np.real(j1s),
    'b', linewidth=2, label=r"$J_{12} = 0.01\, \epsilon$",
)
axes.plot(
    zeta_bars, -1000 * 10 * np.real(j2s),
    'r--',  linewidth=2, label=r"$J_{12} = 0.1\, \epsilon$",
)
axes.plot(
    zeta_bars, -1000 * 2 * np.real(j3s),
    'g-.', linewidth=2, label=r"$J_{12} = 0.5\, \epsilon$",
)

axes.set_xscale('log')
axes.set_xlabel(r"$\bar\zeta$", fontsize=30)
axes.set_xlim((zeta_bars[0], zeta_bars[-1]))

axes.set_ylabel(
    r"$j_{\mathrm{ss}}\; /\; (\epsilon J_{12}) \times 10^3$",
    fontsize=30,
)
axes.set_ylim((0, 2))

axes.legend(loc=0);
```

## About

```{code-cell} ipython3
qt.about()
```

## Testing

This section can include some tests to verify that the expected outputs are generated within the notebook. We put this section at the end of the notebook, so it's not interfering with the user experience. Please, define the tests using assert, so that the cell execution fails if a wrong output is generated.

```{code-cell} ipython3
assert 1 == 1
```
