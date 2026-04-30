---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# HEOM 3：量子热输运

+++

## 引言

本笔记将 QuTiP 的 HEOM 求解器应用到一个耦合双玻色热浴的量子系统，并演示如何从辅助密度算符（ADO）中提取系统-热浴热流信息。
我们考虑参考文献 \[1\] 的装置：两个相互耦合的量子比特，每个量子比特连接各自热浴。
量子比特哈密顿量为

$$ \begin{aligned} H_{\text{S}} &= H_1 + H_2 + H_{12} , \quad\text{ where }\\
H_K &= \frac{\epsilon}{2} \bigl(\sigma_z^K + 1\bigr) \quad  (K=1,2) \quad\text{ and }\quad H_{12} = J_{12} \bigl( \sigma_+^1 \sigma_-^2 + \sigma_-^1 \sigma_+^2 \bigr) . \end{aligned} $$

其中 $\sigma^K_{x,y,z,\pm}$ 表示第 $K$ 个量子比特的常见 Pauli 算符，$\epsilon$ 是量子比特本征频率，$J_{12}$ 是耦合常数。

每个量子比特都耦合到自己的热浴，因此总哈密顿量为

$$ H_{\text{tot}} = H_{\text{S}} + \sum_{K=1,2} \bigl( H_{\text{B}}^K + Q_K \otimes X_{\text{B}}^K \bigr) , $$

其中 $H_{\text{B}}^K$ 是第 $K$ 个热浴的自由哈密顿量，$X_{\text{B}}^K$ 是其耦合算符，且系统耦合算符取 $Q_K=\sigma_x^K$。
我们假设热浴谱密度服从 Drude 形式

$$ J_K(\omega) = \frac{2 \lambda_K \gamma_K \omega}{\omega^2 + \gamma_K^2} , $$

其中 $\lambda_K$ 是耦合强度，$\gamma_K$ 是截止频率。

首先定义系统与热浴参数。
参数值采用参考文献 \[1\] 图 3(a)。
注意我们取 $\hbar=k_B=1$，并以 $\epsilon$ 为单位度量全部频率与能量。

参考文献：

&nbsp;&nbsp; \[1\] Kato and Tanimura, [J. Chem. Phys. **143**, 064107](https://doi.org/10.1063/1.4928192) (2015).

+++

## 设置

```{code-cell} ipython3
import dataclasses

import numpy as np
import matplotlib.pyplot as plt

import qutip as qt
from qutip.core.environment import (CFExponent, DrudeLorentzEnvironment,
                                    system_terminator)
from qutip.solver.heom import HEOMSolver

from ipywidgets import IntProgress
from IPython.display import display

%matplotlib inline
```

## 辅助函数

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

## 系统与热浴定义

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
        env = DrudeLorentzEnvironment(
            lam=self.lam, gamma=self.gamma, T=self.T, tag=tag
        )
        env_approx, delta = env.approximate(
            "pade", Nk=Nk, compute_delta=True, tag=tag
        )
        return (
            (env_approx, self.Q()),
            system_terminator(self.Q(), delta),
            delta,
        )

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)
```

## 热流

根据参考文献 \[2\]，我们考虑两种从量子比特流向热浴的热流定义。
所谓浴热流为 $j_{\text{B}}^K=\partial_t\langle H_{\text{B}}^K\rangle$，系统热流为 $j_{\text{S}}^K=\mathrm i\,\langle[H_{\text{S}},Q_K]X_{\text{B}}^K\rangle$。
如参考文献 \[2\] 所示，它们可由 HEOM 的 ADO 表示为：
$$ \begin{aligned} \mbox{} \\
    j_{\text{B}}^K &= \!\!\sum_{\substack{\mathbf n\\ \text{Level 1}\\ \text{Bath $K$}}}\!\! \nu[\mathbf n] \operatorname{tr}\bigl[ Q_K \rho_{\mathbf n} \bigr] - 2 C_I^K(0) \operatorname{tr}\bigl[ Q_k^2 \rho \bigr] + \Gamma_{\text{T}}^K \operatorname{tr}\bigl[ [[H_{\text{S}}, Q_K], Q_K]\, \rho \bigr] , \\[.5em]
    j_{\text{S}}^K &= \mathrm i\!\! \sum_{\substack{\mathbf n\\ \text{Level 1}\\ \text{Bath $k$}}}\!\! \operatorname{tr}\bigl[ [H_{\text{S}}, Q_K]\, \rho_{\mathbf n} \bigr] + \Gamma_{\text{T}}^K \operatorname{tr}\bigl[ [[H_{\text{S}}, Q_K], Q_K]\, \rho \bigr] . \\ \mbox{}
\end{aligned} $$
求和遍历所有与第 $K$ 个热浴对应、且只有一个激发的一级多重指标 $\mathbf n$；$\nu[\mathbf n]$ 是对应热浴自关联函数 $C^K(t)$ 的（负）指数，$\Gamma_{\text{T}}^K$ 是 Ishizaki-Tanimura 终止子（即补偿用有限指数和近似关联函数所引入误差的修正项）。
在浴热流表达式中，我们略去了含 $[Q_1,Q_2]$ 的项，因为本例中该对易子为零。

&nbsp;&nbsp; \[2\] Kato and Tanimura, [J. Chem. Phys. **145**, 224105](https://doi.org/10.1063/1.4971370) (2016).

+++

在 QuTiP 中，这些热流可方便地按如下方式计算：

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

        if exp.type == CFExponent.types["I"]:
            cI0 += exp.ck
        elif exp.type == CFExponent.types["RI"]:
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

注意在长时间极限下，由能量守恒应有 $j_{\text{B}}^1=-j_{\text{B}}^2$ 与 $j_{\text{S}}^1=-j_{\text{S}}^2$。又因耦合算符对易（$[Q_1,Q_2]=0$），还应有 $j_{\text{B}}^1=j_{\text{S}}^1$ 与 $j_{\text{B}}^2=j_{\text{S}}^2$。因此四条热流在长时间极限应一致（仅差符号）。参考文献 \[2\] 主要分析的正是这一长时间值。

+++

## 模拟

+++

在本模拟中，我们用热浴 Padé 分解的第一项表示谱密度，并采用 HEOM 层级深度 7。

```{code-cell} ipython3
Nk = 1
NC = 7
```

### 时间演化

固定 $J_{12}=0.1\epsilon$（对应参考文献 \[2\] 图 3(a-ii)），并取固定耦合强度 $\lambda_1=\lambda_2=J_{12}/(2\epsilon)$（对应参考文献 \[2\] 中 $\bar\zeta=1$）。
在这些参数下，我们研究系统态与热流的时间演化。

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

bath1, b1term, b1delta = bath_p1.bath(Nk, tag="bath 1")
Q1 = bath_p1.Q()

bath2, b2term, b2delta = bath_p2.bath(Nk, tag="bath 2")
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

先绘制 $\langle\sigma_z^1\rangle$ 以观察系统态随时间变化：

```{code-cell} ipython3
fig, axes = plt.subplots(figsize=(8, 8))
axes.plot(tlist, np.real(result.expect[0]), 'r', linewidth=2)
axes.set_xlabel('t', fontsize=28)
axes.set_ylabel(r"$\langle \sigma_z^1 \rangle$", fontsize=28);
```

可见系统态较快热化；但热流需要更长时间才收敛到长时间极限。

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

### 稳态热流

这里我们通过改变耦合强度并对每个耦合值求稳态，尝试复现参考文献 \[1\] 图 3(a) 中的 HEOM 曲线。

```{code-cell} ipython3
def heat_currents(sys, bath_p1, bath_p2, Nk, NC, options):
    """ Calculate the steady sate heat currents for the given system and
        bath.
    """

    bath1, b1term, b1delta = bath_p1.bath(Nk, tag="bath 1")
    Q1 = bath_p1.Q()

    bath2, b2term, b2delta = bath_p2.bath(Nk, tag="bath 2")
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
j1s = [calculate_heat_current(0.01, zb, Nk) for zb in zeta_bars]
j2s = [calculate_heat_current(0.1, zb, Nk) for zb in zeta_bars]
j3s = [calculate_heat_current(0.5, zb, Nk) for zb in zeta_bars]
```

## 绘图

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

## 关于

```{code-cell} ipython3
qt.about()
```
