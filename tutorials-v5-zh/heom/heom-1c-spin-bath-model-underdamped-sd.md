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

# HEOM 1c：自旋-热浴模型（欠阻尼情形）

+++

## 引言

HEOM 方法可求解系统及其环境的动力学与稳态；环境效应通过一组辅助密度矩阵编码。

本例展示一个与单个玻色环境接触的二能级系统演化。系统性质由哈密顿量和系统-环境耦合算符决定。

默认玻色环境服从一类特定哈密顿量（[见论文](https://arxiv.org/abs/2010.10806)），其参数通过谱密度及自由热浴关联函数给出。

下面示例演示如何建模欠阻尼布朗运动谱密度。

注意下文取 $\hbar = k_\mathrm{B} = 1$。

### 布朗运动（欠阻尼）谱密度
欠阻尼谱密度为：

$$J_U = \frac{\alpha^2 \Gamma \omega}{(\omega_c^2 - \omega^2)^2 + \Gamma^2 \omega^2)}.$$

其中 $\alpha$ 控制耦合强度，$\Gamma$ 为截止频率，$\omega_c$ 给出共振频率。在 HEOM 中我们需要指数分解形式：

该谱密度的 Matsubara 分解（按实部与虚部）为：



\begin{equation*}
    c_k^R = \begin{cases}
               \alpha^2 \coth(\beta( \Omega + i\Gamma/2)/2)/4\Omega & k = 0\\
               \alpha^2 \coth(\beta( \Omega - i\Gamma/2)/2)/4\Omega & k = 0\\
              -2\alpha^2\Gamma/\beta \frac{\epsilon_k }{((\Omega + i\Gamma/2)^2 + \epsilon_k^2)(\Omega - i\Gamma/2)^2 + \epsilon_k^2)}      & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    \nu_k^R = \begin{cases}
               -i\Omega  + \Gamma/2, i\Omega  +\Gamma/2,             & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}




\begin{equation*}
    c_k^I = \begin{cases}
               i\alpha^2 /4\Omega & k = 0\\
                -i\alpha^2 /4\Omega & k = 0\\
           \end{cases}
\end{equation*}

\begin{equation*}
    \nu_k^I = \begin{cases}
               i\Omega  + \Gamma/2, -i\Omega  + \Gamma/2,             & k = 0\\
           \end{cases}
\end{equation*}

注意在上式及下文中，我们取 $\hbar = k_\mathrm{B} = 1$。

+++

## 设置

```{code-cell} ipython3
import contextlib
import time

import numpy as np
from matplotlib import pyplot as plt

from qutip import (about, basis, brmesolve, destroy, expect, qeye,
                   sigmax, sigmaz, tensor)
from qutip.core.environment import (
    ExponentialBosonicEnvironment, UnderDampedEnvironment
)
from qutip.solver.heom import HEOMSolver

%matplotlib inline
```

## 辅助函数

先定义若干辅助函数，用于计算关联函数展开、绘图和计时。

```{code-cell} ipython3
def cot(x):
    """ Vectorized cotangent of x. """
    return 1.0 / np.tan(x)
```

```{code-cell} ipython3
def coth(x):
    """ Vectorized hyperbolic cotangent of x. """
    return 1.0 / np.tanh(x)
```

```{code-cell} ipython3
def underdamped_matsubara_params(lam, gamma, T, nk):
    """ Calculation of the real and imaginary expansions of the
        underdamped correlation functions.
    """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.0
    beta = 1.0 / T

    ckAR = [
        (lam**2 / (4*Om)) * coth(beta * (Om + 1.0j * Gamma) / 2),
        (lam**2 / (4*Om)) * coth(beta * (Om - 1.0j * Gamma) / 2),
    ]
    ckAR.extend(
        (-2 * lam**2 * gamma / beta) * (2 * np.pi * k / beta) /
        (((Om + 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2) *
         ((Om - 1.0j * Gamma)**2 + (2 * np.pi * k / beta)**2)) + 0.j
        for k in range(1, nk + 1)
    )
    vkAR = [
        -1.0j * Om + Gamma,
        1.0j * Om + Gamma,
    ]
    vkAR.extend(2 * np.pi * k * T + 0.0j for k in range(1, nk + 1))

    factor = 1.0 / 4

    ckAI = [
        -factor * lam**2 * 1.0j / Om,
        factor * lam**2 * 1.0j / Om,
    ]
    vkAI = [
        -(-1.0j * Om - Gamma),
        -(1.0j * Om - Gamma),
    ]

    return ckAR, vkAR, ckAI, vkAI
```

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.

        Each plot in plots consists of: (solver_result, measurement_operation,
        color, label).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
        fig_created = True
    else:
        fig = None
        fig_created = False

    # add kw arguments to each plot if missing
    plots = [p if len(p) == 5 else p + ({},) for p in plots]
    for result, m_op, color, label, kw in plots:
        exp = np.real(expect(result.states, m_op))
        kw.setdefault("linewidth", 2)
        axes.plot(result.times, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=28)

    return fig
```

```{code-cell} ipython3
@contextlib.contextmanager
def timer(label):
    """ Simple utility for timing functions:

        with timer("name"):
            ... code to time ...
    """
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start}")
```

```{code-cell} ipython3
# Solver options:

options = {
    "nsteps": 15000,
    "store_states": True,
    "rtol": 1e-14,
    "atol": 1e-14,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## 系统与热浴定义

下面设置系统哈密顿量、热浴和系统测量算符：

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0.5  # Energy of the 2-level system.
Del = 1.0  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# Initial state of the system.
rho0 = basis(2, 0) * basis(2, 0).dag()
```

```{code-cell} ipython3
# System-bath coupling (underdamed spectral density)
Q = sigmaz()  # coupling operator

# Bath properties:
gamma = 0.1  # cut off frequency
lam = 0.5    # coupling strength
w0 = 1.0     # resonance frequency
T = 1.0
beta = 1.0 / T

# HEOM parameters:

# number of exponents to retain in the Matsubara expansion of the
# bath correlation function:
Nk = 2

# Number of levels of the hierarchy to retain:
NC = 10

# Times to solve for:
tlist = np.linspace(0, 50, 1000)
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

### 先看看欠阻尼谱密度的形状：

```{code-cell} ipython3
def plot_spectral_density():
    """ Plot the underdamped spectral density """
    w = np.linspace(0, 5, 1000)
    J = lam**2 * gamma * w / ((w0**2 - w**2)**2 + (gamma**2) * (w**2))

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(w, J, 'r', linewidth=2)
    axes.set_xlabel(r'$\omega$', fontsize=28)
    axes.set_ylabel(r'J', fontsize=28)


plot_spectral_density()
```

由于谱密度中 Lorentz 峰的存在，关联函数会表现出明显振荡。

+++

### 接下来绘制关联函数本身：

```{code-cell} ipython3
def Mk(t, k, gamma, w0, beta):
    """ Calculate the Matsubara terms for a given t and k. """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.0
    ek = 2 * np.pi * k / beta

    return (
        (-2 * lam**2 * gamma / beta) * ek * np.exp(-ek * np.abs(t))
        / (((Om + 1.0j * Gamma)**2 + ek**2) * ((Om - 1.0j * Gamma)**2 + ek**2))
    )


def c(t, Nk, lam, gamma, w0, beta):
    """ Calculate the correlation function for a vector of times, t. """
    Om = np.sqrt(w0**2 - (gamma / 2)**2)
    Gamma = gamma / 2.0

    Cr = (
        coth(beta * (Om + 1.0j * Gamma) / 2) * np.exp(1.0j * Om * t)
        + coth(beta * (Om - 1.0j * Gamma) / 2) * np.exp(-1.0j * Om * t)
    )

    Ci = np.exp(-1.0j * Om * t) - np.exp(1.0j * Om * t)

    return (
        (lam**2 / (4 * Om)) * np.exp(-Gamma * np.abs(t)) * (Cr + Ci) +
        np.sum([
            Mk(t, k, gamma=gamma, w0=w0, beta=beta)
            for k in range(1, Nk + 1)
        ], 0)
    )


def plot_correlation_function():
    """ Plot the underdamped correlation function. """
    t = np.linspace(0, 20, 1000)
    corr = c(t, Nk=3, lam=lam, gamma=gamma, w0=w0, beta=beta)

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(t, np.real(corr), '-', color="black", label="Re[C(t)]")
    axes.plot(t, np.imag(corr), '-', color="red", label="Im[C(t)]")
    axes.set_xlabel(r't', fontsize=28)
    axes.set_ylabel(r'C', fontsize=28)
    axes.legend(loc=0, fontsize=12)


plot_correlation_function()
```

观察 Matsubara 项对该谱密度的影响很有帮助。可见它主要改变了 $t=0$ 附近的实部。

```{code-cell} ipython3
def plot_matsubara_correlation_function_contributions():
    """ Plot the underdamped correlation function. """
    t = np.linspace(0, 20, 1000)

    M_Nk2 = np.sum(
        [Mk(t, k, gamma=gamma, w0=w0, beta=beta) for k in range(1, 2 + 1)], 0
    )

    M_Nk100 = np.sum(
        [Mk(t, k, gamma=gamma, w0=w0, beta=beta) for k in range(1, 100 + 1)], 0
    )

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(t, np.real(M_Nk2), '-', color="black", label="Re[M(t)] Nk=2")
    axes.plot(t, np.real(M_Nk100), '--', color="red", label="Re[M(t)] Nk=100")
    axes.set_xlabel(r't', fontsize=28)
    axes.set_ylabel(r'M', fontsize=28)
    axes.legend(loc=0, fontsize=12)


plot_matsubara_correlation_function_contributions()
```

## 求解随时间的动力学

+++

接下来用 Matsubara 分解计算指数项，并将其分为实部与虚部。

HEOM 代码会自动优化：当实部与虚部有相同指数时会合并项。vkAI 与 vkAR 列表第一项就是这种情况。

```{code-cell} ipython3
ckAR, vkAR, ckAI, vkAI = underdamped_matsubara_params(
    lam=lam, gamma=gamma, T=T, nk=Nk,
)
```

得到热浴关联函数列表后，构造 `ExponentialBosonicEnvironment` 并将其传给 `HEOMSolver`。

求解器会构建决定系统与辅助密度算符时间演化的“右端项”（RHS），从而用于动力学或稳态求解。

下面创建热浴与求解器，并通过 `.run(rho0, tlist)` 求动力学。

```{code-cell} ipython3
with timer("RHS construction time"):
    env = ExponentialBosonicEnvironment(ckAR, vkAR, ckAI, vkAI)
    HEOMMats = HEOMSolver(Hsys, (env, Q), NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
]);
```

实际应用中通常不会手工做这类欠阻尼关联函数展开，因为 QuTiP 已提供 `UnderDampedEnvironment` 类可直接构建该热浴。
尽管如此，掌握这类展开方法仍是非常有用的技能。


下面展示如何使用该内置功能：

```{code-cell} ipython3
# Compare to built-in under-damped bath:

with timer("RHS construction time"):
    env = UnderDampedEnvironment(lam=lam, gamma=gamma, w0=w0, T=T)
    env_approx = env.approximate("matsubara", Nk=Nk)
    HEOM_udbath = HEOMSolver(Hsys, (env_approx, Q), NC, options=options)

with timer("ODE solver time"):
    result_udbath = HEOM_udbath.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (result_udbath, P11p, 'b', "P11 (UnderDampedEnvironment)"),
    (result_udbath, P12p, 'r', "P12 (UnderDampedEnvironment)"),
]);
```

`UnderDampedEnvironment` 还可方便地计算功率谱、关联函数与谱密度的解析表达。下图中实线为精确表达，虚线为有限指数项近似；在本例中二者符合得非常好。

```{code-cell} ipython3
w = np.linspace(-3, 3, 1000)
w2 = np.linspace(0, 3, 1000)
t = np.linspace(0, 10, 1000)
env_cf = env.correlation_function(t)

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(w, env.power_spectrum(w))
axs[0, 0].plot(w, env_approx.power_spectrum(w), "--")
axs[0, 0].set(xlabel=r"$\omega$", ylabel=r"$S(\omega)$")
axs[0, 1].plot(w2, env.spectral_density(w2))
axs[0, 1].plot(w2, env_approx.spectral_density(w2), "--")
axs[0, 1].set(xlabel=r"$\omega$", ylabel=r"$J(\omega)$")
axs[1, 0].plot(t, np.real(env_cf))
axs[1, 0].plot(t, np.real(env_approx.correlation_function(t)), "--")
axs[1, 0].set(xlabel=r"$t$", ylabel=r"$C_{R}(t)$")
axs[1, 1].plot(t, np.imag(env_cf))
axs[1, 1].plot(t, np.imag(env_approx.correlation_function(t)), "--")
axs[1, 1].set(xlabel=r"$t$", ylabel=r"$C_{I}(t)$")

fig.tight_layout()
plt.show()
```

## 结果比较

+++

### 下面把结果与 QuTiP 的 Bloch-Redfield 求解器进行比较：

```{code-cell} ipython3
with timer("ODE solver time"):
    resultBR = brmesolve(
        Hsys, rho0, tlist,
        a_ops=[[sigmaz(), env]], options=options,
    )
```

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Mats"),
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultBR, P11p, 'g--', "P11 Bloch Redfield"),
    (resultBR, P12p, 'g--', "P12 Bloch Redfield"),
]);
```

### 最后，计算解析稳态结果并把所有方法放在一起比较：

+++

把环境看作单个阻尼模时，其反应坐标的热态在高温且 $\gamma$ 较小时应能给出稳态结果：

```{code-cell} ipython3
dot_energy, dot_state = Hsys.eigenstates()
deltaE = dot_energy[1] - dot_energy[0]

gamma2 = gamma
wa = w0  # reaction coordinate frequency
g = lam / np.sqrt(2 * wa)  # coupling

NRC = 10

Hsys_exp = tensor(qeye(NRC), Hsys)
Q_exp = tensor(qeye(NRC), Q)
a = tensor(destroy(NRC), qeye(2))

H0 = wa * a.dag() * a + Hsys_exp
# interaction
H1 = g * (a.dag() + a) * Q_exp

H = H0 + H1

energies, states = H.eigenstates()
rhoss = 0 * states[0] * states[0].dag()
for kk, energ in enumerate(energies):
    rhoss += states[kk] * states[kk].dag() * np.exp(-beta * energies[kk])
rhoss = rhoss / rhoss.norm()

P12RC = tensor(qeye(NRC), basis(2, 0) * basis(2, 1).dag())
P12RC = expect(rhoss, P12RC)

P11RC = tensor(qeye(NRC), basis(2, 0) * basis(2, 0).dag())
P11RC = expect(rhoss, P11RC)
```

```{code-cell} ipython3
rcParams = {
    "axes.titlesize": 25,
    "axes.labelsize": 30,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
    "legend.fontsize": 28,
    "axes.grid": False,
    "savefig.bbox": "tight",
    "lines.markersize": 5,
    "font.family": "STIXgeneral",
    "mathtext.fontset": "stix",
    "font.serif": "STIX",
    "text.usetex": False,
}
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(12, 7))

with plt.rc_context(rcParams):
    plt.yticks([P11RC, 0.6, 1.0], [0.38, 0.6, 1])

    plot_result_expectations([
        (resultBR, P11p, 'y-.', "Bloch-Redfield"),
        (resultMats, P11p, 'b', "Matsubara $N_k=3$"),
    ], axes=axes)
    axes.plot(
        tlist, [P11RC for t in tlist],
        color='black', linestyle="-.", linewidth=2,
        label="Thermal state",
    )

    axes.set_xlabel(r'$t \Delta$', fontsize=30)
    axes.set_ylabel(r'$\rho_{11}$', fontsize=30)

    axes.locator_params(axis='y', nbins=4)
    axes.locator_params(axis='x', nbins=4)

    axes.legend(loc=0)

    fig.tight_layout()
```

## 关于

```{code-cell} ipython3
about()
```

## 测试

本节可包含测试，以验证笔记本是否生成预期输出。我们把该节放在末尾，以免影响阅读流程。请使用 `assert` 定义测试；当输出错误时，单元应执行失败。

```{code-cell} ipython3
assert np.allclose(
    expect(P11p, resultMats.states[-100:]), P11RC, rtol=1e-2,
)
assert np.allclose(
    expect(P11p, resultBR.states[-100:]), P11RC, rtol=1e-2,
)
```
