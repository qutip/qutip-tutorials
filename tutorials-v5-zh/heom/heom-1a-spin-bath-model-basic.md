---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# HEOM 1a：自旋-热浴模型（入门）

+++

## 引言

HEOM 方法可求解系统及其环境的动力学与稳态，
其中环境效应通过一组辅助密度
矩阵来编码。

本示例展示单个二能级系统在单个玻色环境中的演化。
系统性质由哈密顿量给出，
以及描述系统如何与环境耦合的
耦合算符。

默认玻色环境服从某种特定哈密顿量（[见论文](https://arxiv.org/abs/2010.10806)），其参数体现在谱密度与随后得到的自由热浴关联函数中。

下面示例以 HEOM 中常用的过阻尼 Drude-Lorentz
谱密度为例，展示
Matsubara、Padé 与拟合分解的使用方式，并比较
它们的收敛性。

### Drude-Lorentz（过阻尼）谱密度

Drude-Lorentz 谱密度为：

$$J_D(\omega)= \frac{2\omega\lambda\gamma}{{\gamma}^2 + \omega^2}$$

其中 $\lambda$ 控制耦合强度，$\gamma$ 是截止
频率。我们采用约定
\begin{equation*}
C(t) = \int_0^{\infty} d\omega \frac{J_D(\omega)}{\pi}[\coth(\beta\omega) \
       \cos(\omega \tau) - i \sin(\omega \tau)]
\end{equation*}

在 HEOM 中必须使用指数分解：

\begin{equation*}
C(t)=\sum_{k=0}^{k=\infty} c_k e^{-\nu_k t}
\end{equation*}

例如，Drude-Lorentz 谱密度的 Matsubara 分解
可写为：

\begin{equation*}
    \nu_k = \begin{cases}
               \gamma               & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    c_k = \begin{cases}
               \lambda \gamma (\cot(\beta \gamma / 2) - i) \
               & k = 0\\
               4 \lambda \gamma \nu_k / \{(\nu_k^2 - \gamma^2)\beta \} \
               & k \geq 1\\
           \end{cases}
\end{equation*}

注意在上式及下文中，我们取 $\hbar = k_\mathrm{B} = 1$。

+++

## 设置

```{code-cell} ipython3
import contextlib
import time

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from qutip import (
    about, basis, brmesolve, destroy, expect, liouvillian,
    qeye, sigmax, sigmaz, spost, spre, tensor
)
from qutip.core.environment import (
    DrudeLorentzEnvironment, ExponentialBosonicEnvironment, system_terminator
)
from qutip.solver.heom import HEOMSolver, HSolverDL

%matplotlib inline
```

## 辅助函数

先定义一些辅助函数，用于计算关联函数展开、绘图以及计时：

```{code-cell} ipython3
def cot(x):
    """Vectorized cotangent of x."""
    return 1.0 / np.tan(x)
```

```{code-cell} ipython3
def dl_matsubara_params(lam, gamma, T, nk):
    """Calculation of the real and imaginary expansions of the Drude-Lorenz
    correlation functions.
    """
    ckAR = [lam * gamma * cot(gamma / (2 * T))]
    ckAR.extend(
        8 * lam * gamma * T * np.pi * k * T / (
            (2 * np.pi * k * T) ** 2 - gamma**2
        )
        for k in range(1, nk + 1)
    )
    vkAR = [gamma]
    vkAR.extend(2 * np.pi * k * T for k in range(1, nk + 1))

    ckAI = [lam * gamma * (-1.0)]
    vkAI = [gamma]

    return ckAR, vkAR, ckAI, vkAI
```

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """Plot the expectation values of operators as functions of time.

    Each plot in plots consists of (solver_result, measurement_operation,
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
    """Simple utility for timing functions:

    with timer("name"):
        ... code to time ...
    """
    start = time.time()
    yield
    end = time.time()
    print(f"{label}: {end - start}")
```

```{code-cell} ipython3
# Default solver options:

default_options = {
    "nsteps": 1500,
    "store_states": True,
    "rtol": 1e-12,
    "atol": 1e-12,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## 系统与热浴定义

下面设置系统哈密顿量、热浴以及系统测量算符：

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
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz()  # coupling operator

# Bath properties:
gamma = 0.5  # cut off frequency
lam = 0.1  # coupling strength
T = 0.5
beta = 1.0 / T

# HEOM parameters
NC = 5  # cut off parameter for the bath
Nk = 2  # terms in the Matsubara expansion of the correlation function

# Times to solve for
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

### 首先，先看一下谱密度

现在可以开始了。先根据前面参数看看谱密度的形状：

```{code-cell} ipython3
def plot_spectral_density():
    """Plot the Drude-Lorentz spectral density"""
    w = np.linspace(0, 5, 1000)
    J = w * 2 * lam * gamma / (gamma**2 + w**2)

    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
    axes.plot(w, J, "r", linewidth=2)
    axes.set_xlabel(r"$\omega$", fontsize=28)
    axes.set_ylabel(r"J", fontsize=28)


plot_spectral_density()
```

接下来用 Matsubara 分解计算指数项，这里
把它们拆分为实部和虚部。

HEOM 代码会自动优化：当
实部与虚部具有相同指数时会合并项。
在 vkAI 与 vkAR 列表第一项中就明显如此。

```{code-cell} ipython3
ckAR, vkAR, ckAI, vkAI = dl_matsubara_params(nk=Nk, lam=lam, gamma=gamma, T=T)
```

构造好描述热浴关联函数的列表后，
可由其创建 `ExponentialBosonicEnvironment`，并将环境传给 `HEOMSolver`。

求解器会构建决定
系统与辅助密度算符随时间演化的“右端项”（RHS）。随后即可
用它求动力学或稳态。

下面创建热浴和求解器，并通过
调用 `.run(rho0, tlist)` 求解动力学。

```{code-cell} ipython3
options = {**default_options}

with timer("RHS construction time"):
    env = ExponentialBosonicEnvironment(ckAR, vkAR, ckAI, vkAI)
    HEOMMats = HEOMSolver(Hsys, (env, Q), NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultMats, P11p, "b", "P11 Mats"),
        (resultMats, P12p, "r", "P12 Mats"),
    ]
);
```

在实际工作中，通常不会手工做这类对
Drude-Lorentz 关联函数的繁琐展开，因为 QuTiP 已提供
`DrudeLorentzEnvironment` 类可直接构造该热浴。尽管如此，
掌握这种展开方法能帮助你为其他谱密度构建
自定义热浴。

下面展示如何使用这一内置功能：

```{code-cell} ipython3
# Compare to built-in Drude-Lorentz bath:

with timer("RHS construction time"):
    # Abstract representation of D-L Environment
    dlenv = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T)
    # Matsubara approximation of D-L Environment
    dlenv_approx = dlenv.approximate(method="matsubara", Nk=Nk)
    HEOM_dlbath = HEOMSolver(Hsys, (dlenv_approx, Q), NC, options=options)

with timer("ODE solver time"):
    result_dlbath = HEOM_dlbath.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (result_dlbath, P11p, "b", "P11 (DrudeLorentzEnvironment)"),
        (result_dlbath, P12p, "r", "P12 (DrudeLorentzEnvironment)"),
    ]
);
```

`DrudeLorentzEnvironment` 还可方便获得功率谱、关联函数和谱密度。其近似环境对象是 `BosonicEnvironment`，可访问近似关联函数及其对应的有效功率谱与谱密度。下图中实线为精确表达，虚线为采用有限指数的 Matsubara 关联函数近似。

`DrudeLorentzEnvironment` 的精确关联函数计算使用 Padé 近似。默认 Padé 项数是 $10$，但在低温条件下 $10$ 项可能不足。下文会补充 Padé 的更多说明。下面继续展示该内置用法：

```{code-cell} ipython3
w = np.linspace(-10, 20, 1000)
w2 = np.linspace(0, 20, 1000)

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(w, dlenv.power_spectrum(w))
axs[0, 0].plot(w, dlenv_approx.power_spectrum(w), "--")
axs[0, 0].set(xlabel=r"$\omega$", ylabel=r"$S(\omega)$")
axs[0, 1].plot(w2, dlenv.spectral_density(w2))
axs[0, 1].plot(w2, dlenv_approx.spectral_density(w2), "--")
axs[0, 1].set(xlabel=r"$\omega$", ylabel=r"$J(\omega)$")
axs[1, 0].plot(w2, np.real(dlenv.correlation_function(w2, Nk=100)))  # 100 Pade
axs[1, 0].plot(w2, np.real(dlenv_approx.correlation_function(w2)), "--")
axs[1, 0].set(xlabel=r"$t$", ylabel=r"$C_{R}(t)$")
axs[1, 1].plot(w2, np.imag(dlenv.correlation_function(w2, Nk=100)))
axs[1, 1].plot(w2, np.imag(dlenv_approx.correlation_function(w2)), "--")
axs[1, 1].set(xlabel=r"$t$", ylabel=r"$C_{I}(t)$")

fig.tight_layout()
plt.show()
```

我们还提供了一个兼容旧接口的遗留类 `HSolverDL`，可自动计算
Drude-Lorentz 关联函数，以保持与 QuTiP 早期 HEOM 求解器
的向后兼容：

```{code-cell} ipython3
# Compare to legacy class:

# The legacy class performs the above collation of coefficients automatically,
# based upon the parameters for the Drude-Lorentz spectral density.

with timer("RHS construction time"):
    HEOMlegacy = HSolverDL(Hsys, Q, lam, T, NC, Nk, gamma, options=options)

with timer("ODE solver time"):
    resultLegacy = HEOMlegacy.run(rho0, tlist)  # normal  115
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultLegacy, P11p, "b", "P11 Legacy"),
        (resultLegacy, P12p, "r", "P12 Legacy"),
    ]
);
```

另一个保留用于便利的遗留类是 `DrudeLorentzBath`。下面这段代码
```python
dlenv = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T)
dlenv_approx = dlenv.approximate(method="matsubara", Nk=Nk)  # Computes Matsubara exponents
HEOM_dlbath = HEOMSolver(Hsys, (dlenv_approx, Q), NC, options=options)
```
与我们上面使用的写法是等价的：
```python
dlbath = DrudeLorentzBath(Q, lam=lam, gamma=gamma, T=T, Nk=Nk)
HEOM_dlbath = HEOMSolver(Hsys, dlbath, NC, options=options)
```

+++

## Ishizaki-Tanimura 终止子

为了加快收敛（即减少 Matsubara 展开中需保留的指数项数），
我们可把 $Re[C(t=0)]$ 分量近似为
delta 分布，并把它作为 Lindblad 修正项并入。这个技巧
通常称为 Ishizaki-Tanimura 终止子。

更具体地，给定

\begin{equation*}
C(t)=\sum_{k=0}^{\infty} c_k e^{-\nu_k t}
\end{equation*}

由于 $\nu_k=\frac{2 \pi k}{\beta }$，当 $1/\nu_k$ 远小于
其它重要时间尺度时，可近似
$ e^{-\nu_k t} \approx \delta(t)/\nu_k$，从而 $C(t)=\sum_{k=N_k}^{\infty}
\frac{c_k}{\nu_k} \delta(t)$

一个方便做法是先计算整体求和
$ C(t)=\sum_{k=0}^{\infty} \frac{c_k}{\nu_k} =  2 \lambda / (\beta \gamma)- i\lambda $
，再减去层级中已保留的有限 Matsubara 项贡献
，把剩余部分作为
Lindblad 形式的修正项处理。

如果先画出包含大量 Matsubara 项的关联函数，这一做法会更直观。
绘图时可使用前文提到的
`DrudeLorentzEnvironment` 工具函数。

```{code-cell} ipython3
def plot_correlation_expansion_divergence():
    """We plot the correlation function with a large number of Matsubara terms
    to show that the real part is slowly diverging at t = 0.
    """
    t = np.linspace(0, 2, 100)

    # correlation coefficients with 100 pade and with 2 matsubara terms
    corr_100 = dlenv.correlation_function(t, Nk=100)
    corr_2 = dlenv_approx.correlation_function(t)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(
        t, np.real(corr_2), color="b", linewidth=3, label=rf"Mats = {Nk} real"
    )
    ax1.plot(
        t, np.imag(corr_2), color="r", linewidth=3, label=rf"Mats = {Nk} imag"
    )
    ax1.plot(
        t, np.real(corr_100), "b--", linewidth=3, label=r"Pade = 100 real"
    )
    ax1.plot(
        t, np.imag(corr_100), "r--", linewidth=3, label=r"Pade = 100 imag"
    )

    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$C$")
    ax1.legend()


plot_correlation_expansion_divergence()
```

下面评估加入 Ishizaki-Tanimura 终止子后的结果：

```{code-cell} ipython3
# Run HEOM solver including the Ishizaki-Tanimura terminator

# Notes:
#
# * here, we will first show how to compute the terminator manually
#
# * when using the built-in DrudeLorentzEnvironment the terminator (L_bnd) is
#   available from by setting the parameter compute_delta to True in the
#   approximate method
#
# * in the legacy HSolverDL function the terminator is included automatically
#   if the parameter bnd_cut_approx=True is used.

op = -2 * spre(Q) * spost(Q.dag()) + spre(Q.dag() * Q) + spost(Q.dag() * Q)

approx_factr = (2 * lam / (beta * gamma)) - 1j * lam

approx_factr -= lam * gamma * (-1.0j + cot(gamma / (2 * T))) / gamma
for k in range(1, Nk + 1):
    vk = 2 * np.pi * k * T

    approx_factr -= (4 * lam * gamma * T * vk / (vk**2 - gamma**2)) / vk

L_bnd = -approx_factr * op

Ltot = -1.0j * (spre(Hsys) - spost(Hsys)) + L_bnd
Ltot = liouvillian(Hsys) + L_bnd

options = {**default_options, "rtol": 1e-14, "atol": 1e-14}

with timer("RHS construction time"):
    env = ExponentialBosonicEnvironment(ckAR, vkAR, ckAI, vkAI)
    HEOMMatsT = HEOMSolver(Ltot, (env, Q), NC, options=options)

with timer("ODE solver time"):
    resultMatsT = HEOMMatsT.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultMatsT, P11p, "b", "P11 Mats + Term"),
        (resultMatsT, P12p, "r", "P12 Mats + Term"),
    ]
);
```

若使用内置 Drude-Lorentz 环境，可简写为：

```{code-cell} ipython3
options = {**default_options, "rtol": 1e-14, "atol": 1e-14}

with timer("RHS construction time"):
    dlenv_approx, delta = dlenv.approximate(
        "matsubara", Nk=Nk, compute_delta=True
    )
    Ltot = liouvillian(Hsys) + system_terminator(Q, delta)
    HEOM_dlbath_T = HEOMSolver(Ltot, (dlenv_approx, Q), NC, options=options)

with timer("ODE solver time"):
    result_dlbath_T = HEOM_dlbath_T.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations([
    (result_dlbath_T, P11p, "b", "P11 Mats (DrudeLorentzEnvironment + Term)"),
    (result_dlbath_T, P12p, "r", "P12 Mats (DrudeLorentzEnvironment + Term)"),
]);
```

我们还可以与 QuTiP Bloch-Redfield 求解器的解进行比较：

```{code-cell} ipython3
options = {**default_options}

with timer("ODE solver time"):
    resultBR = brmesolve(
        Hsys, rho0, tlist, a_ops=[[sigmaz(), dlenv]], options=options
    )
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultMats, P11p, "b", "P11 Mats"),
        (resultMats, P12p, "r", "P12 Mats"),
        (resultMatsT, P11p, "b--", "P11 Mats + Term"),
        (resultMatsT, P12p, "r--", "P12 Mats + Term"),
        (resultBR, P11p, "g--", "P11 Bloch Redfield"),
        (resultBR, P12p, "g--", "P12 Bloch Redfield"),
    ]
);
```

## Padé 分解

+++

Matsubara 分解并非唯一选择，也可以使用
收敛更快的 Padé 分解。

```{code-cell} ipython3
def deltafun(j, k):
    if j == k:
        return 1.0
    else:
        return 0.0


def pade_eps(lmax):
    Alpha = np.zeros((2 * lmax, 2 * lmax))
    for j in range(2 * lmax):
        for k in range(2 * lmax):
            # Fermionic (see other example notebooks):
            #   Alpha[j][k] = (deltafun(j, k+1) + deltafun(j, k-1))
            #                 / sqrt((2 * (j + 1) - 1) * (2 * (k + 1) - 1))
            # Bosonic:
            Alpha[j][k] = (deltafun(j, k + 1) + deltafun(j, k - 1)) / np.sqrt(
                (2 * (j + 1) + 1) * (2 * (k + 1) + 1)
            )

    eigvalsA = np.linalg.eigvalsh(Alpha)
    eps = [-2 / val for val in eigvalsA[0:lmax]]
    return eps


def pade_chi(lmax):
    AlphaP = np.zeros((2 * lmax - 1, 2 * lmax - 1))
    for j in range(2 * lmax - 1):
        for k in range(2 * lmax - 1):
            # Fermionic:
            #   AlphaP[j][k] = (deltafun(j, k + 1) + deltafun(j, k - 1))
            #                  / sqrt((2 * (j + 1) + 1) * (2 * (k + 1) + 1))
            # Bosonic [this is +3 because +1 (bose) + 2*(+1) (from bm+1)]:
            AlphaP[j][k] = (deltafun(j, k + 1) + deltafun(j, k - 1)) / np.sqrt(
                (2 * (j + 1) + 3) * (2 * (k + 1) + 3)
            )

    eigvalsAP = np.linalg.eigvalsh(AlphaP)
    chi = [-2 / val for val in eigvalsAP[0:(lmax - 1)]]
    return chi


def pade_kappa_epsilon(lmax):
    eps = pade_eps(lmax)
    chi = pade_chi(lmax)

    kappa = [0]
    prefactor = 0.5 * lmax * (2 * (lmax + 1) + 1)

    for j in range(lmax):
        term = prefactor
        for k in range(lmax - 1):
            term *= (chi[k] ** 2 - eps[j] ** 2) / (
                eps[k] ** 2 - eps[j] ** 2 + deltafun(j, k)
            )

        for k in range(lmax - 1, lmax):
            term /= eps[k] ** 2 - eps[j] ** 2 + deltafun(j, k)

        kappa.append(term)

    epsilon = [0] + eps

    return kappa, epsilon


def pade_corr(tlist, lmax):
    kappa, epsilon = pade_kappa_epsilon(lmax)

    eta_list = [lam * gamma * (cot(gamma * beta / 2.0) - 1.0j)]
    gamma_list = [gamma]

    if lmax > 0:
        for ll in range(1, lmax + 1):
            eta_list.append(
                (kappa[ll] / beta)
                * 4
                * lam
                * gamma
                * (epsilon[ll] / beta)
                / ((epsilon[ll] ** 2 / beta**2) - gamma**2)
            )
            gamma_list.append(epsilon[ll] / beta)

    c_tot = []
    for t in tlist:
        c_tot.append(
            sum(
                [
                    eta_list[ll] * np.exp(-gamma_list[ll] * t)
                    for ll in range(lmax + 1)
                ]
            )
        )
    return c_tot, eta_list, gamma_list


tlist_corr = np.linspace(0, 2, 100)
cppLP, etapLP, gampLP = pade_corr(tlist_corr, 2)
corr_100 = dlenv.correlation_function(tlist_corr, Nk=100)
corr_2 = dlenv.approximate("matsubara", Nk=2).correlation_function(tlist_corr)

fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(
    tlist_corr,
    np.real(cppLP),
    color="b",
    linewidth=3,
    label=r"real pade 2 terms",
)
ax1.plot(
    tlist_corr,
    np.real(corr_100),
    "r--",
    linewidth=3,
    label=r"real pade 100 terms",
)
ax1.plot(
    tlist_corr,
    np.real(corr_2),
    "g--",
    linewidth=3,
    label=r"real mats 2 terms",
)

ax1.set_xlabel("t")
ax1.set_ylabel(r"$C_{R}(t)$")
ax1.legend()

fig, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(
    tlist_corr,
    np.real(cppLP) - np.real(corr_100),
    color="b",
    linewidth=3,
    label=r"pade error",
)
ax1.plot(
    tlist_corr,
    np.real(corr_2) - np.real(corr_100),
    "r--",
    linewidth=3,
    label=r"mats error",
)

ax1.set_xlabel("t")
ax1.set_ylabel(r"Error")
ax1.legend();
```

```{code-cell} ipython3
# put pade parameters in lists for heom solver
ckAR = [np.real(eta) + 0j for eta in etapLP]
ckAI = [np.imag(etapLP[0]) + 0j]
vkAR = [gam + 0j for gam in gampLP]
vkAI = [gampLP[0] + 0j]

options = {**default_options, "rtol": 1e-14, "atol": 1e-14}

with timer("RHS construction time"):
    bath = ExponentialBosonicEnvironment(ckAR, vkAR, ckAI, vkAI)
    HEOMPade = HEOMSolver(Hsys, (bath, Q), NC, options=options)

with timer("ODE solver time"):
    resultPade = HEOMPade.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultMats, P11p, "b", "P11 Mats"),
        (resultMats, P12p, "r", "P12 Mats"),
        (resultMatsT, P11p, "y", "P11 Mats + Term"),
        (resultMatsT, P12p, "g", "P12 Mats + Term"),
        (resultPade, P11p, "b--", "P11 Pade"),
        (resultPade, P12p, "r--", "P12 Pade"),
    ]
);
```

如前所述，Drude-Lorentz 热浴的 Padé 分解也可通过
内置 `DrudeLorentzEnvironment` 获得。类似终止子
章节，在 Padé 近似下也可通过请求近似函数计算 delta
项来轻松得到终止子。

下面展示如何使用内置 Drude-Lorentz 环境获得
Padé 分解近似及其终止子（这里终止子
改进不明显，因为 Padé 展开本身已经把
关联函数拟合得很好）：

```{code-cell} ipython3
options = {**default_options, "rtol": 1e-14, "atol": 1e-14}

with timer("RHS construction time"):
    env_approx, delta = dlenv.approximate("pade", Nk=2, compute_delta=True)
    Ltot = liouvillian(Hsys) + system_terminator(Q, delta)
    HEOM_dlpbath_T = HEOMSolver(Ltot, (env_approx, Q), NC, options=options)

with timer("ODE solver time"):
    result_dlpbath_T = HEOM_dlpbath_T.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (result_dlpbath_T, P11p, "b", "P11 Pad茅 + Term"),
        (result_dlpbath_T, P12p, "r", "P12 Pad茅 + Term"),
    ]
);
```

### 下面比较 Matsubara 与 Padé 的关联函数拟合

在本例中直接拟合关联函数并不高效，但
在需要大量指数项的场景（例如接近零温）
会非常有用。下面先手工拟合
再展示内置工具做法。

手工拟合前，先收集较大量 Padé 项并在
多个时间点上求和：

```{code-cell} ipython3
tlist2 = np.linspace(0, 2, 10000)

corr_100_t10k = dlenv.correlation_function(tlist2, Nk=100)
# Nk specifies the number of pade terms to be used for the correlation function

corrRana = np.real(corr_100_t10k)
corrIana = np.imag(corr_100_t10k)

corrRMats = np.real(dlenv_approx.correlation_function(tlist2))
```

然后用标准最小二乘方法进行拟合：

```{code-cell} ipython3
def wrapper_fit_func(x, N, args):
    """ Fit function wrapper that unpacks its arguments. """
    x = np.array(x)
    a = np.array(args[:N])
    b = np.array(args[N:(2 * N)])
    return fit_func(x, a, b)


def fit_func(x, a, b):
    """ Fit function. Calculates the value of the
        correlation function at each x, given the
        fit parameters in a and b.
    """
    return np.sum(
        a[:, None] * np.exp(np.multiply.outer(b, x)),
        axis=0,
    )


def fitter(ans, tlist, k):
    """ Compute fit with k exponents. """
    upper_a = abs(max(ans, key=abs)) * 10
    # sets initial guesses:
    guess = (
        [upper_a / k] * k +  # guesses for a
        [0] * k  # guesses for b
    )
    # sets lower bounds:
    b_lower = (
        [-upper_a] * k +  # lower bounds for a
        [-np.inf] * k  # lower bounds for b
    )
    # sets higher bounds:
    b_higher = (
        [upper_a] * k +  # upper bounds for a
        [0] * k  # upper bounds for b
    )
    param_bounds = (b_lower, b_higher)
    p1, p2 = curve_fit(
        lambda x, *params_0: wrapper_fit_func(x, k, params_0),
        tlist,
        ans,
        p0=guess,
        sigma=[0.01 for t in tlist],
        bounds=param_bounds,
        maxfev=1e8,
    )
    a, b = p1[:k], p1[k:]
    return (a, b)
```

```{code-cell} ipython3
kR = 4  # number of exponents to use for real part
poptR = []
with timer("Correlation (real) fitting time"):
    for i in range(kR):
        poptR.append(fitter(corrRana, tlist2, i + 1))

kI = 1  # number of exponents for imaginary part
poptI = []
with timer("Correlation (imaginary) fitting time"):
    for i in range(kI):
        poptI.append(fitter(corrIana, tlist2, i + 1))
```

并绘制拟合结果：

```{code-cell} ipython3
# Define line styles and colors
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
colors = ["blue", "green", "purple", "orange", "red", "brown"]

# Define a larger linewidth
linewidth = 2.5

# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the real part on the first subplot (ax1)
ax1.plot(tlist2, corrRana, label="Analytic", color=colors[0],
         linestyle=linestyles[0], linewidth=linewidth)
ax1.plot(tlist2, corrRMats, label="Matsubara", color=colors[1],
         linestyle=linestyles[1], linewidth=linewidth)

for i in range(kR):
    y = fit_func(tlist2, *poptR[i])
    ax1.plot(tlist2, y, label=f"Fit with {i+1} terms", color=colors[i + 2],
             linestyle=linestyles[i + 2], linewidth=linewidth)

ax1.set_ylabel(r"$C_{R}(t)$")
ax1.set_xlabel(r"$t$")
ax1.legend()

# Plot the imaginary part on the second subplot (ax2)
ax2.plot(tlist2, corrIana, label="Analytic", color=colors[0],
         linestyle=linestyles[0], linewidth=linewidth)

for i in range(kI):
    y = fit_func(tlist2, *poptI[i])
    ax2.plot(tlist2, y, label=f"Fit with {i+1} terms", color=colors[i + 3],
             linestyle=linestyles[i + 3], linewidth=linewidth)

ax2.set_ylabel(r"$C_{I}(t)$")
ax2.set_xlabel(r"$t$")
ax2.legend()

# Add overall plot title and show the figure
fig.suptitle(
    "Comparison of Analytic and Fit to Correlations"
    " (Real and Imaginary Parts)",
    fontsize=16,
)
plt.show()
```

```{code-cell} ipython3
# Set the exponential coefficients from the fit parameters

ckAR1 = poptR[-1][0]
ckAR = [x + 0j for x in ckAR1]

vkAR1 = poptR[-1][1]
vkAR = [-x + 0j for x in vkAR1]

ckAI1 = poptI[-1][0]
ckAI = [x + 0j for x in ckAI1]

vkAI1 = poptI[-1][1]
vkAI = [-x + 0j for x in vkAI1]
```

```{code-cell} ipython3
# overwrite imaginary fit with analytical value (not much reason to use the
# fit for this)

ckAI = [lam * gamma * (-1.0) + 0.0j]
vkAI = [gamma + 0.0j]
```

```{code-cell} ipython3
options = {**default_options}

NC = 4

with timer("RHS construction time"):
    bath = ExponentialBosonicEnvironment(ckAR, vkAR, ckAI, vkAI)
    HEOMFit = HEOMSolver(Hsys, (bath, Q), NC, options=options)

with timer("ODE solver time"):
    resultFit = HEOMFit.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultFit, P11p, "b", "P11 Fit"),
        (resultFit, P12p, "r", "P12 Fit"),
    ]
);
```

接着使用内置拟合函数。`BosonicEnvironment` 类包含
可自动执行该拟合的方法。关于
内置函数的更多细节见 `HEOM 1d: Spin-Bath model, fitting of spectrum and correlation functions`

```{code-cell} ipython3
max_val = dlenv.correlation_function(0).real
guess = [max_val / 3, 0, 0, 0]
lower = [-max_val, -np.inf, -np.inf, -np.inf]
upper = [max_val, 0, 0, 0]
envfit, fitinfo = dlenv.approximate(
    "cf", tlist=tlist2, full_ansatz=True, Ni_max=1, Nr_max=3,
    upper=upper, lower=lower, guess=guess)
```

`approximate("cf", ...)` 方法返回 `ExponentialBosonicEnvironment` 对象，
其中包含原始
环境的衰减指数表示，以及一个记录拟合信息的字典。
该字典给出拟合摘要及归一化
均方根误差（用于评估拟合质量）。

```{code-cell} ipython3
print(fitinfo["summary"])
```

随后可将内置拟合结果与手工拟合进行比较

```{code-cell} ipython3
# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the real part on the first subplot (ax1)
ax1.plot(tlist2, corrRana, label="Original", marker="o", markevery=500)
ax1.plot(tlist2, fit_func(tlist2, *poptR[-1]), color="r", label="Manual Fit")
ax1.plot(tlist2, np.real(envfit.correlation_function(tlist2)), "k--",
         label="Built-in fit")
ax1.set_ylabel(r"$C_{R}(t)$")
ax1.set_xlabel(r"$t$")
ax1.legend()

# Plot the imaginary part on the second subplot (ax2)
ax2.plot(tlist2, corrIana, label="Original", marker="o", markevery=500)
ax2.plot(tlist2, fit_func(tlist2, *poptI[-1]), color="r", label="Manual Fit")
ax2.plot(tlist2, np.imag(envfit.correlation_function(tlist2)), "k--",
         label="Built-in fit")
ax2.set_ylabel(r"$C_{I}(t)$")
ax2.set_xlabel(r"$t$")
ax2.legend()

# Adjust layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

```{code-cell} ipython3
options = {**default_options}

with timer("RHS construction time"):
    HEOMFit_2 = HEOMSolver(Hsys, (envfit, Q), NC, options=options)

with timer("ODE solver time"):
    resultFit_2 = HEOMFit_2.run(rho0, tlist)
```

```{code-cell} ipython3
plot_result_expectations(
    [
        (resultFit, P11p, "b", "P11 Fit"),
        (resultFit, P12p, "r", "P12 Fit"),
        (resultFit_2, P11p, "r--", "P11 Built-in-Fit"),
        (resultFit_2, P12p, "b--", "P12 Built-in-Fit"),
    ]
);
```

## 反应坐标方法

+++

这里构建一个受反应坐标启发的模型来刻画
稳态行为，并与 HEOM 预测比较。该结果
在窄谱密度情况下更准确。我们会把这一节的布居与
相干结果用于下方最终汇总图。

```{code-cell} ipython3
dot_energy, dot_state = Hsys.eigenstates()
deltaE = dot_energy[1] - dot_energy[0]

gamma2 = deltaE / (2 * np.pi * gamma)
wa = 2 * np.pi * gamma2 * gamma  # reaction coordinate frequency
g = np.sqrt(np.pi * wa * lam / 2.0)  # reaction coordinate coupling
# reaction coordinate coupling factor over 2 because of diff in J(w)
# (it is 2 lam now):
g = np.sqrt(np.pi * wa * lam / 4.0)  #

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


class ReactionCoordinateResult:
    def __init__(self, states, times):
        self.states = states
        self.times = times


resultRC = ReactionCoordinateResult([rhoss] * len(tlist), tlist)

P12RC = tensor(qeye(NRC), basis(2, 0) * basis(2, 1).dag())
P11RC = tensor(qeye(NRC), basis(2, 0) * basis(2, 0).dag())
```

## 汇总绘制所有结果

最后，把各方法结果统一绘出，看看彼此对比情况。

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
fig, axes = plt.subplots(2, 1, sharex=False, figsize=(12, 15))

with plt.rc_context(rcParams):

    plt.sca(axes[0])
    plt.yticks([expect(P11RC, resultRC.states[0]), 0.6, 1.0], [0.32, 0.6, 1])
    plot_result_expectations(
        [
            (resultBR, P11p, "y-.", "Bloch-Redfield"),
            (resultMats, P11p, "b", "Matsubara $N_k=2$"),
            (
                resultMatsT,
                P11p,
                "g--",
                "Matsubara $N_k=2$ & Terminator",
                {"linewidth": 3},
            ),
            (
                resultFit,
                P11p,
                "r",
                r"Fit $N_f = 4$, Pade $N_k=100$",
                {"dashes": [3, 2]},
            ),
            (
                resultRC,
                P11RC,
                "--",
                "Thermal",
                {"linewidth": 2, "color": "black"},
            ),
        ],
        axes=axes[0],
    )
    axes[0].set_ylabel(r"$\rho_{11}$", fontsize=30)
    axes[0].legend(loc=0)
    axes[0].text(5, 0.9, "(a)", fontsize=30)
    axes[0].set_xlim(0, 50)

    plt.sca(axes[1])
    plt.yticks(
        [np.real(expect(P12RC, resultRC.states[0])), -0.2, 0.0, 0.2],
        [-0.33, -0.2, 0, 0.2],
    )
    plot_result_expectations(
        [
            (resultBR, P12p, "y-.", "Bloch-Redfield"),
            (resultMats, P12p, "b", "Matsubara $N_k=2$"),
            (
                resultMatsT,
                P12p,
                "g--",
                "Matsubara $N_k=2$ & Terminator",
                {"linewidth": 3},
            ),
            (
                resultFit,
                P12p,
                "r",
                r"Fit $N_f = 4$, Pade $N_k=100$",
                {"dashes": [3, 2]},
            ),
            (
                resultRC,
                P12RC,
                "--",
                "Thermal",
                {"linewidth": 2, "color": "black"},
            ),
        ],
        axes=axes[1],
    )
    axes[1].text(5, 0.1, "(b)", fontsize=30)
    axes[1].set_xlabel(r"$t \Delta$", fontsize=30)
    axes[1].set_ylabel(r"$\rho_{01}$", fontsize=30)
    axes[1].set_xlim(0, 50)
```

到这里，我们完成了对 HEOM 玻色环境建模的一次较完整入门。

+++

## 关于

```{code-cell} ipython3
about()
```

## 测试

本节可包含测试，以验证笔记本是否生成预期输出。我们把该节放在末尾，以免影响阅读流程。请使用 `assert` 定义测试；当输出错误时，单元应执行失败。

```{code-cell} ipython3
# Check P11p
assert np.allclose(
    expect(P11p, resultMatsT.states),
    expect(P11p, resultPade.states),
    rtol=1e-2,
)
assert np.allclose(
    expect(P11p, resultMatsT.states),
    expect(P11p, resultFit.states),
    rtol=1e-2,
)

# Check P12p
assert np.allclose(
    expect(P12p, resultMatsT.states),
    expect(P12p, resultPade.states),
    rtol=1e-2,
)
assert np.allclose(
    expect(P12p, resultMatsT.states),
    expect(P12p, resultFit.states),
    rtol=1e-1,
)
```
