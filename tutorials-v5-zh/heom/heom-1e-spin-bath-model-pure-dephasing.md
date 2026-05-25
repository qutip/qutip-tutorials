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

# HEOM 1e：自旋-热浴模型（纯退相干）

+++

## 引言

HEOM 方法可求解系统及其环境的动力学与稳态；环境部分通过辅助密度矩阵编码。

本例展示与单个玻色环境耦合的二能级系统演化。系统性质由哈密顿量和系统-环境耦合算符决定。

默认玻色环境服从特定哈密顿量（见相关论文），其参数通过谱密度及自由热浴关联函数给出。

下例继续使用 HEOM 常见的过阻尼 Drude-Lorentz 谱密度。我们展示 Matsubara 与 Padé 的解析分解，以及如何用有限指数集合拟合后者。与 1a 的差异在于：这里假设系统项与耦合项对易，从而得到可解析求解的“纯退相干”模型。该模型很适合用来检验其它近似（例如关联函数拟合）的有效性与收敛性。（一般而言，对拟合方法来说纯退相干是“最困难情形”之一。）

### Drude-Lorentz 谱密度

Drude-Lorentz 谱密度为：

$$J(\omega)=\omega \frac{2\lambda\gamma}{{\gamma}^2 + \omega^2}$$

其中 $\lambda$ 控制耦合强度，$\gamma$ 为截止频率。
我们采用约定
\begin{equation*}
C(t) = \int_0^{\infty} d\omega \frac{J_D(\omega)}{\pi}[\coth(\beta\omega) \cos(\omega \tau) - i \sin(\omega \tau)]
\end{equation*}

在 HEOM 中必须使用指数分解：

\begin{equation*}
C(t)=\sum_{k=0}^{k=\infty} c_k e^{-\nu_k t}
\end{equation*}

Drude-Lorentz 谱密度的 Matsubara 分解为：

\begin{equation*}
    \nu_k = \begin{cases}
               \gamma               & k = 0\\
               {2 \pi k} / {\beta \hbar}  & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    c_k = \begin{cases}
               \lambda \gamma (\cot(\beta \gamma / 2) - i) / \hbar               & k = 0\\
               4 \lambda \gamma \nu_k / \{(nu_k^2 - \gamma^2)\beta \hbar^2 \}    & k \geq 1\\
           \end{cases}
\end{equation*}

注意在上式及下文中，我们取 $\hbar = k_\mathrm{B} = 1$。

+++

## 设置

```{code-cell} ipython3
import contextlib
import time

import numpy as np
import scipy
from matplotlib import pyplot as plt

from qutip import about, basis, expect, liouvillian, sigmax, sigmaz
from qutip.core.environment import DrudeLorentzEnvironment, system_terminator
from qutip.solver.heom import HEOMSolver

%matplotlib inline
```

## 辅助函数

先定义若干辅助函数，用于计算关联函数展开、绘图和计时。

```{code-cell} ipython3
def cot(x):
    """ Vectorized cotangent of x. """
    return 1.0 / np.tan(x)


def coth(x):
    """ Vectorized hyperbolic cotangent of x. """
    return 1.0 / np.tanh(x)
```

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.

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
        if m_op is None:
            t, exp = result
        else:
            t = result.times
            exp = np.real(expect(result.states, m_op))
        kw.setdefault("linewidth", 2)
        axes.plot(t, exp, color, label=label, **kw)

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

下面设置系统哈密顿量、热浴与系统测量算符：

+++

这里设定 $H_{sys}=0$，因此相互作用哈密顿量与系统哈密顿量对易，可将数值结果与已知解析解比较。原则上可取 $\epsilon\neq0$，但那只会引入快速系统振荡，所以这里设为 0 更方便。

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0.0  # Energy of the 2-level system.
Del = 0.0  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz()  # coupling operator

# Bath properties:
gamma = 0.5  # cut off frequency
lam = 0.1  # coupling strength
T = 0.5
beta = 1.0 / T

# HEOM parameters:
# cut off parameter for the bath:
NC = 6
# number of exponents to retain in the Matsubara expansion
# of the correlation function:
Nk = 3

# Times to solve for
tlist = np.linspace(0, 50, 1000)
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresponding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresponding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

为了得到非平凡结果，我们把初态准备为叠加态，观察热浴如何破坏相干。

```{code-cell} ipython3
# Initial state of the system.
psi = (basis(2, 0) + basis(2, 1)).unit()
rho0 = psi * psi.dag()
```

随后定义环境对象，后续所有模拟都从该环境出发。


```{code-cell} ipython3
env = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T, Nk=Nk)
```

## 模拟 1：Matsubara 分解（不使用 Ishizaki-Tanimura 终止子）

```{code-cell} ipython3
with timer("RHS construction time"):
    env_mats = env.approximate(method="matsubara", Nk=Nk)
    HEOMMats = HEOMSolver(Hsys, (env_mats, Q), NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results so far
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Matsubara"),
    (resultMats, P12p, 'r', "P12 Matsubara"),
]);
```

## 模拟 2：Matsubara 分解（含终止子）

```{code-cell} ipython3
with timer("RHS construction time"):
    env_mats, delta = env.approximate(
        method="matsubara", Nk=Nk, compute_delta=True
    )
    Ltot = liouvillian(Hsys) + system_terminator(Q, delta)
    HEOMMatsT = HEOMSolver(Ltot, (env_mats, Q), NC, options=options)

with timer("ODE solver time"):
    resultMatsT = HEOMMatsT.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
plot_result_expectations([
    (resultMats, P11p, 'b', "P11 Matsubara"),
    (resultMats, P12p, 'r', "P12 Matsubara"),
    (resultMatsT, P11p, 'b--', "P11 Matsubara and terminator"),
    (resultMatsT, P12p, 'r--', "P12 Matsubara and terminator"),
]);
```

## 模拟 3：Padé 分解

与 1a 一样，这里可与 Padé 和拟合方法进行比较。

```{code-cell} ipython3
with timer("RHS construction time"):
    env_pade = env.approximate(method="pade", Nk=Nk)
    HEOMPade = HEOMSolver(Hsys, (env_pade, Q), NC, options=options)

with timer("ODE solver time"):
    resultPade = HEOMPade.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
plot_result_expectations([
    (resultMatsT, P11p, 'b', "P11 Matsubara (+term)"),
    (resultMatsT, P12p, 'r', "P12 Matsubara (+term)"),
    (resultPade, P11p, 'b--', "P11 Pade"),
    (resultPade, P12p, 'r--', "P12 Pade"),
]);
```

## 模拟 4：拟合方法

```{code-cell} ipython3
tfit = np.linspace(0, 10, 1000)
with timer("RHS construction time"):
    env_fit, _ = env.approximate(
        method="cf", tlist=tfit, Ni_max=1, Nr_max=3, target_rmse=None
    )
    HEOMFit = HEOMSolver(Hsys, (env_fit, Q), NC, options=options)

with timer("ODE solver time"):
    resultFit = HEOMFit.run(rho0, tlist)
```

## 解析计算

```{code-cell} ipython3
def pure_dephasing_evolution_analytical(tlist, wq, ck, vk):
    """
    Computes the propagating function appearing in the pure dephasing model.

    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.

    wq: float
        The qubit frequency in the Hamiltonian.

    ck: ndarray
        The list of coefficients in the correlation function.

    vk: ndarray
        The list of frequencies in the correlation function.

    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    evolution = np.array(
        [np.exp(-1j * wq * t - correlation_integral(t, ck, vk)) for t in tlist]
    )
    return evolution


def correlation_integral(t, ck, vk):
    r"""
    Computes the integral sum function appearing in the pure dephasing model.

    If the correlation function is a sum of exponentials then this sum
    is given by:

    .. math:

        \int_0^{t}d\tau D(\tau) = \sum_k\frac{c_k}{\mu_k^2}e^{\mu_k t}
        + \frac{\bar c_k}{\bar \mu_k^2}e^{\bar \mu_k t}
        - \frac{\bar \mu_k c_k + \mu_k \bar c_k}{\mu_k \bar \mu_k} t
        + \frac{\bar \mu_k^2 c_k + \mu_k^2 \bar c_k}{\mu_k^2 \bar \mu_k^2}

    Parameters
    ----------
    t: float
        A float specifying the time at which to calculate the integral.

    ck: ndarray
        The list of coefficients in the correlation function.

    vk: ndarray
        The list of frequencies in the correlation function.

    Returns
    -------
    integral: float
        The value of the integral function at time t.
    """
    t1 = np.sum((ck / vk**2) * (np.exp(vk * t) - 1))
    t2 = np.sum((ck.conj() / vk.conj()**2) * (np.exp(vk.conj() * t) - 1))
    t3 = np.sum((ck / vk + ck.conj() / vk.conj()) * t)
    return 2 * (t1 + t2 - t3)
```

对于纯退相干解析解，我们可尽可能多地累加 Matsubara 项：

```{code-cell} ipython3
lmaxmats2 = 15000

vk = [complex(-gamma)]
vk.extend([complex(-2.0 * np.pi * k * T) for k in range(1, lmaxmats2)])

ck = [complex(lam * gamma * (-1.0j + cot(gamma * beta / 2.0)))]
ck.extend(
    [complex(4 * lam * gamma * T * (-v) / (v**2 - gamma**2)) for v in vk[1:]]
)

P12_ana = 0.5 * pure_dephasing_evolution_analytical(
    tlist, 0, np.asarray(ck), np.asarray(vk)
)
```

另一种做法是不经关联函数，直接对传播子积分。

```{code-cell} ipython3
def JDL(omega, lamc, omega_c):
    return 2.0 * lamc * omega * omega_c / (omega_c**2 + omega**2)


def integrand(omega, lamc, omega_c, Temp, t):
    return (
        (-4.0 * JDL(omega, lamc, omega_c) / omega**2) *
        (1.0 - np.cos(omega*t)) * (coth(omega/(2*Temp)))
        / np.pi
    )


P12_ana2 = [
    0.5 * np.exp(
        scipy.integrate.quad(integrand, 0, np.inf, args=(lam, gamma, T, t))[0]
    )
    for t in tlist
]
```

## 比较结果

```{code-cell} ipython3
plot_result_expectations([
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultMatsT, P12p, 'r--', "P12 Mats + Term"),
    (resultPade, P12p, 'b--', "P12 Pade"),
    (resultFit, P12p, 'g', "P12 Fit"),
    ((tlist, np.real(P12_ana)), None, 'b', "Analytic 1"),
    ((tlist, np.real(P12_ana2)), None, 'y--', "Analytic 2"),
]);
```

上图差异不明显，我们改用对数坐标再看一次：

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

plot_result_expectations([
    (resultMats, P12p, 'r', "P12 Mats"),
    (resultMatsT, P12p, 'r--', "P12 Mats + Term"),
    (resultPade, P12p, 'b-.', "P12 Pade"),
    (resultFit, P12p, 'g', "P12 Fit"),
    ((tlist, np.real(P12_ana)), None, 'b', "Analytic 1"),
    ((tlist, np.real(P12_ana2)), None, 'y--', "Analytic 2"),
], axes)

axes.set_yscale('log')
axes.legend(loc=0, fontsize=12);
```

## 关于

```{code-cell} ipython3
about()
```

## 测试

本节可包含测试，以验证笔记本是否生成预期输出。我们把该节放在末尾，以免影响阅读流程。请使用 `assert` 定义测试；当输出错误时，单元应执行失败。

```{code-cell} ipython3
assert np.allclose(
    expect(P12p, resultMats.states[:15]), np.real(P12_ana)[:15],
    rtol=1e-2,
)
assert np.allclose(
    expect(P12p, resultMatsT.states[:100]), np.real(P12_ana)[:100],
    rtol=1e-3,
)
assert np.allclose(
    expect(P12p, resultPade.states[:100]), np.real(P12_ana)[:100],
    rtol=1e-3,
)
assert np.allclose(
    expect(P12p, resultFit.states[:50]), np.real(P12_ana)[:50],
    rtol=1e-3,
)
assert np.allclose(P12_ana, P12_ana2, rtol=1e-3)
```
