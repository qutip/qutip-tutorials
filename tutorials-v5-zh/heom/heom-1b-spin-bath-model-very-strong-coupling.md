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

# HEOM 1b：自旋-热浴模型（超强耦合）

+++

## 引言

HEOM 方法可求解系统及其环境的动力学与稳态；环境效应通过一组辅助密度矩阵编码。

本例展示一个与单个玻色环境接触的二能级系统演化。系统性质由哈密顿量与系统-环境耦合算符决定。

默认玻色环境服从一类特定哈密顿量，其参数通过谱密度及相应自由热浴关联函数给出。

下面示例继续使用 HEOM 中常见的过阻尼 Drude-Lorentz 谱密度，分别采用 Matsubara、Padé 与拟合分解并比较收敛性。

本笔记与 1a 的示例相近，但耦合显著更强（见 [Shi *et al.*, J. Chem. Phys **130**, 084105 (2009)](https://doi.org/10.1063/1.3077918)）。更详细背景请参阅 HEOM 1a。

与 1a 一样，我们给出多组不同近似下的模拟，以展示关联函数近似方式对结果的影响：

- 模拟 1：Matsubara 分解（不使用 Ishizaki-Tanimura 终止子）
- 模拟 2：Matsubara 分解（含终止子）
- 模拟 3：Padé 分解
- 模拟 4：拟合方法

最后，我们还与 Bloch-Redfield 方法比较：

- 模拟 5：Bloch-Redfield

该方法在本例中无法给出正确演化。


### Drude-Lorentz（过阻尼）谱密度

Drude-Lorentz 谱密度为：

$$J_D(\omega)= \frac{2\omega\lambda\gamma}{{\gamma}^2 + \omega^2}$$

其中 $\lambda$ 控制耦合强度，$\gamma$ 为截止频率。我们采用约定
\begin{equation*}
C(t) = \int_0^{\infty} d\omega \frac{J_D(\omega)}{\pi}[\coth(\beta\omega) \cos(\omega \tau) - i \sin(\omega \tau)]
\end{equation*}

在 HEOM 中必须将其写成指数分解：

\begin{equation*}
C(t)=\sum_{k=0}^{k=\infty} c_k e^{-\nu_k t}
\end{equation*}

例如，Drude-Lorentz 谱密度的 Matsubara 分解为：

\begin{equation*}
    \nu_k = \begin{cases}
               \gamma               & k = 0\\
               {2 \pi k} / {\beta }  & k \geq 1\\
           \end{cases}
\end{equation*}

\begin{equation*}
    c_k = \begin{cases}
               \lambda \gamma (\cot(\beta \gamma / 2) - i)             & k = 0\\
               4 \lambda \gamma \nu_k / \{(nu_k^2 - \gamma^2)\beta \}    & k \geq 1\\
           \end{cases}
\end{equation*}

注意在上式及下文中，我们取 $\hbar = k_\mathrm{B} = 1$。

+++

## 设置

```{code-cell} ipython3
import contextlib
import time

import numpy as np
import matplotlib.pyplot as plt

from qutip import about, basis, brmesolve, expect, liouvillian, sigmax, sigmaz
from qutip.core.environment import DrudeLorentzEnvironment, system_terminator
from qutip.solver.heom import HEOMSolver

%matplotlib inline
```

## 辅助函数

先定义若干辅助函数，用于计算关联函数展开、绘图以及计时。

```{code-cell} ipython3
def cot(x):
    """ Vectorized cotangent of x. """
    return 1.0 / np.tan(x)
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
eps = 0.0  # Energy of the 2-level system.
Del = 0.2  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()
```

```{code-cell} ipython3
# Initial state of the system.
rho0 = basis(2, 0) * basis(2, 0).dag()
```

```{code-cell} ipython3
# System-bath coupling (Drude-Lorentz spectral density)
Q = sigmaz()  # coupling operator

# Bath properties (see Shi et al., J. Chem. Phys. 130, 084105 (2009)):
gamma = 1.0  # cut off frequency
lam = 2.5    # coupling strength
T = 1.0      # in units where Boltzmann factor is 1
beta = 1.0 / T

# HEOM parameters:

# number of exponents to retain in the Matsubara expansion of the
# bath correlation function:
Nk = 1

# Number of levels of the hierarchy to retain:
NC = 13

# Times to solve for:
tlist = np.linspace(0, np.pi / Del, 600)
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

### 绘制谱密度

先简要检查谱密度。

```{code-cell} ipython3
env = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T, Nk=500)
w = np.linspace(0, 5, 1000)
J = env.spectral_density(w)

# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))
axes.plot(w, J, 'r', linewidth=2)
axes.set_xlabel(r'$\omega$', fontsize=28)
axes.set_ylabel(r'J', fontsize=28);
```

## 模拟 1：Matsubara 分解（不使用 Ishizaki-Tanimura 终止子）

```{code-cell} ipython3
with timer("RHS construction time"):
    matsEnv = env.approximate(method="matsubara", Nk=Nk)
    HEOMMats = HEOMSolver(Hsys, (matsEnv, Q), NC, options=options)

with timer("ODE solver time"):
    resultMats = HEOMMats.run(rho0, tlist)
```

## 模拟 2：Matsubara 分解（含终止子）

```{code-cell} ipython3
with timer("RHS construction time"):
    matsEnv, delta = env.approximate(
        method="matsubara", Nk=Nk, compute_delta=True
    )
    terminator = system_terminator(Q, delta)
    Ltot = liouvillian(Hsys) + terminator
    HEOMMatsT = HEOMSolver(Ltot, (matsEnv, Q), NC, options=options)

with timer("ODE solver time"):
    resultMatsT = HEOMMatsT.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

P11_mats = np.real(expect(resultMats.states, P11p))
axes.plot(
    tlist, np.real(P11_mats),
    'b', linewidth=2, label="P11 (Matsubara)",
)

P11_matsT = np.real(expect(resultMatsT.states, P11p))
axes.plot(
    tlist, np.real(P11_matsT),
    'b--', linewidth=2,
    label="P11 (Matsubara + Terminator)",
)

axes.set_xlabel(r't', fontsize=28)
axes.legend(loc=0, fontsize=12);
```

## 模拟 3：Padé 分解

```{code-cell} ipython3
# First, compare Matsubara and Pade decompositions
padeEnv = env.approximate("pade", Nk=Nk)


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(16, 8))

ax1.plot(
    tlist, np.real(env.correlation_function(tlist)),
    "r", linewidth=2, label="Exact",
)
ax1.plot(
    tlist, np.real(matsEnv.correlation_function(tlist)),
    "g--", linewidth=2, label=f"Mats (Nk={Nk})",
)
ax1.plot(
    tlist, np.real(padeEnv.correlation_function(tlist)),
    "b--", linewidth=2, label=f"Pade (Nk={Nk})",
)

ax1.set_xlabel(r't', fontsize=28)
ax1.set_ylabel(r"$C_R(t)$", fontsize=28)
ax1.legend(loc=0, fontsize=12)

tlist2 = tlist[0:50]
ax2.plot(
    tlist2, np.abs(matsEnv.correlation_function(tlist2)
                   - env.correlation_function(tlist2)),
    "g", linewidth=2, label="Mats Error",
)
ax2.plot(
    tlist2, np.abs(padeEnv.correlation_function(tlist2)
                   - env.correlation_function(tlist2)),
    "b--", linewidth=2, label="Pade Error",
)

ax2.set_xlabel(r't', fontsize=28)
ax2.legend(loc=0, fontsize=12);
```

```{code-cell} ipython3
with timer("RHS construction time"):
    HEOMPade = HEOMSolver(Hsys, (padeEnv, Q), NC, options=options)

with timer("ODE solver time"):
    resultPade = HEOMPade.run(rho0, tlist)
```

```{code-cell} ipython3
# Plot the results
fig, axes = plt.subplots(figsize=(8, 8))

axes.plot(
    tlist, np.real(P11_mats),
    'b', linewidth=2, label="P11 (Matsubara)",
)
axes.plot(
    tlist, np.real(P11_matsT),
    'b--', linewidth=2, label="P11 (Matsubara + Terminator)",
)

P11_pade = np.real(expect(resultPade.states, P11p))
axes.plot(
    tlist, np.real(P11_pade),
    'r', linewidth=2, label="P11 (Pade)",
)

axes.set_xlabel(r't', fontsize=28)
axes.legend(loc=0, fontsize=12);
```

## 模拟 4：拟合方法

在 `HEOM 1a: Spin-Bath model (introduction)` 中给出了手动拟合示例，这里仅使用内置工具。更详细说明见
`HEOM 1d: Spin-Bath model, fitting of spectrum and correlation functions`
`HEOM 1d: Spin-Bath model, fitting of spectrum and correlation functions`

```{code-cell} ipython3
tfit = np.linspace(0, 10, 10000)
lower = [0, -np.inf, -1e-6, -3]
guess = [np.real(env.correlation_function(0)) / 10, -10, 0, 0]
upper = [5, 0, 1e-6, 0]
# for better fits increase the first element in upper, or change approximate
# method that makes the simulation much slower (Larger C(t) as C(0) is fit
# better)

envfit, fitinfo = env.approximate(
    "cf", tlist=tfit, Nr_max=2, Ni_max=1, full_ansatz=True,
    sigma=0.1, maxfev=1e6, target_rmse=None,
    lower=lower, upper=upper, guess=guess,
)
```

```{code-cell} ipython3
print(fitinfo["summary"])
```

我们可以快速将拟合结果与 Padé 展开比较

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

ax1.plot(
    tlist, np.real(env.correlation_function(tlist)),
    "r", linewidth=2, label="Exact",
)
ax1.plot(
    tlist, np.real(envfit.correlation_function(tlist)),
    "g--", linewidth=2, label="Fit", marker="o", markevery=50,
)
ax1.plot(
    tlist, np.real(padeEnv.correlation_function(tlist)),
    "b--", linewidth=2, label=f"Pade (Nk={Nk})",
)

ax1.set_xlabel(r"t", fontsize=28)
ax1.set_ylabel(r"$C_R(t)$", fontsize=28)
ax1.legend(loc=0, fontsize=12)

ax2.plot(
    tlist, np.imag(env.correlation_function(tlist)),
    "r", linewidth=2, label="Exact",
)
ax2.plot(
    tlist, np.imag(envfit.correlation_function(tlist)),
    "g--", linewidth=2, label="Fit", marker="o", markevery=50,
)
ax2.plot(
    tlist, np.imag(padeEnv.correlation_function(tlist)),
    "b--", linewidth=2, label=f"Pade (Nk={Nk})",
)

ax2.set_xlabel(r"t", fontsize=28)
ax2.set_ylabel(r"$C_I(t)$", fontsize=28)
ax2.legend(loc=0, fontsize=12)
plt.show()
```

```{code-cell} ipython3
with timer("RHS construction time"):
    # We reduce NC slightly here for speed of execution because we retain
    # 3 exponents in ckAR instead of 1. Please restore full NC for
    # convergence though:
    HEOMFit = HEOMSolver(Hsys, (envfit, Q), int(NC * 0.7), options=options)

with timer("ODE solver time"):
    resultFit = HEOMFit.run(rho0, tlist)
```

## 模拟 5：Bloch-Redfield

```{code-cell} ipython3
with timer("ODE solver time"):
    resultBR = brmesolve(
        Hsys, rho0, tlist,
        a_ops=[[sigmaz(), env]], sec_cutoff=0, options=options,
    )
```

## 汇总绘制所有结果

最后，把不同方法得到的结果放在一起比较。

```{code-cell} ipython3
# Calculate expectation values in the bases:
P11_mats = np.real(expect(resultMats.states, P11p))
P11_matsT = np.real(expect(resultMatsT.states, P11p))
P11_pade = np.real(expect(resultPade.states, P11p))
P11_fit = np.real(expect(resultFit.states, P11p))
P11_br = np.real(expect(resultBR.states, P11p))
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
    # Plot the results
    plt.yticks([0.99, 1.0], [0.99, 1])
    axes.plot(
        tlist, np.real(P11_mats),
        'b', linewidth=2, label=f"Matsubara $N_k={Nk}$",
    )
    axes.plot(
        tlist, np.real(P11_matsT),
        'g--', linewidth=3,
        label=f"Matsubara $N_k={Nk}$ & terminator",
    )
    axes.plot(
        tlist, np.real(P11_pade),
        'y-.', linewidth=2, label=f"Pad茅 $N_k={Nk}$",
    )
    axes.plot(
        tlist, np.real(P11_fit),
        'r', dashes=[3, 2], linewidth=2,
        label=r"Fit $N_f = 3$, $N_k=15 \times 10^3$",
    )
    axes.plot(
        tlist, np.real(P11_br),
        'b-.', linewidth=1, label="Bloch Redfield",
    )

    axes.locator_params(axis='y', nbins=6)
    axes.locator_params(axis='x', nbins=6)
    axes.set_ylabel(r'$\rho_{11}$', fontsize=30)
    axes.set_xlabel(r'$t\;\gamma$', fontsize=30)
    axes.set_xlim(tlist[0], tlist[-1])
    axes.set_ylim(0.98405, 1.0005)
    axes.legend(loc=0)
```

## 关于

```{code-cell} ipython3
about()
```

## 测试

本节可包含测试，以验证笔记本是否生成预期输出。我们把该节放在末尾，以免影响阅读流程。请使用 `assert` 定义测试；当输出错误时，单元应执行失败。

```{code-cell} ipython3
assert np.allclose(P11_matsT, P11_pade, rtol=1e-3)
assert np.allclose(P11_matsT, P11_fit, rtol=1e-3)
```
