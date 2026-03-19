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

# HEOM 1d：自旋-热浴模型，谱与关联函数拟合

+++

## 引言

HEOM 方法可求解系统及其环境的动力学与稳态，环境部分
通过一组辅助密度矩阵编码。

本例展示一个与单个玻色环境耦合的二能级系统演化。

系统性质由哈密顿量给出，并通过耦合算符描述其与环境的相互作用。

默认玻色环境服从一类特定哈密顿量（[见论文](https://arxiv.org/abs/2010.10806)），其参数通过谱密度及自由热浴关联函数给出。

下面我们以三种方式建模带指数截止的 Ohmic 环境：

* 第一种：用一组欠阻尼布朗振子函数拟合谱密度。
* 第二种：先计算关联函数，再用特定指数函数族进行拟合。
* 第三种：使用内置 OhmicBath 类，并考察 QuTiP 提供的其它近似方法。

在每种情况下，我们都将利用拟合参数得到关联函数展开系数，构造热浴描述（即 `BosonicEnvironment` 对象），并将其传给 `HEOMSolver` 以求解系统动力学。

+++

## 设置

```{code-cell} ipython3
# mpmath is required for this tutorial,
# for the evaluation of gamma and zeta
# functions in the expression for the correlation:
from mpmath import mp

import numpy as np
from matplotlib import pyplot as plt

from qutip import about, basis, expect, sigmax, sigmaz
from qutip.core.environment import BosonicEnvironment, OhmicEnvironment
from qutip.solver.heom import HEOMSolver

%matplotlib inline

mp.dps = 15
mp.pretty = True
```

## 系统与热浴定义

下面设置系统哈密顿量、热浴与系统测量算符：

+++

### 系统哈密顿量

```{code-cell} ipython3
# Defining the system Hamiltonian
eps = 0    # Energy of the 2-level system.
Del = 0.2  # Tunnelling term
Hsys = 0.5 * eps * sigmaz() + 0.5 * Del * sigmax()

# Initial state of the system.
rho0 = basis(2, 0) * basis(2, 0).dag()
```

### 系统测量算符

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresonding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresonding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

### Ohmic 热浴关联函数与谱密度的解析表达

+++

开始拟合前，先检查关联函数与谱密度的解析表达，并写出对应 Python 实现。

关联函数表达式如下（推导可见 http://www1.itp.tu-berlin.de/brandes/public_html/publications/notes.pdf 的式 7.59，这里仅把一个 $\pi$ 因子并入关联函数定义）：

\begin{align}
C(t) =& \: \frac{1}{\pi}\alpha \omega_{c}^{1 - s} \beta^{- (s + 1)} \: \times \\
      & \: \Gamma(s + 1) \left[ \zeta \left(s + 1, \frac{1 + \beta \omega_c - i \omega_c t}{\beta \omega_c}\right) + \zeta \left(s + 1, \frac{1 + i \omega_c t}{\beta \omega_c}\right) \right]
\end{align}

其中 $\Gamma$ 是 Gamma 函数，

\begin{equation}
\zeta(z, u) \equiv \sum_{n=0}^{\infty} \frac{1}{(n + u)^z}, \; u \neq 0, -1, -2, \ldots
\end{equation}

这是广义 Zeta 函数。Ohmic 情形对应 $s=1$。

对应的 Ohmic 谱密度为：

\begin{equation}
J(\omega) = \omega \alpha e^{- \frac{\omega}{\omega_c}}
\end{equation}

```{code-cell} ipython3
def ohmic_correlation(t, alpha, wc, beta, s=1):
    """ The Ohmic bath correlation function as a function of t
        (and the bath parameters).
    """
    corr = (1 / np.pi) * alpha * wc ** (1 - s)
    corr *= beta ** (-(s + 1)) * mp.gamma(s + 1)
    z1_u = (1 + beta * wc - 1.0j * wc * t) / (beta * wc)
    z2_u = (1 + 1.0j * wc * t) / (beta * wc)
    # Note: the arguments to zeta should be in as high precision as possible.
    # See http://mpmath.org/doc/current/basics.html#providing-correct-input
    return np.array(
        [
            complex(corr * (mp.zeta(s + 1, u1) + mp.zeta(s + 1, u2)))
            for u1, u2 in zip(z1_u, z2_u)
        ],
        dtype=np.complex128,
    )
```

```{code-cell} ipython3
def ohmic_spectral_density(w, alpha, wc):
    """ The Ohmic bath spectral density as a function of w
        (and the bath parameters).
    """
    return w * alpha * np.e ** (-w / wc)
```

```{code-cell} ipython3
def ohmic_power_spectrum(w, alpha, wc, beta):
    """ The Ohmic bath power spectrum as a function of w
        (and the bath parameters).
        We here obtain it naively using the Fluctuation-Dissipation Theorem,
        but this fails at w=0 where the limit should be taken properly
    """
    bose = (1 / (np.e ** (w * beta) - 1)) + 1
    return w * alpha * np.e ** (-abs(w) / wc) * 2 * bose
```

### 热浴与 HEOM 参数

+++

最后设置本例使用的热浴参数

```{code-cell} ipython3
Q = sigmaz()
alpha = 3.25
T = 0.5
wc = 1.0
s = 1
```

并设置 HEOM 层级截断：

```{code-cell} ipython3
# HEOM parameters:

# The max_depth defaults to 5 so that the notebook executes more
# quickly. Change it to 11 to wait longer for more accurate results.
max_depth = 5

# options used for the differential equation solver
# "bdf" integration method is faster here
options = {
    "nsteps": 15000,
    "store_states": True,
    "rtol": 1e-12,
    "atol": 1e-12,
    "method": "bdf",
}
```

# 获取环境的衰减指数表示

为了执行 HEOM 模拟，我们需要把关联
函数表示为衰减指数和，即写成

$$C(\tau)= \sum_{k=0}^{N-1}c_{k}e^{-\nu_{k}t}$$

由于环境关联函数与其功率谱通过
傅里叶变换相联系，因此这种关联函数表示也意味着
功率谱具有如下形式

$$S(\omega)= \sum_{k}2 Re\left( \frac{c_{k}}{\nu_{k}- i \omega}\right)$$

获得这种分解有多种方法，本教程
将覆盖以下方案：

- 非线性最小二乘：
    - 在谱密度上（`sd`）
    - 在关联函数上（`cf`）
    - 在功率谱上（`ps`）
- 基于 Prony 多项式的方法
    - 在关联函数上使用 Prony（`prony`）
    - 在关联函数上使用 ESPRIT（`esprit`）
- 基于有理逼近的方法
    - 在功率谱上使用 AAA 算法（`aaa`）
    - 在关联函数及其 FFT 上使用 ESPIRA-I（`espira-I`）
    - 在关联函数及其 FFT 上使用 ESPIRA-II（`espira-II`）

+++

## 构建用户自定义玻色环境

在做指数近似之前，我们先构建描述精确环境的
`BosonicEnviroment`。
这里简要说明如何通过给定谱密度
创建用户自定义 `BosonicEnviroment`。同样也可以
通过关联函数或功率谱来构造。此处
使用前面定义的 Ohmic 谱密度：

```{code-cell} ipython3
w = np.linspace(0, 25, 20000)
J = ohmic_spectral_density(w, alpha, wc)
```

`BosonicEnvironment` 类提供了专门构造器，可
从任意谱密度、关联函数或
功率谱创建环境。例如：

```{code-cell} ipython3
# From an array
sd_env = BosonicEnvironment.from_spectral_density(J=J, wlist=w, T=T)
```

温度参数可选，但若给出温度，便可自动计算对应功率谱和关联函数。例如，自动计算得到的功率谱与前面解析定义的 `ohmic_power_spectrum` 一致：

```{code-cell} ipython3
# Here we avoid w=0
np.allclose(
    sd_env.power_spectrum(w[1:]), ohmic_power_spectrum(w[1:], alpha, wc, 1 / T)
)
```

指定温度后，QuTiP 还可通过快速傅里叶变换自动计算关联函数：

```{code-cell} ipython3
tlist = np.linspace(0, 10, 500)
plt.plot(
    tlist,
    sd_env.correlation_function(tlist).real,
    label="BosonicEnvironment FFT (Real Part)",
)
plt.plot(
    tlist,
    ohmic_correlation(tlist, alpha, wc, 1 / T).real,
    "--",
    label="Analytical (Real Part)",
)
plt.plot(
    tlist,
    np.imag(sd_env.correlation_function(tlist)),
    label="BosonicEnvironment FFT (Imaginary Part)",
)
plt.plot(
    tlist,
    np.imag(ohmic_correlation(tlist, alpha, wc, 1 / T)),
    "--",
    label="Analytical (Imaginary Part)",
)
plt.ylabel("C(t)")
plt.xlabel("t")
plt.legend()
plt.show()
```

注意上面我们是用数组 `w` 与 `J` 构建 `BosonicEnvironment`。
也可以直接使用纯 Python 函数。
这种情况下务必指定参数 `wMax`，即当频率超过该值时
谱密度或功率谱已衰减到可忽略水平。也就是对 $\omega>\omega_{max}$，函数
可视为近似零。如下写法与上面环境等价：

```{code-cell} ipython3
# From a function
sd_env2 = BosonicEnvironment.from_spectral_density(
    ohmic_spectral_density, T=T, wMax=25 * wc, args={"alpha": alpha, "wc": wc}
)
```

## 通过拟合谱密度构建指数环境

构建好 `BosonicEnvironment` 后，可通过拟合谱密度、
功率谱或关联函数，得到环境的衰减
指数表示。

先做谱密度的非线性最小二乘拟合，使用 $k$ 个欠阻尼谐振子的 Meier-Tannor 形式（J. Chem. Phys. 111, 3365 (1999); https://doi.org/10.1063/1.479669）：

\begin{equation}
J_{\mathrm approx}(\omega; a, b, c) = \sum_{i=0}^{k-1} \frac{2 a_i b_i w}{((w + c_i)^2 + b_i^2) ((w - c_i)^2 + b_i^2)}
\end{equation}

其中 $a,b,c$ 是拟合参数向量，长度均为 $k$。
其思想是把任意谱密度表示为
若干具有不同系数的欠阻尼谱密度之和，对每个分量
都可使用 Matsubara 分解。

利用 `approximate` 方法即可完成拟合。其输出是一个包含 `ExponentialBosonicEnvironment`
与拟合信息字典的元组，
字典中记录了拟合相关的关键信息。

默认会自动增加拟合项数，直到达到目标精度
或达到允许的最大项数 `Nmax`。（目标精度可设为 None，
此时只用指定的 `Nmax` 个指数项进行拟合。）

```{code-cell} ipython3
# adding a small uncertainty "sigma" helps the fit routine
approx_env, fitinfo = sd_env.approximate("sd", w, Nmax=4, sigma=0.0001)
```

为了快速了解拟合结果，可查看 ``fitinfo`` 中的摘要

```{code-cell} ipython3
print(fitinfo["summary"])
```

由于近似关联函数对应的有效谱密度与功率谱可由 `approx_env` 获取，我们可以将它们与原始目标进行比较：

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.plot(w, J, label="Original spectral density")
ax1.plot(w, approx_env.spectral_density(w), "--", label="Effective fitted SD")
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel(r"$J$")
ax1.legend()

ax2.plot(w, np.abs(J - approx_env.spectral_density(w)), label="Error")
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel(r"$|J-J_{approx}|$")
ax2.legend()

fig.tight_layout()
plt.show()
```

这里会看到近似（有效）谱密度与目标之间有较大偏差。原因是每个欠阻尼模使用的指数项（即 Matsubara 项）太少。所有模默认使用相同 Matsubara 项数；若不显式指定，默认是 $1$，对当前温度显然不足。下面增大指数项数后重试。

```{code-cell} ipython3
# 3 Matsubara terms per mode instead of one (default)
approx_env, fitinfo = sd_env.approximate("sd", w, Nmax=4, Nk=3, sigma=0.0001)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

ax1.plot(w, J, label="Original spectral density")
ax1.plot(w, approx_env.spectral_density(w), "--", label="Effective fitted SD")
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel(r"$J$")
ax1.legend()

ax2.plot(w, np.abs(J - approx_env.spectral_density(w)), label="Error")
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel(r"$|J-J_{approx}|$")
ax2.legend()

fig.tight_layout()
plt.show()
```

指数项数越多，仿真越慢，因此应尽量用“最少但足够”的指数项来正确描述热浴性质（功率谱、谱密度和关联函数）。

再细看上一次拟合：绘制每个拟合项的贡献。

```{code-cell} ipython3
def plot_fit_components(func, J, w, lam, gamma, w0):
    """ Plot the individual components of a fit to the spectral density
        and how they contribute to the full fit"""
    plt.figure(figsize=(10, 5))
    plt.plot(w, J, "r--", linewidth=2, label="original")
    for i in range(len(lam)):
        component = func(w, lam[i], gamma[i], w0[i])
        plt.plot(w, component, label=rf"$k={i+1}$")
    plt.xlabel(r"$\omega$", fontsize=20)
    plt.ylabel(r"$J(\omega)$", fontsize=20)
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.show()

lam = fitinfo["params"][:, 0]
gamma = fitinfo["params"][:, 1]
w0 = fitinfo["params"][:, 2]

def _sd_fit_model(wlist, a, b, c):
    return (
        2 * a * b * wlist
        / (((wlist + c) ** 2 + b**2) * ((wlist - c) ** 2 + b**2))
    )

plot_fit_components(_sd_fit_model, J, w, lam, gamma, w0)
```

同时比较拟合功率谱与解析谱密度：

```{code-cell} ipython3
def plot_power_spectrum(alpha, wc, beta):
    """ Plot the power spectrum of a fit against the actual power spectrum. """
    w = np.linspace(-10, 10, 50000)
    s_orig = ohmic_power_spectrum(w, alpha=alpha, wc=wc, beta=beta)
    s_fit = approx_env.power_spectrum(w)
    fig, axes = plt.subplots(1, 1, sharex=True)
    axes.plot(w, s_orig, "r", linewidth=2, label="original")
    axes.plot(w, np.real(s_fit), "b", linewidth=2, label="fit")

    axes.set_xlabel(r"$\omega$", fontsize=20)
    axes.set_ylabel(r"$S(\omega)$", fontsize=20)
    axes.legend()

plot_power_spectrum(alpha, wc, 1 / T)
```

若想考察拟合参数与仿真参数变化对系统行为的影响，可使用这个辅助函数。

```{code-cell} ipython3
def generate_spectrum_results(N, Nk, max_depth):
    """ Run the HEOM with the given bath parameters and
        and return the results of the evolution.
    """
    approx_env, fitinfo = sd_env.approximate(
        "sd", w, Nmax=N, Nk=Nk, sigma=0.0001, target_rmse=None
    )
    tlist = np.linspace(0, 30 * np.pi / Del, 600)

    print(f"Starting calculations for N={N}, Nk={Nk}"
          f" and max_depth={max_depth} ... ")

    HEOM_spectral_fit = HEOMSolver(
        Hsys, (approx_env, Q), max_depth=max_depth,
        options={**options, 'progress_bar': False},
    )
    results_spectral_fit = HEOM_spectral_fit.run(rho0, tlist)
    return results_spectral_fit
```

```{code-cell} ipython3
def plot_result_expectations(plots, axes=None):
    """ Plot the expectation values of operators as functions of time.

        Each plot in plots consists of (solver_result,
        measurement_operation, color, label).
    """
    if axes is None:
        fig, axes = plt.subplots(1, 1, sharex=True)
        fig_created = True
    else:
        fig = None
        fig_created = False

    # add kw arguments to each plot if missing
    plots = [p if len(p) == 5 else p + ({},) for p in plots]
    for result, m_op, color, label, kw in plots:
        exp = np.real(expect(result.states, m_op))
        kw.setdefault("linewidth", 2)
        if color == "rand":
            axes.plot(result.times, exp, color=np.random.rand(3),
                      label=label, **kw)
        else:
            axes.plot(result.times, exp, color, label=label, **kw)

    if fig_created:
        axes.legend(loc=0, fontsize=12)
        axes.set_xlabel("t", fontsize=20)

    return fig
```

下面对不同收敛参数（拟合项数、Matsubara 项数、层级深度）生成结果。对当前参数，层级深度大约需到 `11`，计算会稍慢。

```{code-cell} ipython3
# Generate results for different number of lorentzians in fit:

results_spectral_fit_pk = [
    generate_spectrum_results(n, Nk=1, max_depth=max_depth)
    for n in range(1, 5)
]

plot_result_expectations([
    (result, P11p, "rand", f"P11 (spectral fit) $k$={pk + 1}")
    for pk, result in enumerate(results_spectral_fit_pk)
]);
```

```{code-cell} ipython3
# generate results for different number of Matsubara terms per Lorentzian:

Nk_list = range(1, 4)
results_spectral_fit_nk = [
    generate_spectrum_results(4, Nk=Nk, max_depth=max_depth)
    for Nk in Nk_list
]

plot_result_expectations([
    (result, P11p, "rand", f"P11 (spectral fit) N_k={nk}")
    for nk, result in zip(Nk_list, results_spectral_fit_nk)
]);
```

```{code-cell} ipython3
# generate results for different hierarchy depths:

Nc_list = range(3, max_depth+1)
results_spectral_fit_nc = [
    generate_spectrum_results(4, Nk=1, max_depth=Nc)
    for Nc in Nc_list
]

plot_result_expectations([
    (result, P11p, "rand", f"P11 (spectral fit) $N_C={nc}$")
    for nc, result in zip(Nc_list, results_spectral_fit_nc)
]);
```

## 通过拟合关联函数获得衰减指数表示

+++

在成功完成谱密度拟合并据此构造 HEOM 玻色热浴的 Matsubara 展开与终止子后，我们继续第二种情况：直接拟合关联函数。

这里分别拟合实部与虚部，采用如下 Ansatz

$$C_R^F(t) = \sum_{i=1}^{k_R} c_R^ie^{-\gamma_R^i t}\cos(\omega_R^i t)$$

$$C_I^F(t) = \sum_{i=1}^{k_I} c_I^ie^{-\gamma_I^i t}\sin(\omega_I^i t)$$

该拟合同样可通过 `approximate` 方便完成。与谱密度拟合相比，区别在于这里需要做两次拟合：实部一次、虚部一次。

+++

注意当 $C_I^F(0) \neq 0$ 时，上述 Ansatz 不再理想。此时可设置 `full_ansatz=True`，
使用更一般的 Ansatz，但拟合会明显变慢。细节见文档。

```{code-cell} ipython3
def generate_corr_results(N, max_depth):
    tlist = np.linspace(0, 30 * np.pi / Del, 600)
    approx_env, fitinfo = sd_env.approximate(
        "cf", tlist=tlist, Ni_max=N, Nr_max=N, maxfev=1e8, target_rmse=None
    )

    print(f"Starting calculations for N={N}"
          f" and max_depth={max_depth} ... ")

    HEOM_corr_fit = HEOMSolver(
        Hsys, (approx_env, Q), max_depth=max_depth,
        options={**options, 'progress_bar': False},
    )
    results_corr_fit = HEOM_corr_fit.run(rho0, tlist)
    return results_corr_fit
```

```{code-cell} ipython3
# Generate results for different number of exponentials in fit:
results_corr_fit_pk = [
    generate_corr_results(i, max_depth=max_depth)
    for i in range(1, 4)
]

plot_result_expectations([
    (result, P11p, "rand", f"P11 (correlation fit) k_R=k_I={pk + 1}")
    for pk, result in enumerate(results_corr_fit_pk)
]);
```

```{code-cell} ipython3
# Comparison plot

fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 6))

plot_result_expectations([
    (results_corr_fit_pk[0], P11p, "y", "Correlation Fct. Fit $k_R=k_I=1$"),
    (results_corr_fit_pk[2], P11p, "k", "Correlation Fct. Fit $k_R=k_I=3$"),
    (results_spectral_fit_pk[0], P11p, "b", "Spectral Density Fit $k_J=1$"),
    (results_spectral_fit_pk[3], P11p, "r-.", "Spectral Density Fit $k_J=4$"),
], axes=axes)

axes.set_yticks([0.6, 0.8, 1])
axes.set_ylabel(r"$\rho_{11}$", fontsize=20)
axes.set_xlabel(r"$t\;\omega_c$", fontsize=20)
axes.legend(loc=0, fontsize=15);
```

# 使用 Ohmic Environment 类

由于 Ohmic 谱在开放量子系统建模中很常见，QuTiP 提供了专门的 `OhmicEnvironment` 类，可快速复现上面结果并便于实现拟合后的 Ohmic 热浴。

```{code-cell} ipython3
obs = OhmicEnvironment(T, alpha, wc, s=1)
tlist = np.linspace(0, 30 * np.pi / Del, 600)
```

与其它 `BosonicEnvironment` 一样，我们可通过 `approximate` 获取环境的衰减指数表示。下面先重复
前面已经探索过的
方法：

```{code-cell} ipython3
sd_approx_env, fitinfo = obs.approximate(
    method="sd", wlist=w, Nmax=4, Nk=3, sigma=0.0001, target_rmse=None
)
print(fitinfo["summary"])
HEOM_ohmic_sd_fit = HEOMSolver(
    Hsys, (sd_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_sd_fit = HEOM_ohmic_sd_fit.run(rho0, tlist)
```

```{code-cell} ipython3
cf_approx_env, fitinfo = obs.approximate(
    method="cf", tlist=tlist, Nr_max=4, Ni_max=4, maxfev=1e8, target_rmse=None
)
print(fitinfo["summary"])
HEOM_ohmic_corr_fit = HEOMSolver(
    Hsys, (cf_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_corr_fit = HEOM_ohmic_corr_fit.run(rho0, tlist)
```

## 其它近似方法
### 基于 Prony 多项式的方法

Prony 多项式是许多谱分析技术的数学基础，可用于估计信号频率、阻尼因子和幅值。其核心思想是把信号解释为复指数和，并构造一个多项式，其根对应系统频率或极点。

这些方法考虑形如

$$f(t)=\sum_{k=0}^{N-1} c_{k} e^{-\nu_{k} t} =\sum_{k=0}^{N-1} c_{k} z_{k}^{t}  $$

$z_k$ 可视为矩阵 pencil 的广义特征值

\begin{align}
z_{j}  {\mathbf H}_{2N-L,L}(0) - {\mathbf H}_{2N-L,L}(1) = {\mathbf V}_{2N-L,M}
({\mathbf z})   \mathrm{diag}  \Big( \left( (z_{j} - z_{k})\gamma_{k} 
\right)_{k=1}^{M} \Big) {\mathbf V}_{L,M}({\mathbf z})^{T}
\end{align}



之后可通过求解最小二乘 Vandermonde 系统得到振幅 $c_k$：

$$ V_{N,M}(z)c = f $$

其中 $V_{N,M}(z)$ 是 Vandermonde 矩阵


$$V_{M,N}(z)=\begin{pmatrix} 
1 &1 &\dots &1 \\
z_{1} & z_{2} &\dots & z_{N} \\
z_{1}^{2} & z_{2}^{2} &\dots & z_{N}^{2} \\
\vdots & \vdots & \ddots & \vdots  \\
z_{1}^{M} & z_{2}^{M} &\dots & z_{N}^{M} \\
\end{pmatrix}$$

这里 $M$ 是信号长度，$N$ 是指数项数，$f=f(t_{sample})$ 是采样点上的信号，$c=(c_1,\dots,c_N)$ 为待求向量。

这些方法的主要差别在于如何求多项式根：是直接解该系统，还是先做低秩近似。[这篇文章](https://academic.oup.com/imajna/article-abstract/43/2/789/6525860?redirectedFrom=fulltext)是很好的参考，QuTiP 实现基于该文以及作者提供的 Matlab 实现。

Prony 类方法包括：

- Prony
- ESPRIT
- ESPIRA

虽然 ESPIRA 也是 Prony 类方法，但由于它基于有理多项式逼近，
我们将其归入另一组方法。

+++

##### 在关联函数上使用原始 Prony 方法

该方法可通过 `approximate` 并传入 `"prony"` 使用。相比此前方法，Prony 类方法胜在简单：无需先验函数信息，只需提供采样点与期望指数项数。

```{code-cell} ipython3
tlist2 = np.linspace(0, 40, 100)
```

```{code-cell} ipython3
prony_approx_env, fitinfo = obs.approximate("prony", tlist2, Nr=4)
print(fitinfo["summary"])
HEOM_ohmic_prony_fit = HEOMSolver(
    Hsys, (prony_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_prony_fit = HEOM_ohmic_prony_fit.run(rho0, tlist)
```

与 Prony 类似，也可使用 ESPRIT；两者主要差异
在于 pencil 矩阵的构造方式。

```{code-cell} ipython3
esprit_approx_env, fitinfo = obs.approximate(
    "esprit", tlist2, Nr=4, separate=False
)
print(fitinfo["summary"])
HEOM_ohmic_es_fit = HEOMSolver(
    Hsys, (esprit_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_es_fit = HEOM_ohmic_es_fit.run(rho0, tlist)
```

## 拟合功率谱

到目前为止，前述方法都拟合的是谱密度或
关联函数。这里我们改为拟合功率谱。

### AAA 算法

Adaptive Antoulas-Anderson（AAA）算法是一种
以多项式商形式逼近函数的方法

\begin{align}
    f(z) =\frac{q(z)}{p(z)} \approx \sum_{j=1}^{m} \frac{residues}{z-poles}
\end{align}

我们并不直接在关联函数上使用 AAA，而是作用于功率谱。得到功率谱的有理形式后，可利用下式恢复关联函数：

\begin{align}
    s(\omega) = \int_{-\infty}^{\infty} dt e^{i \omega t} C(t)  = 2 \Re \left(\sum_{k} \frac{c_{k}}{\nu_{k}-i \omega} \right)
\end{align}

这使我们可识别出

\begin{align}
    \nu_{k}= i \times poles \\
    c_{k} = -i \times residues
\end{align}

当采样点按对数尺度给出时，该方法效果最好：

```{code-cell} ipython3
wlist = np.concatenate((-np.logspace(3, -8, 3500), np.logspace(-8, 3, 3500)))

aaa_aprox_env, fitinfo = obs.approximate("aaa", wlist, Nmax=8, tol=1e-15)
print(fitinfo["summary"])
HEOM_ohmic_aaa_fit = HEOMSolver(
    Hsys, (aaa_aprox_env, Q), max_depth=max_depth, options=options
)
results_ohmic_aaa_fit = HEOM_ohmic_aaa_fit.run(rho0, tlist)
```

### 在功率谱上做 NLSQ

教程前半部分我们已经用了基于非线性
最小二乘的方法。这里也是同类方法，但作用在功率
谱上。相较直接拟合谱密度，它不需要指定 $N_k$；
相较拟合关联函数，它更容易得到满足 KMS 关系的
近似。

这里把功率谱拟合为

$$S(\omega) = \sum_{k=1}^{N}\frac{2(a_k c_k + b_k (d_k - \omega))}
{(\omega - d_k)^2 + c_k^2}= 2 \Re \left(\sum_{k} \frac{c_{k}}{\nu_{k}-i \omega} \right)$$

```{code-cell} ipython3
w2 = np.concatenate((-np.linspace(10, 1e-2, 100), np.linspace(1e-2, 10, 100)))

ps_approx_env, fitinfo = obs.approximate("ps", w2, Nmax=4)
print(fitinfo["summary"])
HEOM_ohmic_ps_fit = HEOMSolver(
    Hsys, (ps_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_ps_fit = HEOM_ohmic_ps_fit.run(rho0, tlist)
```

### ESPIRA

ESPIRA 是一种 Prony 风格方法。虽然输入是关联函数，
但它利用参数估计（Prony 做的事）与有理逼近之间关系。
其有理逼近在 DFT 域通过 AAA 完成，因此在同一次拟合中
同时利用了
功率谱与关联函数信息。

我们提供两种 ESPIRA 实现：ESPIRA-I 与 ESPIRA-II。ESPIRA-I
通常更好，但在不少场景（尤其 `separate=True` 时）
ESPIRA-II 会更优。若预期存在
极慢衰减指数项，建议用 ESPIRA-II；否则通常
推荐 ESPIRA-I。

+++

##### ESPIRA I

```{code-cell} ipython3
tlist4 = np.linspace(0, 20, 1000)

espi_approx_env, fitinfo = obs.approximate("espira-I", tlist4, Nr=4)
print(fitinfo["summary"])
HEOM_ohmic_espira_fit = HEOMSolver(
    Hsys, (espi_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_espira_fit = HEOM_ohmic_espira_fit.run(rho0, tlist)
```

##### ESPIRA-II

```{code-cell} ipython3
espi2_approx_env, fitinfo = obs.approximate(
    "espira-II", tlist4, Nr=4, Ni=4, separate=True
)
print(fitinfo["summary"])
HEOM_ohmic_espira_fit2 = HEOMSolver(
    Hsys, (espi2_approx_env, Q), max_depth=max_depth, options=options
)
results_ohmic_espira2_fit = HEOM_ohmic_espira_fit2.run(rho0, tlist)
```

最后绘制不同方法得到的动力学结果

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(10, 6))

plot_result_expectations([
    (results_ohmic_corr_fit, P11p, "r", "Correlation Fct. Fit"),
    (results_ohmic_sd_fit, P11p, "g--", "Spectral Density Fit"),
    (results_ohmic_ps_fit, P11p, "g--", "Power Spectrum Fit Ohmic Bath"),
    (results_ohmic_prony_fit, P11p, "k", " Prony Fit"),
    (results_ohmic_es_fit, P11p, "b-.", "ESPRIT Fit"),
    (results_ohmic_aaa_fit, P11p, "r-.", "Matrix AAA Fit"),
    (results_ohmic_espira_fit, P11p, "k", "ESPIRA I Fit"),
    (results_ohmic_espira2_fit, P11p, "--", "ESPIRA II Fit"),
], axes=axes)

axes.set_ylabel(r"$\rho_{11}$", fontsize=20)
axes.set_xlabel(r"$t\;\omega_c$", fontsize=20)
axes.legend(loc=0, fontsize=15)
axes.set_yscale("log")
```

## 关于

```{code-cell} ipython3
about()
```

## 测试

本节可包含测试，以验证笔记本是否生成预期输出。我们把该节放在末尾，以免影响阅读流程。请使用 `assert` 定义测试；当输出错误时，单元应执行失败。

```{code-cell} ipython3
assert np.allclose(
    expect(P11p, results_spectral_fit_pk[2].states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)
assert np.allclose(
    expect(P11p, results_ohmic_aaa_fit.states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)
assert np.allclose(
    expect(P11p, results_ohmic_prony_fit.states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)

assert np.allclose(
    expect(P11p, results_ohmic_es_fit.states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)
assert np.allclose(
    expect(P11p, results_ohmic_espira_fit.states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)
assert np.allclose(
    expect(P11p, results_ohmic_espira2_fit.states),
    expect(P11p, results_spectral_fit_pk[3].states),
    rtol=1e-2,
)
```
