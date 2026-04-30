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

# HEOM 5a：费米单杂质模型

+++

## 引言

这里我们建模一个与两个电子引线（或电子库）耦合的单费米子系统（例如单量子点、分子晶体管等）。实现上主要遵循 Christian Schinabeck 博士论文及相关文献中的定义：https://open.fau.de/items/36fdd708-a467-4b59-bf4e-4a2110fbc431。

记号约定：

* $K=L/R$ 表示左/右引线
* $\sigma=\pm$ 表示输入/输出

我们为引线选用在化学势附近有峰值的 Lorentz 谱密度。这样可简化关联函数记号（必要时也可放宽该假设）。

$$J(\omega) = \frac{\Gamma  W^2}{((\omega-\mu_K)^2 +W^2 )}$$

费米分布函数为：

$$f_F (x) = (\exp(x) + 1)^{-1}$$

两者结合可写出关联函数：

$$C^{\sigma}_K(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega e^{\sigma i \omega t} \Gamma_K(\omega) f_F[\sigma\beta(\omega - \mu)]$$

与玻色情形类似，我们可用 Matsubara、Padé 或拟合方法把它展开为指数级数。

Padé 分解把费米分布近似为

$$f_F(x) \approx f_F^{\mathrm{approx}}(x) = \frac{1}{2} - \sum_l^{l_{max}} \frac{2k_l x}{x^2 + \epsilon_l^2}$$

其中 $k_l$ 与 $\epsilon_l$ 是 J. Chem. Phys. 133, 10106 中定义的系数。

对关联函数积分可得

$$C_K^{\sigma}(t) \approx \sum_{l=0}^{l_{max}} \eta_K^{\sigma_l} e^{-\gamma_{K,\sigma,l}t}$$

其中：

$$\eta_{K,0} = \frac{\Gamma_KW_K}{2} f_F^{approx}(i\beta_K W)$$

$$\gamma_{K,\sigma,0} = W_K - \sigma i\mu_K$$ 

$$\eta_{K,l\neq 0} = -i\cdot \frac{k_m}{\beta_K} \cdot \frac{\Gamma_K W_K^2}{-\frac{\epsilon^2_m}{\beta_K^2} + W_K^2}$$

$$\gamma_{K,\sigma,l\neq 0}= \frac{\epsilon_m}{\beta_K} - \sigma i \mu_K$$

在本笔记中我们将：

* 比较 Matsubara 与 Padé 近似，并与系统和引线间电流的解析结果对照。

* 绘制电流随两引线偏置电压差变化的曲线。

+++

## 设置

```{code-cell} ipython3
import contextlib
import dataclasses
import time

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from qutip import about, basis, destroy, expect
from qutip.core.environment import LorentzianEnvironment
from qutip.solver.heom import HEOMSolver

from IPython.display import display
from ipywidgets import IntProgress

%matplotlib inline
```

## 辅助函数

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

# We set store_ados to True so that we can
# use the auxilliary density operators (ADOs)
# to calculate the current between the leads
# and the system.

options = {
    "nsteps": 1500,
    "store_states": True,
    "store_ados": True,
    "rtol": 1e-12,
    "atol": 1e-12,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## 系统与热浴定义

下面设置系统哈密顿量、热浴以及系统测量算符：

```{code-cell} ipython3
# Define the system Hamiltonian:

# The system is a single fermion with energy level split e1:
d1 = destroy(2)
e1 = 1.0
H = e1 * d1.dag() * d1
```

```{code-cell} ipython3
# Define parameters for left and right fermionic baths.
# Each bath is a lead (i.e. a wire held at a potential)
# with temperature T and chemical potential mu.

@dataclasses.dataclass
class LorentzianBathParameters:
    lead: str
    Q: object  # coupling operator
    gamma: float = 0.01  # coupling strength
    W: float = 1.0  # cut-off
    T: float = 0.025851991  # temperature
    theta: float = 2.0  # bias

    def __post_init__(self):
        assert self.lead in ("L", "R")
        self.beta = 1 / self.T
        if self.lead == "L":
            self.mu = self.theta / 2.0
        else:
            self.mu = -self.theta / 2.0

    def J(self, w):
        """ Spectral density. """
        return self.gamma * self.W**2 / ((w - self.mu)**2 + self.W**2)

    def fF(self, w, sign=1.0):
        """ Fermi distribution for this bath. """
        x = sign * self.beta * (w - self.mu)
        return fF(x)

    def lamshift(self, w):
        """ Return the lamb shift. """
        return 0.5 * (w - self.mu) * self.J(w) / self.W

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def fF(x):
    """ Return the Fermi distribution. """
    # in units where kB = 1.0
    return 1 / (np.exp(x) + 1)


bath_L = LorentzianBathParameters(Q=d1, lead="L")
bath_R = LorentzianBathParameters(Q=d1, lead="R")
```

## 谱密度

先绘制谱密度。

```{code-cell} ipython3
w_list = np.linspace(-2, 2, 100)

fig, ax = plt.subplots(figsize=(12, 7))

spec_L = bath_L.J(w_list)
spec_R = bath_R.J(w_list)

ax.plot(
    w_list, spec_L,
    "b--", linewidth=3,
    label=r"J_L(w)",
)
ax.plot(
    w_list, spec_R,
    "r--", linewidth=3,
    label=r"J_R(w)",
)

ax.set_xlabel("w")
ax.set_ylabel(r"$J(\omega)$")
ax.legend();
```

## 引线的发射与吸收

接着绘制引线的发射与吸收过程。

```{code-cell} ipython3
w_list = np.linspace(-2, 2, 100)

fig, ax = plt.subplots(figsize=(12, 7))

# Left lead emission and absorption

gam_L_in = bath_L.J(w_list) * bath_L.fF(w_list, sign=1.0)
gam_L_out = bath_L.J(w_list) * bath_L.fF(w_list, sign=-1.0)

ax.plot(
    w_list, gam_L_in,
    "b--", linewidth=3,
    label=r"S_L(w) input (absorption)",
)
ax.plot(
    w_list, gam_L_out,
    "r--", linewidth=3,
    label=r"S_L(w) output (emission)",
)

# Right lead emission and absorption

gam_R_in = bath_R.J(w_list) * bath_R.fF(w_list, sign=1.0)
gam_R_out = bath_R.J(w_list) * bath_R.fF(w_list, sign=-1.0)

ax.plot(
    w_list, gam_R_in,
    "b", linewidth=3,
    label=r"S_R(w) input (absorption)",
)
ax.plot(
    w_list, gam_R_out,
    "r", linewidth=3,
    label=r"S_R(w) output (emission)",
)

ax.set_xlabel("w")
ax.set_ylabel(r"$S(\omega)$")
ax.legend();
```

## 比较 Matsubara 与 Padé 近似

先用 Lorentz 谱密度关联函数的 Padé 展开求系统演化：

```{code-cell} ipython3
# HEOM dynamics using the Pade approximation:

# Times to solve for and initial system state:
tlist = np.linspace(0, 100, 1000)
rho0 = basis(2, 0) * basis(2, 0).dag()

Nk = 10  # Number of exponents to retain in the expansion of each bath

envL = LorentzianEnvironment(
    bath_L.T, bath_L.mu, bath_L.gamma, bath_L.W,
)
envL_pade = envL.approx_by_pade(Nk=Nk, tag="L")
envR = LorentzianEnvironment(
    bath_R.T, bath_R.mu, bath_R.gamma, bath_R.W,
)
envR_pade = envR.approx_by_pade(Nk=Nk, tag="R")

with timer("RHS construction time"):
    solver_pade = HEOMSolver(
        H,
        [(envL_pade, bath_L.Q), (envR_pade, bath_R.Q)],
        max_depth=2,
        options=options,
    )

with timer("ODE solver time"):
    result_pade = solver_pade.run(rho0, tlist)

with timer("Steady state solver time"):
    rho_ss_pade, ado_ss_pade = solver_pade.steady_state()
```

现在绘制结果，可以看到初始激发杂质的衰减。单看这个图信息不多，稍后我们会与 Matsubara 展开和解析解对比：

```{code-cell} ipython3
# Plot the Pade results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

axes.plot(
    tlist, expect(result_pade.states, rho0),
    'r--', linewidth=2,
    label="P11 (Pade)",
)
axes.axhline(
    expect(rho_ss_pade, rho0),
    color='r', linestyle="dotted", linewidth=1,
    label="P11 (Pade steady state)",
)

axes.set_xlabel('t', fontsize=28)
axes.legend(fontsize=12);
```

下面对 Matsubara 展开做同样计算：

```{code-cell} ipython3
# HEOM dynamics using the Matsubara approximation:

envL_mats = envL.approx_by_matsubara(Nk=Nk, tag="L")
envR_mats = envR.approx_by_matsubara(Nk=Nk, tag="R")


with timer("RHS construction time"):
    solver_mats = HEOMSolver(
        H,
        [(envL_mats, bath_L.Q), (envR_mats, bath_R.Q)],
        max_depth=2,
        options=options,
    )

with timer("ODE solver time"):
    result_mats = solver_mats.run(rho0, tlist)

with timer("Steady state solver time"):
    rho_ss_mats, ado_ss_mats = solver_mats.steady_state()
```

可以看到 Matsubara 与 Padé 结果有明显差异。

```{code-cell} ipython3
# Plot the Pade results
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 8))

axes.plot(
    tlist, expect(result_pade.states, rho0),
    'r--', linewidth=2,
    label="P11 (Pade)",
)
axes.axhline(
    expect(rho_ss_pade, rho0),
    color='r', linestyle="dotted", linewidth=1,
    label="P11 (Pade steady state)",
)

axes.plot(
    tlist, expect(result_mats.states, rho0),
    'b--', linewidth=2,
    label="P11 (Mats)",
)
axes.axhline(
    expect(rho_ss_mats, rho0),
    color='b', linestyle="dotted", linewidth=1,
    label="P11 (Mats steady state)",
)

axes.set_xlabel('t', fontsize=28)
axes.legend(fontsize=12);
```

但哪一个更准确？Matsubara 还是 Padé？

这个简化模型的优点之一是：热浴稳态电流可解析求解，因此可用解析值检验数值结果收敛性（稳态下流入与流出系统的总电流必须平衡，所以一个热浴流出的电流等于另一个热浴流入的电流）。

解析结果的详细推导与参考文献见 [QuTiP-BoFiN 论文](https://arxiv.org/abs/2010.10806)。这里我们仅做所需积分的数值计算。

```{code-cell} ipython3
def analytical_steady_state_current(bath_L, bath_R, e1):
    """ Calculate the analytical steady state current. """

    def integrand(w):
        return (2 / np.pi) * (
            bath_L.J(w) * bath_R.J(w) * (bath_L.fF(w) - bath_R.fF(w)) /
            (
                (bath_L.J(w) + bath_R.J(w))**2 +
                4 * (w - e1 - bath_L.lamshift(w) - bath_R.lamshift(w))**2
            )
        )

    def real_part(x):
        return np.real(integrand(x))

    def imag_part(x):
        return np.imag(integrand(x))

    # in principle the bounds for the integral should be rechecked if
    # bath or system parameters are changed substantially:
    bounds = [-10, 10]

    real_integral, _ = quad(real_part, *bounds)
    imag_integral, _ = quad(imag_part, *bounds)

    return real_integral + 1.0j * imag_integral


curr_ss_analytic = analytical_steady_state_current(bath_L, bath_R, e1)

print(f"Analytical steady state current: {curr_ss_analytic}")
```

要将上述解析结果与 HEOM 结果比较，需要能从 HEOM 解中计算系统到热浴的电流。在 HEOM 描述中，这些电流编码在一级辅助密度算符（ADO）中。

在下面的 `state_current(...)` 函数中，我们提取指定热浴对应的一级 ADO，并累加各项电流贡献：

```{code-cell} ipython3
def state_current(ado_state, bath_tag):
    """ Determine current from the given bath (either "R" or "L") to
        the system in the given ADO state.
    """
    level_1_aux = [
        (ado_state.extract(label), ado_state.exps(label)[0])
        for label in ado_state.filter(level=1, tags=[bath_tag])
    ]

    def exp_sign(exp):
        return 1 if exp.type == exp.types["+"] else -1

    def exp_op(exp):
        return exp.Q if exp.type == exp.types["+"] else exp.Q.dag()

    return -1.0j * sum(
        exp_sign(exp) * (exp_op(exp) * aux).tr() for aux, exp in level_1_aux
    )
```

现在可以从 Padé 与 Matsubara 的 HEOM 结果中计算稳态电流：

```{code-cell} ipython3
curr_ss_pade_L = state_current(ado_ss_pade, "L")
curr_ss_pade_R = state_current(ado_ss_pade, "R")

print(f"Pade steady state current (L): {curr_ss_pade_L}")
print(f"Pade steady state current (R): {curr_ss_pade_R}")
```

```{code-cell} ipython3
curr_ss_mats_L = state_current(ado_ss_mats, "L")
curr_ss_mats_R = state_current(ado_ss_mats, "R")

print(f"Matsubara steady state current (L): {curr_ss_mats_L}")
print(f"Matsubara steady state current (R): {curr_ss_mats_R}")
```

可以看到，各热浴电流满足稳态平衡要求，但 Padé 与 Matsubara 得到的电流数值不同。

现在把三种结果放在一起比较：

```{code-cell} ipython3
print(f"Pade current (R): {curr_ss_pade_R}")
print(f"Matsubara current (R): {curr_ss_mats_R}")
print(f"Analytical curernt: {curr_ss_analytic}")
```

在本例中可以观察到：Padé 近似比 Matsubara 更接近解析电流。

若提高 Matsubara 展开保留项数（即增大 `Nk`），Matsubara 结果可进一步改进。

+++

## 电流与偏置电压关系

+++

现在绘制电流随偏置电压变化的关系（偏置参数为两个热浴的 `theta`）。

我们将对每个 `theta` 同时计算解析稳态电流和基于 Padé 关联展开的 HEOM 稳态电流。

```{code-cell} ipython3
# Theta (bias voltages)

thetas = np.linspace(-4, 4, 100)

# Setup a progress bar:

progress = IntProgress(min=0, max=2 * len(thetas))
display(progress)

# Calculate the current for the list of thetas


def current_analytic_for_theta(e1, bath_L, bath_R, theta):
    """ Return the analytic current for a given theta. """
    current = analytical_steady_state_current(
        bath_L.replace(theta=theta),
        bath_R.replace(theta=theta),
        e1,
    )
    progress.value += 1
    return np.real(current)


def current_pade_for_theta(H, bath_L, bath_R, theta, Nk):
    """ Return the steady state current using the Pade approximation. """
    bath_L = bath_L.replace(theta=theta)
    bath_R = bath_R.replace(theta=theta)

    envL = LorentzianEnvironment(bath_L.T, bath_L.mu, bath_L.gamma, bath_L.W)
    bathL = envL.approx_by_pade(Nk=Nk)
    envR = LorentzianEnvironment(bath_R.T, bath_R.mu, bath_R.gamma, bath_R.W)

    bathR = envR.approx_by_pade(Nk=Nk, tag="R")

    solver_pade = HEOMSolver(
        H, [(bathL, bath_L.Q), (bathR, bath_R.Q)], max_depth=2, options=options
    )
    rho_ss_pade, ado_ss_pade = solver_pade.steady_state()
    current = state_current(ado_ss_pade, bath_tag="R")

    progress.value += 1
    return np.real(current)


curr_ss_analytic_thetas = [
    current_analytic_for_theta(e1, bath_L, bath_R, theta) for theta in thetas
]

# The number of expansion terms has been dropped to Nk=6 to speed
# up notebook execution. Increase to Nk=10 for more accurate results.
curr_ss_pade_theta = [
    current_pade_for_theta(H, bath_L, bath_R, theta, Nk=6) for theta in thetas
]
```

下图显示：即使 `Nk=6`，HEOM 的 Padé 近似对稳态电流也已给出较好结果；将 `Nk` 增至 `10` 后可得到非常高精度。

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(
    thetas, 2.434e-4 * 1e6 * np.array(curr_ss_analytic_thetas),
    color="black", linewidth=3,
    label=r"Analytical",
)
ax.plot(
    thetas, 2.434e-4 * 1e6 * np.array(curr_ss_pade_theta),
    'r--', linewidth=3,
    label=r"HEOM Pade $N_k=10$, $n_{\mathrm{max}}=2$",
)


ax.locator_params(axis='y', nbins=4)
ax.locator_params(axis='x', nbins=4)

ax.set_xticks([-2.5, 0, 2.5])
ax.set_xticklabels([-2.5, 0, 2.5])
ax.set_xlabel(r"Bias voltage $\Delta \mu$ ($V$)", fontsize=28)
ax.set_ylabel(r"Current ($\mu A$)", fontsize=28)
ax.legend(fontsize=25);
```

## 关于

```{code-cell} ipython3
about()
```

## 测试

本节可包含测试，以验证笔记本是否生成预期输出。我们把该节放在末尾，以免影响阅读流程。请使用 `assert` 定义测试；当输出错误时，单元应执行失败。

```{code-cell} ipython3
assert np.allclose(curr_ss_pade_L + curr_ss_pade_R, 0)
assert np.allclose(curr_ss_mats_L + curr_ss_mats_R, 0)
assert np.allclose(curr_ss_pade_R, curr_ss_analytic, rtol=1e-4)
```
