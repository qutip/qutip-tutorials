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

# HEOM 5b：离散玻色模耦合杂质与费米引线

+++

## 引言

这里建模一个与两个电子引线（或电子库）耦合的单费米子系统（如单量子点、分子晶体管等），并进一步耦合一个离散玻色（振动）模式。

实现上主要遵循 Christian Schinabeck 博士论文及相关工作中的定义：https://open.fau.de/items/36fdd708-a467-4b59-bf4e-4a2110fbc431。本示例尤其复现了 https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.201407 的部分结果。

记号约定：

* $K=L/R$ 表示左/右引线
* $\sigma=\pm$ 表示输入/输出

我们为引线选用在化学势附近有峰值的 Lorentz 谱密度。这样可简化关联函数记号（必要时也可放宽该假设）。

$$J(\omega) = \frac{\Gamma  W^2}{((\omega-\mu_K)^2 +W^2 )}$$

费米分布函数为：

$$f_F (x) = (\exp(x) + 1)^{-1}$$

两者结合可写出关联函数：

$$C^{\sigma}_K(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} d\omega e^{\sigma i \omega t} \Gamma_K(\omega) f_F[\sigma\beta(\omega - \mu)]$$

与玻色情形类似，这里也可采用 Matsubara、Padé 或拟合方法。

Padé 分解将费米分布近似为

$$f_F(x) \approx f_F^{\mathrm{approx}}(x) = \frac{1}{2} - \sum_l^{l_{max}} \frac{2k_l x}{x^2 + \epsilon_l^2}$$

$k_l$ 与 $\epsilon_l$ 是 J. Chem. Phys. 133, 10106 中定义的系数。

对关联函数积分可得


$$C_K^{\sigma}(t) \approx \sum_{l=0}^{l_{max}} \eta_K^{\sigma_l} e^{-\gamma_{K,\sigma,l}t}$$

其中

$$\eta_{K,0} = \frac{\Gamma_KW_K}{2} f_F^{approx}(i\beta_K W)$$

$$\gamma_{K,\sigma,0} = W_K - \sigma i\mu_K$$ 

$$\eta_{K,l\neq 0} = -i\cdot \frac{k_m}{\beta_K} \cdot \frac{\Gamma_K W_K^2}{-\frac{\epsilon^2_m}{\beta_K^2} + W_K^2}$$

$$\gamma_{K,\sigma,l\neq 0}= \frac{\epsilon_m}{\beta_K} - \sigma i \mu_K$$

+++

## 与示例 5a 的差异

+++

本例系统与 HEOM 5a 相比有两点主要不同：

* 系统现在包含一个离散玻色模式；
* 电子引线参数 $W$ 设为 $10^4$（即宽带极限）。

新的系统哈密顿量为：

$$
H_{\mathrm{vib}} = H_{\mathrm{SIAM}} + \Omega a^{\dagger}a + \lambda (a+a^{\dagger})c{^\dagger}c.
$$

其中 $H_{\mathrm{SIAM}}$ 是单杂质哈密顿量，其余项是玻色模式哈密顿量及其与杂质的相互作用。

完整模型现在由四部分构成：

* 单杂质
* 一个离散玻色模式
* 两个费米引线。

**注意**：本例数值计算难度较高，系统与热浴组成较多。若想先熟悉费米 HEOM，建议先看示例 5a。

**注意**：为加速笔记本检查与构建，我们将玻色模式截断为 2 个模。若提高到例如 16 个玻色模，可得到更精确结果。

+++

## 设置

```{code-cell} ipython3
import contextlib
import dataclasses
import time

import numpy as np
import matplotlib.pyplot as plt

from qutip import about, destroy, qeye, tensor
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

下面设置系统哈密顿量并指定两个电子库的参数。

```{code-cell} ipython3
# Define the system Hamiltonian:

@dataclasses.dataclass
class SystemParameters:
    e1: float = 0.3  # fermion mode energy splitting
    Omega: float = 0.2  # bosonic mode energy splitting
    Lambda: float = 0.12  # coupling between fermion and boson
    Nbos: int = 2

    def __post_init__(self):
        d = tensor(destroy(2), qeye(self.Nbos))
        a = tensor(qeye(2), destroy(self.Nbos))
        self.H = (
            self.e1 * d.dag() * d +
            self.Omega * a.dag() * a +
            self.Lambda * (a + a.dag()) * d.dag() * d
        )
        self.Q = d

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


sys_p = SystemParameters()
```

```{code-cell} ipython3
# Define parameters for left and right fermionic baths.
# Each bath is a lead (i.e. a wire held at a potential)
# with temperature T and chemical potential mu.

@dataclasses.dataclass
class LorentzianBathParameters:
    lead: str
    gamma: float = 0.01  # coupling strength
    W: float = 1.0  # cut-off
    T: float = 0.025851991  # temperature (in eV)
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


# We set W = 1e4 to investigate the wide-band limit:

bath_L = LorentzianBathParameters(W=10**4, lead="L")
bath_R = LorentzianBathParameters(W=10**4, lead="R")
```

## 引线发射与吸收

接下来绘制引线的发射与吸收。

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

## 下方给出论文中的一组示例数据

这里仅展示“电流随偏置电压变化”的一组示例。一般可继续调节玻色 Fock 空间截断与关联函数展开参数，直到结果收敛。

补充说明：对于非常大的问题，这一步计算可能较慢。

```{code-cell} ipython3
def steady_state_pade_for_theta(sys_p, bath_L, bath_R, theta, Nk, Nc, Nbos):
    """ Return the steady state current using the Pade approximation. """

    sys_p = sys_p.replace(Nbos=Nbos)
    bath_L = bath_L.replace(theta=theta)
    bath_R = bath_R.replace(theta=theta)

    envL = LorentzianEnvironment(bath_L.T, bath_L.mu, bath_L.gamma, bath_L.W)
    envR = LorentzianEnvironment(bath_R.T, bath_R.mu, bath_R.gamma, bath_R.W)

    bathL = envL.approx_by_matsubara(Nk, tag="L")
    bathR = envR.approx_by_matsubara(Nk, tag="R")

    solver_pade = HEOMSolver(
        sys_p.H,
        [(bathL, sys_p.Q), (bathR, sys_p.Q)],
        max_depth=2,
        options=options,
    )
    rho_ss_pade, ado_ss_pade = solver_pade.steady_state()
    current = state_current(ado_ss_pade, bath_tag="R")

    return np.real(2.434e-4 * 1e6 * current)
```

```{code-cell} ipython3
# Parameters:

Nk = 6
Nc = 2
Nbos = 2  # Use Nbos = 16 for more accurate results

thetas = np.linspace(0, 2, 30)

# Progress bar:

progress = IntProgress(min=0, max=len(thetas))
display(progress)

currents = []

for theta in thetas:
    currents.append(steady_state_pade_for_theta(
        sys_p, bath_L, bath_R, theta,
        Nk=Nk, Nc=Nc, Nbos=Nbos,
    ))
    progress.value += 1
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 10))

ax.plot(
    thetas, currents,
    color="green", linestyle='-', linewidth=3,
    label=f"Nk = {5}, max_depth = {Nc}, Nbos = {Nbos}",
)

ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, 0.5, 1])

ax.locator_params(axis='y', nbins=4)
ax.locator_params(axis='x', nbins=4)

ax.set_xlabel(r"Bias voltage $\Delta \mu$ ($V$)", fontsize=30)
ax.set_ylabel(r"Current ($\mu A$)", fontsize=30)
ax.legend(loc=4);
```

## 关于

```{code-cell} ipython3
about()
```
