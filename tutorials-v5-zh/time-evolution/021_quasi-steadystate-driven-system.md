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

# 稳态：时间依赖（周期）量子系统

Authors: J.R. Johansson and P.D. Nation

Updated by: M. Gobbo (2024)

### 简介
本 notebook 将用 `steadystate()`、`propagator_steadystate()`
和 `steadystate_floquet()` 三种方法求解受驱动量子比特的稳态，
并与 QuTiP 中主方程求解器 `mesolve()` 的结果比较。

关于 QuTiP 中稳态求解的更多内容见
[这里](https://qutip.readthedocs.io/en/latest/guide/guide-steady.html)。

### 导入
这里导入示例所需模块。

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from qutip import (
    about,
    basis,
    destroy,
    expect,
    mesolve,
    steadystate,
    propagator,
    propagator_steadystate,
    sigmax,
    sigmaz,
    steadystate_floquet,
)
```

### 系统设置
考虑如下受驱动系统哈密顿量：

$$ H = - \frac{\Delta}{2} \sigma_x - \frac{\epsilon_0}{2} \sigma_z + \frac{A}{2} \sigma_z \sin(\omega t) $$

同时假设系统与外部热浴耦合，
耦合常数为 $\kappa_1$，热浴温度通过平均光子数 $\langle n \rangle$ 表征。
此外再加入一个由常数 $\kappa_2$ 描述的相位波动塌缩通道。

```python
# Parameters
delta = (2 * np.pi) * 0.3
eps_0 = (2 * np.pi) * 1.0
A = (2 * np.pi) * 0.05
w = (2 * np.pi) * 1.0
kappa_1 = 0.15
kappa_2 = 0.05

# Operators
sx = sigmax()
sz = sigmaz()
sm = destroy(2)

# Non-driving Hamiltonian
H0 = -delta / 2.0 * sx - eps_0 / 2.0 * sz

# Driving Hamiltonian
H1 = A / 2.0 * sz
args = {"w": w}

# Total Hamiltonian
H = [H0, [H1, "np.sin(w*t)"]]

# Collapse operators
c_op_list = []

# Thermal population
n_th = 0.5

# Relaxation
rate = kappa_1 * (1 + n_th)

if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sm)

# Excitation
rate = kappa_1 * n_th

if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sm.dag())

# Dephasing
rate = kappa_2
if rate > 0.0:
    c_op_list.append(np.sqrt(rate) * sz)
```

### 时间演化

```python
# Period
T = 2 * np.pi / w

# Simulation time
t_list = np.linspace(0, 50, 500)

# Initial state
psi_0 = basis(2, 0)
psi_1 = basis(2, 1)
```

### 主方程方法

```python
# Solve with the Master equation
output = mesolve(H, psi_0, t_list, c_op_list, [psi_1 * psi_1.dag()], args)
prob_me = output.expect[0]
```

### steadystate 方法
该函数提供多种稳态求解方法，各有优缺点。
本例使用默认参数 `method="direct"`、`solver="solve"`，
即求解

$$
\frac{d\hat \rho_{ss}}{dt} = \mathcal{L}\hat \rho_{ss} = 0
$$

`H0` 是无驱动哈密顿量，`c_ops` 是塌缩算符列表。
输出 `rho_ss` 即系统稳态解。
默认直接法可达到机器精度，但内存开销较大。
其高内存需求主要来自系统 Liouvillian 带宽较大，
及 LU 分解中的 fill-in（额外非零元）增长。

稳态方法与求解器细节可参考
P. D. Nation 的
[Steady-state solution methods for open quantum optical systems](https://arxiv.org/abs/1504.06768)。

```python
# Evaluate the steady-state using the steadystate method
rho_ss = steadystate(H0, c_op_list, method="direct", solver="solve")
prob_ss = expect(psi_1 * psi_1.dag(), rho_ss)
```

### propagator 方法
这里使用 `propagator` 方法求传播子 $U(t)$，满足
$$
\psi(t) = U(t)\psi(0) \qquad \text{or} \qquad \rho_\text{vec}(t) = U(t)\rho_\text{vec}(0)
$$
其中 $\rho_\text{vec}$ 是密度矩阵向量化表示。

`H` 是时间依赖哈密顿量，`T` 是传播子评估时长。
若传入单个时间，则计算从 0 到 T 的传播子。
`c_ops` 是塌缩算符列表，`args` 是时间依赖回调参数。

随后 `propagator_steadystate` 基于先前传播子，
求解“反复作用传播子”下的稳态。
计算上等价于：求 $U(t)$ 的本征值与本征态，
找到最接近 1 的本征值对应稳态，并构造归一化密度矩阵 `rho_pss`。

```python
# Evaluate the steady-state using the propagator method
U = propagator(H, T, c_op_list, args)
rho_pss = propagator_steadystate(U)
prob_pss = expect(psi_1 * psi_1.dag(), rho_pss)
```

### Floquet 方法
最后，`steadystate_floquet` 用于带正弦时间依赖驱动的系统稳态。
由 `H0`、`c_ops` 与 `H1` 构建相应 Liouvillian 超算符：
$\mathcal{L}_0$、$\mathcal{L}_m$、$\mathcal{L}_p$。
前者对应无驱动部分，后两者对应驱动项正/负频分量。
整体写为
$$
\mathcal{M} = \mathcal{L}_0 + \mathcal{L}_m e^{-i\omega t} + \mathcal{L}_p e^{i\omega t}
$$
其中 $\omega$ 为驱动频率。
最后用该超算符 $\mathcal{M}$ 调用前述 `steadystate` 思路求稳态。

```python
# Evaluate the steady-state using the Floquet method
rho_fss = steadystate_floquet(H0, c_op_list, H1, w)
prob_fss = expect(psi_1 * psi_1.dag(), rho_fss)
```

### 结果

```python
# Figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot
ax.plot(t_list, prob_me, label="Master equation")
ax.plot(t_list, prob_ss * np.ones(t_list.shape[0]), label="Steady state")
ax.plot(t_list, prob_pss * np.ones(t_list.shape[0]), label="Propagator steady state")
ax.plot(t_list, prob_fss * np.ones(t_list.shape[0]), label="Floquet steady state")
ax.set_ylim(0, 1)

# Inset
ax_inset = inset_axes(
    ax,
    width="60%",
    height="80%",
    loc="center",
    bbox_to_anchor=(0.2, 0.45, 0.5, 0.45),
    bbox_transform=ax.transAxes,
)
ax_inset.plot(t_list, prob_me, label="Master Equation")
ax_inset.plot(t_list, prob_ss * np.ones(t_list.shape[0]), label="Steady state")
ax_inset.plot(
    t_list, prob_pss * np.ones(t_list.shape[0]), label="Propagator steady state"
)
ax_inset.plot(t_list, prob_fss * np.ones(t_list.shape[0]), label="Floquet steady state")
ax_inset.set_xlim(40, 50)
ax_inset.set_ylim(0.25, 0.3)
ax_inset.set_xticks([40, 45, 50])
ax_inset.set_yticks([0.25, 0.27, 0.3])
mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

# Labels
ax.set_xlabel("Time")
ax.set_ylabel("$P(|1\\rangle)$")
ax.set_title("Excitation probabilty of qubit")
ax.legend()
plt.show()
```

### 环境信息

```python
about()
```

### 测试

```python
np.testing.assert_allclose(prob_ss, np.mean(prob_me[200:]), atol=1e-2)
np.testing.assert_allclose(prob_pss, np.mean(prob_me[200:]), atol=1e-2)
np.testing.assert_allclose(prob_fss, np.mean(prob_me[200:]), atol=1e-2)
np.testing.assert_allclose(prob_ss, prob_pss, atol=1e-2)
np.testing.assert_allclose(prob_ss, prob_fss, atol=1e-2)
np.testing.assert_allclose(prob_pss, prob_fss, atol=1e-2)
```
