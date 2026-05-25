---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Floquet 形式主义

Author: C. Staufenbiel, 2022

参考了 P.D. Nation 与 J.R. Johannson 的
[floquet notebook](https://github.com/qutip/qutip-notebooks/blob/master/examples/floquet-dynamics.ipynb)
以及 [qutip 文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-floquet.html)。

### 简介

在 [floquet_solver notebook](011_floquet_solver.md) 中，
我们介绍了用 Floquet 形式主义求解薛定谔方程和主方程的两个函数
（`fsesolve` 与 `fmmesolve`）。
本 notebook 将转向求解器底层使用的 `FloquetBasis` 类，
重点关注 Floquet 模与准能量。

关于 QuTiP 中 Floquet 形式主义实现的更多信息，见
[官方文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-floquet.html)。

### 导入

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, expect, FloquetBasis,
                   num, plot_wigner, ket, sesolve, sigmax, sigmaz)
```

### 系统设置
为与文档保持一致，考虑如下驱动系统哈密顿量：

$$ H = - \frac{\Delta}{2} \sigma_x - \frac{\epsilon_0}{2} \sigma_z + \frac{A}{2} \sigma_x sin(\omega t) $$

```python
# Constants
delta = 0.2 * 2 * np.pi
eps0 = 1 * 2 * np.pi
A = 2.5 * 2 * np.pi
omega = 1.0 * 2 * np.pi
T = 2 * np.pi / omega

# Hamiltonian
H = [
    -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz(),
    [A / 2 * sigmax(), "sin({w}*t)".format(w=omega)],
]
```

### Floquet 模与准能量
对周期哈密顿量，薛定谔方程解可表示为 Floquet 模 $\phi_\alpha(t)$
与准能量 $\epsilon_\alpha$。
可通过 `FloquetBasis(H, T)` 及其 `.mode(t=0)` 获取 $t=0$ 时结果。

例如，绘制 $t=0$ 时第一个 Floquet 模的 Wigner 分布：

```python
fbasis = FloquetBasis(H, T)
f_modes_t0 = fbasis.mode(t=0)
plot_wigner(f_modes_t0[0]);
```

对上面系统有两个准能量。
可在改变驱动强度 $A$ 时绘制它们的变化。

准能量通过 `FloquetBasis` 的 `.e_quasi` 属性访问。
传入 `sort=True` 可确保按从低到高排序：

```python
A_list = np.linspace(1.0 * omega, 4.5 * omega, 20)
quasienergies1, quasienergies2 = [], []
for A_tmp in A_list:
    # temporary Hamiltonian
    H_tmp = [
        -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz(),
        [A_tmp / 2 * sigmax(), "sin({w}*t)".format(w=omega)],
    ]
    # floquet modes and quasienergies
    e1, e2 = FloquetBasis(H_tmp, T, sort=True).e_quasi
    quasienergies1.append(e1), quasienergies2.append(e2)
```

```python
plt.scatter(A_list / omega, quasienergies1, label="e1")
plt.scatter(A_list / omega, quasienergies2, label="e2")
plt.xlabel("A / w"), plt.ylabel("Quasienergies")
plt.legend();
```

### 用 Floquet 模做时间演化
要计算随机初态 $\psi(0)$ 的时间演化，
需先在 Floquet 基（由 Floquet 模构成）中展开：

$$ \psi(0) = \sum_\alpha c_\alpha \phi_\alpha(0) $$

系数 $c_\alpha$ 可通过 `.to_floquet_basis` 计算：

```python
# Define an initial state:
psi0 = ket("0") + ket("1")
psi0 = psi0.unit()

# Decompose the initial state into its components in the Floquet modes:
f_coeff = fbasis.to_floquet_basis(psi0, t=0)
f_coeff
```

后续时刻 $t>0$ 的 Floquet 模可由传播子 $U(t,0)$ 得到：

$$ \phi_\alpha(t) = exp(-i\epsilon_\alpha t / \hbar) \, U(t,0) \, \phi_\alpha(0) $$

在 QuTiP 中由 `FloquetBasis.mode(t=t)` 实现。
下面将初态传播到 $t=1$：

```python
t = 1.0
f_modes_t1 = fbasis.mode(t=t)
f_modes_t1
```

传播后的 Floquet 模可组合得到时刻 `t` 的系统态 $\psi(t)$。

`.from_floquet_basis(f_coeff, t)` 用于完成该步骤：

```python
psi_t = fbasis.from_floquet_basis(f_coeff, t)
psi_t
```

### 预计算并复用一个周期内的 Floquet 模

Floquet 模与哈密顿量同周期：

$$ \phi_\alpha(t + T) = \phi_\alpha(t) $$

因此只需在 $t \in [0,T]$ 上计算 Floquet 模，
即可外推任意时刻系统态。

`FloquetBasis` 支持通过 `precompute` 参数
在第一周期多个时刻预计算 Floquet 模传播子：

```python
tlist = np.linspace(0, T, 50)
fbasis = FloquetBasis(H, T, precompute=tlist)
```

随后仍可通过 `FloquetBasis.from_floquet_basis(...)` 计算波函数 $\psi(t)$，
此时各时刻 Floquet 模已预计算。

下面用预计算模式计算第一周期内数算符期望值：

```python
p_ex_period = []
for t in tlist:
    psi_t = fbasis.from_floquet_basis(f_coeff, t)
    p_ex_period.append(expect(num(2), psi_t))

plt.plot(tlist, p_ex_period)
plt.ylabel("Occupation prob."), plt.xlabel("Time");
```

第一周期预计算的模式也可用于后续周期。
但若时刻 $t'$ 不满足 $t' = t + nT$
（其中 $t$ 是预计算时刻之一），
则 `FloquetBasis` 会临时计算 $t'$ 对应模式，
并遗忘一个已缓存模式。

在内部，`FloquetBasis` 使用 `qutip.Propagator` 管理预计算模式。
细节见 `Propagator` 文档。
若需要，可通过 `FloquetBasis.U` 直接访问该传播子对象。

下面在前十个周期演示该机制。
若 `tlist` 与已预计算第一周期时刻对齐，
则后续周期期望值计算应较快：

```python
p_ex = []
tlist_10_periods = np.linspace(0, 10 * T, 10 * len(tlist))
for t in tlist_10_periods:
    psi_t = fbasis.from_floquet_basis(f_coeff, t)
    p_ex.append(expect(num(2), psi_t))

# Plot the occupation Probability
plt.plot(tlist_10_periods, p_ex, label="Ten periods - precomputed")
plt.plot(tlist, p_ex_period, label="First period - precomputed")
plt.legend(loc="upper right")
plt.xlabel("Time"), plt.ylabel("Occupation prob.");
```

### 环境信息

```python
about()
```

## 测试

```python
# compute prediction using sesolve
res_sesolve = sesolve(H, psi0, tlist_10_periods, [num(2)])
assert np.allclose(res_sesolve.expect[0], p_ex, atol=0.15)
```
