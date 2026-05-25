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

# 主方程求解器：真空 Rabi 振荡

Authors: J.R. Johansson and P.D. Nation

Slight modifications: C. Staufenbiel (2022)

本 notebook 演示如何使用主方程求解器 `qutip.mesolve`
模拟 Jaynes-Cumming 模型中的量子真空 Rabi 振荡。
我们也考虑该模型的耗散版本，
即腔与原子均与外部环境耦合。


关于主方程求解器背后的理论，请见[文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-master.html#non-unitary-evolution)。


### 包导入

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import about, basis, destroy, mesolve, qeye, sigmaz, tensor

%matplotlib inline
```

# 简介

Jaynes-Cumming 模型是描述量子光-物质相互作用的最简模型：
单个双能级原子与单个电磁腔模相互作用。
该系统哈密顿量（偶极相互作用形式）为

$H = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger + a)(\sigma_- + \sigma_+)$

若采用旋波近似（RWA），则

$H_{\rm RWA} = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger\sigma_- + a\sigma_+)$

其中 $\omega_c$ 与 $\omega_a$ 分别是腔与原子频率，$g$ 为耦合强度。

本例还考虑系统与外部环境耦合，
因此需使用主方程求解器 `qutip.mesolve`。
环境耦合通过塌缩算符描述（见[文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-master.html#non-unitary-evolution)）。
这里考虑腔的两个塌缩算符 $C_1, C_2$（对应光子湮灭与产生）
以及原子的一个塌缩算符 $C_3$：

$C_1 = \sqrt{\kappa (1+\langle n \rangle)} \; a$

$C_2 = \sqrt{\kappa \langle n \rangle}\; a^\dagger$

$C_3 = \sqrt{\gamma} \; \sigma_-$

其中 $\langle n \rangle$ 是环境平均光子数。
设 $\langle n \rangle=0$ 时，将去除光子产生项，仅保留光子湮灭。

### 问题参数

这里使用单位制 $\hbar = 1$：

```python
N = 15  # number of cavity fock states
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
n_th_a = 0.0  # temperature in frequency units
use_rwa = True

tlist = np.linspace(0, 40, 100)
```

### 设置算符、哈密顿量与初态

这里定义复合系统（腔+原子）的初态与算符。
通过张量积构造：第一部分对应腔，第二部分对应原子。
我们取原子初始处于激发态，腔处于基态。

初态由“腔基态 + 原子激发态”组成。
同时定义复合系统中的塌缩算符，
以及包含/不包含旋波近似的哈密顿量。

```python
# intial state
psi0 = tensor(basis(N, 0), basis(2, 0))

# collapse operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2).dag())
sz = tensor(qeye(N), sigmaz())

# Hamiltonian
if use_rwa:
    H = wc * a.dag() * a + wa / 2 * sz + g * (a.dag() * sm + a * sm.dag())
else:
    H = wc * a.dag() * a + wa / 2 * sz + g * (a.dag() + a) * (sm + sm.dag())
```

### 创建耗散塌缩算符列表

构建塌缩算符列表 `c_ops`，后续传入 `qutip.mesolve`。
三个耗散过程各对应一个塌缩算符。

```python
c_op_list = []

# Photon annihilation
rate = kappa * (1 + n_th_a)
c_op_list.append(np.sqrt(rate) * a)

# Photon creation
rate = kappa * n_th_a
c_op_list.append(np.sqrt(rate) * a.dag())

# Atom annihilation
rate = gamma
c_op_list.append(np.sqrt(rate) * sm)
```

### 系统演化

使用 Lindblad 主方程求解器 `qutip.mesolve` 演化系统，
并通过第五个参数 `[a.dag()*a, sm.dag()*sm]`
请求返回 $a^\dagger a$ 和 $\sigma_+\sigma_-$ 的期望值。

```python
output = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])
```

### 结果可视化

绘制腔与原子的激发概率（即上一步求得的期望值）。
可清楚看到能量在腔与原子之间相干往返交换。

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist, output.expect[0], label="Cavity")
ax.plot(tlist, output.expect[1], label="Atom excited state")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability")
ax.set_title("Vacuum Rabi oscillations at T={}".format(n_th_a));
```

### 非零温度
上面设定 $T = 0$，因此忽略了环境导致的光子产生。
将对应变量设为正值可激活该项，并执行同样计算。
与前图相比，可见腔中能量高于原子。

```python
# set temperature
n_th_a = 2.0

# set collapse operators
c_op_list = []
rate = kappa * (1 + n_th_a)
c_op_list.append(np.sqrt(rate) * a)
rate = kappa * n_th_a
c_op_list.append(np.sqrt(rate) * a.dag())
rate = gamma
c_op_list.append(np.sqrt(rate) * sm)

# evolve system
output_temp = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a, sm.dag() * sm])

# plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tlist, output_temp.expect[0], label="Cavity")
ax.plot(tlist, output_temp.expect[1], label="Atom excited state")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability")
ax.set_title("Vacuum Rabi oscillations at T={}".format(n_th_a));
```

### 软件版本：

```python
about()
```

### 测试

```python
# sum of atom and cavity
atom_and_cavity = np.array(output.expect[0]) + np.array(output.expect[1])
assert np.all(np.diff(atom_and_cavity) <= 0.0)

# frequency for analytical solution (with RWA)
output_no_cops = mesolve(H, psi0, tlist, [], [a.dag() * a, sm.dag() * sm])
freq = 1 / 4 * np.sqrt(g**2 * (N + 1))
assert np.allclose(output_no_cops.expect[1],
                   (np.cos(tlist * freq)) ** 2, atol=10**-3)
```
