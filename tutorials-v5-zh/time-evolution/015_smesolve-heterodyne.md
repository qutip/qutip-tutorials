---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 随机求解器：外差检测


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qutip import (
    SMESolver,
    about,
    coherent,
    destroy,
    mesolve,
    plot_expectation_values,
    smesolve,
)

%matplotlib inline
```

## 引言


同相检测（homodyne）和外差检测（heterodyne）都是利用光电计数器测量场正交分量的技术。同相检测（与本振共振）一次测量一个正交分量；外差检测（与本振失谐）可以同时测得两个正交分量。

对于与场耦合且被同相/外差探测器持续监测的量子系统，其演化可由随机主方程描述。本教程比较了在 QuTiP 中实现外差检测随机主方程的两种方法。


## 确定性参考解

```python
N = 15
w0 = 1.0 * 2 * np.pi
A = 0.1 * 2 * np.pi
times = np.linspace(0, 10, 201)
gamma = 0.25
ntraj = 50

a = destroy(N)
x = a + a.dag()
y = -1.0j * (a - a.dag())

H = w0 * a.dag() * a + A * (a + a.dag())

rho0 = coherent(N, np.sqrt(5.0), method="analytic")
c_ops = [np.sqrt(gamma) * a]
e_ops = [a.dag() * a, x, y]
```

```python
result_ref = mesolve(H, rho0, times, c_ops, e_ops)
```

```python
plot_expectation_values(result_ref);
```

## 外差实现 #1

<!-- #region -->
Milburn 形式下的外差随机主方程为

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + \gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_1(t) \sqrt{\gamma} \mathcal{H}[a] \rho(t) + \frac{1}{\sqrt{2}} dW_2(t) \sqrt{\gamma} \mathcal{H}[-ia] \rho(t)$

其中 $\mathcal{D}$ 是标准 Lindblad 耗散超算符，$\mathcal{H}$ 的定义同前，
$dW_i(t)$ 为正态分布增量，满足 $E[dW_i(t)] = \sqrt{dt}$。


在 QuTiP 中，可通过随机主方程求解器 ``smesolve`` / ``SMESolver`` 并设置外差检测来实现。
<!-- #endregion -->

$x$ 与 $y$ 正交分量对应的外差电流为

$J_x(t) = \sqrt{\gamma}\left<x\right> + \sqrt{2} \xi(t)$

$J_y(t) = \sqrt{\gamma}\left<y\right> + \sqrt{2} \xi(t)$

其中 $\xi(t) = \frac{dW}{dt}$。

在 QuTiP 中，这些测量量由传入 ``sc_ops`` 的算符构造。

```python
options = {"store_measurement": True, "map": "parallel"}

result = smesolve(
    H,
    rho0,
    times,
    sc_ops=c_ops,
    heterodyne=True,
    e_ops=e_ops,
    ntraj=ntraj,
    options=options,
)
```

```python
plot_expectation_values([result, result_ref])
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times[1:], m[0, 0, :].real, "b", alpha=0.05)
    ax.plot(times[1:], m[0, 1, :].real, "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_ylim(-10, 10)
ax.set_xlim(0, times.max())
ax.set_xlabel("time", fontsize=12)
ax.plot(times[1:], np.mean(result.measurement, axis=0)[0, 0, :].real, "k", lw=2)
ax.plot(times[1:], np.mean(result.measurement, axis=0)[0, 1, :].real, "k", lw=2)
```

## 外差实现 #2：用两个同相测量表示

<!-- #region -->
也可以把外差方程写成

$\displaystyle d\rho(t) = -i[H, \rho(t)]dt + \frac{1}{2}\gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_1(t) \sqrt{\gamma} \mathcal{H}[a] \rho(t) + \frac{1}{2}\gamma\mathcal{D}[a]\rho(t) dt + \frac{1}{\sqrt{2}} dW_2(t) \sqrt{\gamma} \mathcal{H}[-ia] \rho(t)$


这对应于两个随机塌缩算符的同相检测：$A_1 = \sqrt{\gamma} a / \sqrt{2}$ 与 $A_2 = -i \sqrt{\gamma} a / \sqrt{2}$。
<!-- #endregion -->

此时两个同相电流为

$J_x(t) = \sqrt{\gamma/2}\left<x\right> + \xi(t)$

$J_y(t) = \sqrt{\gamma/2}\left<y\right> + \xi(t)$

其中 $\xi(t) = \frac{dW}{dt}$。

但我们希望得到的是 $x,y$ 正交分量对应的外差形式：

$J_x(t) = \sqrt{\gamma}\left<x\right> + \sqrt{2}\xi(t)$

$J_y(t) = \sqrt{\gamma}\left<y\right> + \sqrt{2}\xi(t)$

在 QuTiP 中，可以使用预定义的同相求解器来解这个问题，并重新缩放 `m_ops` 与 `dW_factors`。

```python
options = {
    "method": "platen",
    "dt": 0.001,
    "store_measurement": True,
    "map": "parallel",
}
sc_ops = [np.sqrt(gamma / 2) * a, -1.0j * np.sqrt(gamma / 2) * a]

solver = SMESolver(H, sc_ops=sc_ops, heterodyne=False, options=options)
solver.m_ops = [np.sqrt(gamma) * x, np.sqrt(gamma) * y]
solver.dW_factors = [np.sqrt(2), np.sqrt(2)]
result = solver.run(rho0, times, e_ops=e_ops, ntraj=ntraj)
```

```python
plot_expectation_values([result, result_ref])
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

for m in result.measurement:
    ax.plot(times[1:], m[0, :].real, "b", alpha=0.05)
    ax.plot(times[1:], m[1, :].real, "r", alpha=0.05)

ax.plot(times, result_ref.expect[1], "b", lw=2)
ax.plot(times, result_ref.expect[2], "r", lw=2)

ax.set_xlim(0, times.max())
ax.set_ylim(-25, 25)
ax.set_xlabel("time", fontsize=12)
ax.plot(
    times[1:], np.array(result.measurement).mean(axis=0)[0, :].real, "k", lw=2
)
ax.plot(
    times[1:], np.array(result.measurement).mean(axis=0)[1, :].real, "k", lw=2
)
```

## 常见问题

对某些系统，数值误差累积可能导致得到的密度矩阵变得不物理。

```python
options = {
    "method": "euler",
    "dt": 0.1,
    "store_states": True,
    "store_measurement": True,
    "map": "parallel",
}

result = smesolve(
    H,
    rho0,
    np.linspace(0, 2, 21),
    sc_ops=c_ops,
    heterodyne=True,
    e_ops=e_ops,
    ntraj=ntraj,
    options=options,
)

result.expect
```

```python
result.states[-1].full()
```

```python
sp.linalg.eigh(result.states[10].full(), eigvals_only=True)
```

通过减小 ``dt``（更小积分步长）可以降低数值误差。
所选求解算法也会影响收敛性与数值误差。
常用算法有：  
- euler：阶数 0.5，速度最快但精度最低；也是唯一接受非对易 `sc_ops` 的算法。
- rouchon：阶数约 1.0，设计目标是保持密度矩阵物理性。
- taylor1.5：阶数 1.5，收敛性好且速度通常可接受。

要列出全部可用积分器，可用 ``SMESolver.avail_integrators()``。

```python
SMESolver.avail_integrators()
```

## 关于

```python
about()
```

```python

```
