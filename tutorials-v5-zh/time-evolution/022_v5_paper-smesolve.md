---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# QuTiPv5 论文示例：随机求解器 - 同相（Homodyne）探测

Authors: Maximilian Meyer-Mölleringhof (m.meyermoelleringhof@gmail.com), Neill Lambert (nwlambert@gmail.com)

## 简介

在开放量子系统建模中，随机噪声可用于模拟大量物理现象。
在 `smesolve()` 求解器中，噪声通过连续测量引入。
这使我们能够生成在噪声测量记录条件下的系统轨迹演化。
从历史上看，这类模型常用于量子光学中描述腔输出光的同相与异相探测。
当然，该求解器本身较为通用，也可应用于其他问题。

本例考虑一个输出受同相探测的光学腔。
该腔满足如下随机主方程：

$d \rho(t) = -i [H, \rho(t)] dt + \mathcal{D}[a] \rho (t) dt + \mathcal{H}[a] \rho\, dW(t)$

其中哈密顿量

$H = \Delta a^\dagger a$

Lindblad 耗散项为

$\mathcal{D}[a] = a \rho a^\dagger - \dfrac{1}{2} a^\dagger a \rho - \dfrac{1}{2} \rho a^\dagger a$.

随机部分

$\mathcal{H}[a]\rho = a \rho + \rho a^\dagger - tr[a \rho + \rho a^\dagger]$

刻画了通过连续监测算符 $a$ 对轨迹进行条件化的效应。
项 $dW(t)$ 为 Wiener 过程增量，满足 $\mathbb{E}[dW] = 0$ 与 $\mathbb{E}[dW^2] = dt$。

注意，类似示例也可见 [QuTiP 用户指南](https://qutip.readthedocs.io/en/qutip-5.0.x/guide/dynamics/dynamics-stochastic.html#stochastic-master-equation)。

```python
import numpy as np
from matplotlib import pyplot as plt
from qutip import about, coherent, destroy, mesolve, smesolve

%matplotlib inline
```

## 问题参数

```python
N = 20  # dimensions of Hilbert space
delta = 10 * np.pi  # cavity detuning
kappa = 1  # decay rate
A = 4  # initial coherent state intensity
```

```python
a = destroy(N)
x = a + a.dag()  # operator for expectation value
H = delta * a.dag() * a  # Hamiltonian
mon_op = np.sqrt(kappa) * a  # continiously monitored operators
```

## 求解时间演化

我们计算在连续监测算符 $a$ 条件下的预测轨迹。
并将其与同一模型下、但不分辨条件轨迹的常规 `mesolve()` 结果比较。

```python
rho_0 = coherent(N, np.sqrt(A))  # initial state
times = np.arange(0, 1, 0.0025)
num_traj = 500  # number of computed trajectories
opt = {"dt": 0.00125, "store_measurement": True, "map": "parallel"}
```

```python
me_solution = mesolve(H, rho_0, times, c_ops=[mon_op], e_ops=[x])
```

```python
stoc_solution = smesolve(
    H, rho_0, times, sc_ops=[mon_op], e_ops=[x], ntraj=num_traj, options=opt
)
```

## 结果比较

我们绘制平均同相电流 $J_x = \langle x \rangle + dW / dt$ 和系统平均行为
$\langle x \rangle$（500 条轨迹）。
并与不包含条件轨迹的常规 `mesolve()` 预测比较。
由于条件化期望值在平均后应与轨迹采样无关，
预期会复现标准 `mesolve()` 的结果。

```python
stoc_meas_mean = np.array(stoc_solution.measurement).mean(axis=0)[0, :].real
```

```python
plt.figure()
plt.plot(times[1:], stoc_meas_mean, lw=2, label=r"$J_x$")
plt.plot(times, stoc_solution.expect[0], label=r"$\langle x \rangle$")
plt.plot(
    times,
    me_solution.expect[0],
    "--",
    color="gray",
    label=r"$\langle x \rangle$ mesolve",
)

plt.legend()
plt.xlabel(r"$t \cdot \kappa$")
plt.show()
```

## 参考文献

[1] [QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)


## 环境信息

```python
about()
```

## 测试

```python
assert np.allclose(
    stoc_solution.expect[0], me_solution.expect[0], atol=1e-1
), "smesolve and mesolve do not preoduce the same trajectory."
```
