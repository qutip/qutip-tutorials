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

# 使用 Dysolve 计算时间传播子

Author: Mathis Beaudoin, 2025

### 简介

本 notebook 演示如何在 QuTiP 中使用 Dysolve 计算时间传播子。
Dysolve 针对如下形式的哈密顿量：$H(t) = H_0 + \cos(\omega t)X$，其中
$H_0$ 是基础哈密顿量，$X$ 是微扰项。对于这类哈密顿量，
它通常比通用方法表现更好。未来还将支持更复杂的振荡微扰。
该功能仍在开发中，后续会加入更多特性。
关于 Dysolve 的细节可参考文档中对应指南。

目前可使用 QuTiP 求解器中的 `DysolvePropagator` 类与 `dysolve_propagator` 函数。
它们的使用结构与同样用于传播子计算的 `Propagator` 类和 `propagator` 函数类似。

下面先导入需要的包。

```python
from qutip.solver.dysolve_propagator import DysolvePropagator, dysolve_propagator
from qutip.solver.propagator import propagator
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, CoreOptions, about
import numpy as np
```

### 使用 `DysolvePropagator` 的单量子比特示例

我们先定义 $H_0$、$X$ 与 $\omega$。
例如：$H(t) = \sigma_z + \cos(10t)\sigma_x$。

```python
H_0 = sigmaz()
X = sigmax()
omega = 10.0
```

可以设置一些选项。
`max_order` 表示传播子近似计算的阶数，整数越高通常越精确（但耗时也更高）。
`a_tol` 是计算中的绝对容差。
此外，时间传播子可用步长为 `max_dt` 的子传播子拼接得到。
例如 `max_dt=0.25` 时，$U(1,0)$ 由
$U(0.25,0)$、$U(0.5,0.25)$、$U(0.75,0.5)$、$U(1,0.75)$ 相乘得到。
当演化时间较长时，这有助于提升精度。
在本例中，我们保留 `a_tol` 与 `max_dt` 默认值，并将 `max_order` 设为 5。

```python
options = {'max_order': 5}
```

现在即可初始化实例。

```python
dy = DysolvePropagator(H_0, X, omega, options=options)
```

之后通过调用该实例并传入初末时间即可计算传播子。
也可以仅给出终止时间，此时起始时间默认为 0。

```python
t_i = -1
t_f = 1
U = dy(t_f, t_i)
```

这将返回单个传播子 $U(t_f = 1, t_i = -1)$。
为验证结果正确性，我们将其与 `propagator` 的结果比较。

```python
# Solve using propagator
def X_coeff(t, omega):
    return np.cos(omega * t)

H = [H_0, [X, X_coeff]]
args = {'omega': omega}
prop = propagator(
    H, [t_i, t_f], args=args, options={"atol": 1e-10, "rtol": 1e-8}
)

# Comparison
with CoreOptions(atol=1e-10, rtol=1e-6):
    assert U == prop[1]
```

### 使用 `dysolve_propagator` 的双量子比特示例

与前一示例流程类似。

```python
# Define the system
H_0 = tensor(sigmax(), sigmaz()) + tensor(qeye(2), sigmay())
X = tensor(qeye(2), sigmaz())
omega = 5.0

# Keep options to default
```

`dysolve_propagator` 可以接收多个时间点。
若传入单个时间值，返回单个传播子 $U(t,0)$；
若传入时间列表，则返回传播子列表
$[U(\text{times}[i], \text{times}[0])]$（对所有 $i$）。

```python
times = [-0.1, 0, 0.1]
Us = dysolve_propagator(H_0, X, omega, times)
```

同样地，我们与 `propagator` 结果进行比较。

```python
# Solve using propagator
def X_coeff(t, omega):
    return np.cos(omega * t)

H = [H_0, [X, X_coeff]]
args = {'omega': omega}
props = propagator(
    H, times, args=args, options={"atol": 1e-10, "rtol": 1e-8}
)

# Comparison
with CoreOptions(atol=1e-10, rtol=1e-6):
    assert Us == props
```

### 环境信息

```python
about()
```
