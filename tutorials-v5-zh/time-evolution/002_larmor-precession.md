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

# 薛定谔方程求解器：拉莫尔进动

Author: C. Staufenbiel, 2022

### 简介

本 notebook 将引导你在 QuTiP 中构建薛定谔方程并使用相应求解器获得时间演化。
我们将以拉莫尔进动为例，介绍
[`qutip.sesolve()`](https://qutip.readthedocs.io/en/latest/apidoc/solver.html#module-qutip.solver.sesolve)
的基本用法。

你也可以在[这里](https://qutip.readthedocs.io/en/latest/guide/guide-dynamics.html)查看更多 QuTiP 时间演化相关内容。

### 设置

首先导入所需函数、类与模块。
```python
import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip import Bloch, QobjEvo, basis, sesolve, sigmay, sigmaz

%matplotlib inline
```

我们先构造一个任意量子比特态（两个基态的叠加态），
并使用 `qutip.Bloch` 在 Bloch 球上可视化。

```python
psi = (2.0 * basis(2, 0) + basis(2, 1)).unit()
b = Bloch()
b.add_states(psi)
b.show()
```

### 恒定磁场下的仿真

定义一个简单哈密顿量，并用 `qutip.sesolve` 求解薛定谔方程。
该哈密顿量描述沿 z 轴的恒定磁场。
在 QuTiP 中，可用对应泡利矩阵 `qutip.sigmaz()` 表示。

要对该哈密顿量求解薛定谔方程，需要传入：
哈密顿量、初态、希望模拟的时间点，以及在这些时间点上要评估的可观测量集合。

这里我们关注 $\sigma_y$ 期望值的时间演化。
如下将这些参数传给 `sesolve`。

```python
# simulate the unitary dynamics
H = sigmaz()
times = np.linspace(0, 10, 100)
result = sesolve(H, psi, times, [sigmay()])
```

`result.expect` 保存了传给 `sesolve` 的各时刻期望值。
`result.expect` 是二维数组：第一维对应不同期望算符。

上面只传了 `sigmay()` 一个期望算符，
因此其结果可由 `result.expect[0]` 访问。
下面绘制该期望值演化。

```python
plt.plot(times, result.expect[0])
plt.xlabel("Time"), plt.ylabel("<sigma_y>")
plt.show()
```

上面我们把 `sigmay()` 传给 `sesolve` 来直接计算期望值。
如果该参数传空列表，`sesolve` 将返回 `times` 各时刻系统量子态。
可通过 `result.states` 访问这些状态，
例如用于在 Bloch 球上绘制进动轨迹。
若求解耗时较长，返回状态也很有用，便于后续计算不同量而无需重复求解。

```python
res = sesolve(H, psi, times, [])
b = Bloch()
b.add_states(res.states[1:30])
b.show()
```

## 变化磁场下的仿真

上面传给 `sesolve` 的是常量哈密顿量。
在 QuTiP 中，常量算符由 `Qobj` 表示。
不过 `sesolve` 也支持时间依赖算符，
在 QuTiP 中由 [`QobjEvo`](https://qutip.readthedocs.io/en/latest/apidoc/time_dep.html#qutip.core.cy.qobjevo.QobjEvo) 表示。
本节中我们将定义线性和周期变化的磁场强度，
并观察 $\sigma_y$ 期望值的变化。
关于 `QobjEvo` 的更多信息见[此 notebook](https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/qobjevo.ipynb)。

先定义两个磁场强度函数。
传给 `QobjEvo` 的函数需接收两个参数：时间与可选参数字典。


```python
def linear(t, args):
    return 0.3 * t


def periodic(t, args):
    return np.cos(0.5 * t)


# Define QobjEvos
H_lin = QobjEvo([[sigmaz(), linear]], tlist=times)
H_per = QobjEvo([[sigmaz(), periodic]], tlist=times)
```

随后与前节类似，使用 `sesolve` 求解薛定谔方程。

```python
result_lin = sesolve(H_lin, psi, times, [sigmay()])
result_per = sesolve(H_per, psi, times, [sigmay()])


# Plot <sigma_y> for linear increasing field strength
plt.plot(times, result_lin.expect[0])
plt.xlabel("Time"), plt.ylabel("<sigma_y>")
plt.show()
```

可看到拉莫尔进动频率随时间增大。
这正是时间依赖哈密顿量导致的直接结果。
同样地，我们可绘制周期变化磁场对应结果。

```python
plt.plot(times, result_per.expect[0])
plt.xlabel("Time"), plt.ylabel("<sigma_y>")
plt.show()
```

### 总结
我们可以用 `sesolve` 求解幺正时间演化。
它不仅适用于常量哈密顿量，也支持借助 `QobjEvo` 表示的时间依赖哈密顿量。

### 环境信息

```python
qutip.about()
```

### 测试

这一部分可用于验证 notebook 是否产生预期输出。
我们将测试放在末尾，以免影响阅读体验。
请使用 `assert` 定义测试，这样输出不符合预期时单元会报错。

```python
assert np.allclose(result.expect[0][0], 0)
assert np.allclose(result_lin.expect[0][0], 0)
assert np.allclose(result_per.expect[0][0], 0)
assert 1 == 1
```
