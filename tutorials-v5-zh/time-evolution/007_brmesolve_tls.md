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

<!-- #region -->
# Bloch-Redfield 求解器：双能级系统

Author: C.Staufenbiel, 2022

灵感来自 P.D. Nation 的
[`brmesolve notebook`](https://github.com/qutip/qutip-notebooks/blob/master/examples/brmesolve.ipynb)。


### 简介

Bloch-Redfield 求解器是求解主方程的另一种方法。
与 Lindblad 主方程求解器 `qutip.mesolve()` 相比，
Bloch-Redfield 求解器 `qutip.brmesolve()` 的核心差异在于环境相互作用的描述方式。
在 `qutip.mesolve()` 中，我们通过塌缩算符描述耗散，
它们不一定总有直接物理解释。
而 `qutip.brmesolve()` 需要所谓*噪声功率谱*来描述耗散，
即耗散强度随频率 $\omega$ 的函数。

本 notebook 将介绍 `qutip.brmesolve()` 的基本用法，并与 `qutip.mesolve()` 比较。
关于 Bloch-Redfield 求解器的更多内容，见后续 notebook 与
[QuTiP 文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html)。

### 导入
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, bloch_redfield_tensor, brmesolve, expect,
                   hinton, liouvillian, mesolve, plot_expectation_values,
                   sigmam, sigmax, sigmay, sigmaz, steadystate, anim_hinton)
# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

%matplotlib inline
```




## 双能级系统演化

本例考虑由以下哈密顿量描述的简单双能级系统：

$$ H = \frac{\epsilon}{2} \sigma_z$$

另外定义与环境的常量耗散率 $\gamma$。

```python
epsilon = 0.5 * 2 * np.pi
gamma = 0.25
times = np.linspace(0, 10, 100)
```

先为 `qutip.mesolve()` 设置哈密顿量、初态和塌缩算符。
初态取叠加态，并观察 $\sigma_x, \sigma_y, \sigma_z$ 的期望值。

```python
# Setup Hamiltonian and initial state
H = epsilon / 2 * sigmaz()
psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()

# Setup the master equation solver
c_ops = [np.sqrt(gamma) * sigmam()]
e_ops = [sigmax(), sigmay(), sigmaz()]
result_me = mesolve(H, psi0, times, c_ops, e_ops)
```

对于 `qutip.brmesolve`，需要以厄米算符形式给出系统-热浴耦合，
并给出噪声功率谱来定义不同频率上的耦合强度。
这里我们定义：频率为正时耦合常量，负频率无耗散。
这样可使用厄米算符 `sigmax()`
（而非上面 `sigmam` 那样的非厄米算符）。

使用厄米算符可简化内部数值实现，
且在给定多个环境算符时可使不同环境算符间交叉关联项消失。

```python
a_op = [sigmax(), lambda w: gamma * (w > 0.0)]
```

此时向 Bloch-Redfield 求解器传入 `a_ops`（而非 `c_ops`）。

```python
result_brme = brmesolve(H, psi0, times, [a_op], e_ops)
```

现在比较 `e_ops` 中各算符的期望值。
如预期，`mesolve` 与 `brmesolve` 给出近似一致结果。

```python
fig, axes = plot_expectation_values(
    [result_me, result_brme], ylabels=["<X>", "<Y>", "<Z>"]
)
for ax in axes:
    ax.legend(['mesolove', 'brmesolve'], loc='upper right')
```

## 存储态而非期望值
和 QuTiP 其他求解器一样，我们也可在每个时间点获取密度矩阵，
而不是直接返回期望值。
做法是将 `e_ops` 传空列表。
若希望同时得到期望值（`e_ops` 非空）与状态，
可传入 `options={"store_states": True}`。

```python
# run solvers without e_ops
me_s = mesolve(H, psi0, times, c_ops, e_ops=[])
brme_s = brmesolve(H, psi0, times, [a_op], e_ops=[])

# calculate expecation values
x_me = expect(sigmax(), me_s.states)
x_brme = expect(sigmax(), brme_s.states)

# plot the expectation values
plt.plot(times, x_me, label="ME")
plt.plot(times, x_brme, label="BRME")
plt.legend(), plt.xlabel("time"), plt.ylabel("<X>");
```

可使用 `qutip.anim_hinton()` 可视化时间演化。
动画显示系统态收敛到基态。
即使改变初态，最终结果也相同。

```python
fig, ani = anim_hinton(me_s)
# close an auto-generated plot and animation
plt.close()
ani
```

## Bloch-Redfield 张量

系统动力学由 Bloch-Redfield 主方程描述，
其核心对象是 Bloch-Redfield 张量 $R_{abcd}$
（见[Bloch-Redfield 主方程文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html)）。
因此动力学由该张量决定。
在 QuTiP 中可通过 `qutip.bloch_redfield_tensor()` 计算该张量：
传入系统哈密顿量与 `a_ops` 即可构造 $R_{abcd}$。
该函数还会返回**哈密顿量本征态**（计算过程中会得到）。


```python
R, H_ekets = bloch_redfield_tensor(H, [a_op])

# calculate lindblad liouvillian from H
L = liouvillian(H, c_ops)
```

接下来使用 Bloch-Redfield 张量与 Lindblad Liouvillian 分别求稳态。
由于上面看到两种求解器动力学相近，稳态也应一致。
我们用 `qutip.hinton()` 绘制两种方法得到的稳态密度矩阵，可见一致。

注意：由 Bloch-Redfield 张量得到的稳态密度矩阵需用哈密顿量本征态进行基变换，
因为 `R` 在 `H` 的本征基中表示。

```python
# Obtain steadystate from Bloch-Redfield Tensor
rhoss_br_eigenbasis = steadystate(R)
rhoss_br = rhoss_br_eigenbasis.transform(H_ekets, True)

# Steadystate from Lindblad liouvillian
rhoss_me = steadystate(L)

# Plot the density matrices using a hinton plot
fig, ax = hinton(rhoss_br)
ax.set_title("Bloch-Redfield steadystate")
fig, ax = hinton(rhoss_me)
ax.set_title("Lindblad-ME steadystate");
```

## 环境信息

```python
about()
```

## 测试

```python
# Verify that mesolve and brmesolve generate similar results
assert np.allclose(result_me.expect[0], result_brme.expect[0])
assert np.allclose(result_me.expect[1], result_brme.expect[1])
assert np.allclose(result_me.expect[2], result_brme.expect[2])
assert np.allclose(x_me, x_brme)

# assume steadystate is the same
assert np.allclose(rhoss_br.full(), rhoss_me.full())
```
