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

# 蒙特卡洛求解器：腔中光子的产生与湮灭

<!-- #region -->
Authors: J.R. Johansson and P.D. Nation

Modifications: C. Staufenbiel (2022)

### 简介

本教程演示 `qutip.mcsolve()` 中实现的*蒙特卡洛求解器*功能。
关于 *MC Solver* 的更多信息可参考
[QuTiP 文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-monte.html)。

我们的目标是复现实验结果：



>  Gleyzes et al., "Quantum jumps of light recording the birth and death of a photon in a cavity", [Nature **446**,297 (2007)](http://dx.doi.org/10.1038/nature05589).


特别地，我们将模拟初态为单光子 Fock 态 $ |1\rangle$ 时，
热环境导致的光学腔内光子产生与湮灭过程，对应论文图 3 的内容。

## 导入
先导入相关功能：
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import about, basis, destroy, mcsolve, mesolve

%matplotlib inline
```

## 系统设置
在这个例子中，我们考虑简谐振子哈密顿量 $H = a^\dagger a$，并设定腔内初态为单光子。

```python
N = 5  # number of modes in the
a = destroy(N)  # Destroy operator
H = a.dag() * a  # oscillator Hamiltonian
psi0 = basis(N, 1)  # Initial Fock state with one photon
```

与外部热浴的耦合由耦合常数 $\kappa$ 描述，
热浴温度通过平均光子数 $\langle n \rangle$ 表征。
在 QuTiP 中，系统与热浴相互作用通过塌缩算符定义。
该例包含两个塌缩算符：
一个用于光子湮灭（$C_1$），一个用于光子产生（$C_2$）：

$C_1 = \sqrt{\kappa (1 + \langle n \rangle)} \; a$

$C_2 = \sqrt{\kappa \langle n \rangle} \; a^\dagger$

下面给出 $\kappa$ 与 $\langle n \rangle$ 的数值。

```python
kappa = 1.0 / 0.129  # Coupling rate to heat bath
nth = 0.063  # Temperature with <n>=0.063

# collapse operators for the thermal bath
c_ops = []
c_ops.append(np.sqrt(kappa * (1 + nth)) * a)
c_ops.append(np.sqrt(kappa * nth) * a.dag())
```

## 蒙特卡洛仿真
*蒙特卡洛求解器*可用于模拟系统动力学的单次实现。
这与*主方程求解器*不同，后者对应大量相同系统实现的系综平均。
`qutip.mcsolve()` 也支持通过传入轨迹数 `ntraj` 对多次独立实现求平均。
若 `ntraj = 1`，只模拟一次，可看到单次系统动力学；
若 `ntraj` 较大，结果会趋近 `qutip.mesolve()` 的平均解。

`ntraj` 还可传入列表，`qutip.mcsolve()` 会计算这些指定轨迹数的结果。
注意列表项需按升序排列，因为前一次结果会被复用。

这里我们关注不同 `ntraj` 下 $a^\dagger a$ 的时间演化，
并与 `qutip.mesolve()` 的结果比较。

```python
ntraj = [1, 5, 15, 904]  # number of MC trajectories
mc = []  # MC results
tlist = np.linspace(0, 0.8, 100)

# Solve using MCSolve for different ntraj
for n in ntraj:
    result = mcsolve(H, psi0, tlist, c_ops, [a.dag() * a], ntraj=n)
    mc.append(result)
me = mesolve(H, psi0, tlist, c_ops, [a.dag() * a])
```

## 复现论文中的图
利用上述结果，我们可以复现前述论文的图 3。
各子图展示了系统中 $\langle a^\dagger a \rangle$ 的时间演化。
并展示了 `mcsolve` 在不同 `ntraj` 设置下的效果。
当 `ntraj = 1` 时，得到单次量子系统轨迹；
当 `ntraj > 1` 时，输出为多次实现后的平均。

```python
fig = plt.figure(figsize=(8, 8), frameon=False)
plt.subplots_adjust(hspace=0.0)

for i in range(len(ntraj)):
    ax = plt.subplot(4, 1, i + 1)
    ax.plot(
        tlist, mc[i].expect[0], "b", lw=2,
        label="#trajectories={}".format(ntraj[i])
    )
    ax.plot(tlist, me.expect[0], "r--", lw=2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel(r"$\langle P_{1}(t)\rangle$")
    ax.legend()

ax.set_xlabel(r"Time (s)");
```

## 环境信息

```python
about()
```

## 测试

```python
np.testing.assert_allclose(me.expect[0], mc[3].expect[0], atol=10**-1)
assert np.all(np.diff(me.expect[0]) <= 0)
```
