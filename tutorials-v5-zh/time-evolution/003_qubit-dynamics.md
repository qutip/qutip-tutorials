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

# 主方程求解器：单量子比特动力学

Authors: J.R. Johansson and P.D. Nation

Modified by: C. Staufebiel (2022)

### 简介
本 notebook 将探索与环境相互作用的单量子比特动力学。
量子比特状态演化由主方程控制。
我们使用 qutip 中实现的主方程求解器 `qutip.mesolve`，
在不同设置下求取量子比特时间演化。

关于主方程求解器（及其理论背景），可参考
[QuTiP 文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-master.html)。

### 导入
这里导入示例所需模块。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import Bloch, about, basis, mesolve, sigmam, sigmax, sigmay, sigmaz

%matplotlib inline
```

### 系统设置
先考虑一个基础量子比特哈密顿量，
它通过泡利矩阵 $\sigma_x$ 翻转量子比特态：

$H = \frac{\Delta}{2} \sigma_x$

此外加入一个塌缩算符，用于描述量子比特向外部环境的能量耗散：

$C = \sqrt{g} \sigma_z$

其中 $g$ 为耗散系数。
设定量子比特在 $t=0$ 处于基态。

```python
# coefficients
delta = 2 * np.pi
g = 0.25

# hamiltonian
H = delta / 2.0 * sigmax()

# list of collapse operators
c_ops = [np.sqrt(g) * sigmaz()]

# initial state
psi0 = basis(2, 0)

# times
tlist = np.linspace(0, 5, 100)
```

### 时间演化
将上述定义传入 `qutip.mesolve`。
塌缩算符需以列表形式给出（即使只有一个）。
第五个参数传入待观测算符列表，
求解器会在 `tlist` 各时刻返回对应期望值。
本例中我们求 $\sigma_z$ 的期望值。

```python
res = mesolve(H, psi0, tlist, c_ops, [sigmaz()])
```

对于该哈密顿量与耗散过程，
可推导出 $\sigma_z$ 期望值解析解：

```python
sz_analytic = np.cos(2 * np.pi * tlist) * np.exp(-tlist * g)
```

### 与解析解比较
通过绘制 `mesolve` 结果与解析解，可验证结果正确性。
`mesolve` 返回期望值可通过 `result.expect[0]` 访问。

```python
plt.scatter(tlist, res.expect[0], c="r", marker="x", label="mesolve")
plt.plot(tlist, sz_analytic, label="Analytic")
plt.xlabel("Time"), plt.ylabel("<sigma_z>")
plt.legend();
```

## Bloch 球上的量子比特动力学

我们还可在 Bloch 球上可视化量子比特态演化。
为得到更丰富图像，下面使用稍复杂动力学。

### 旋转态

考虑如下哈密顿量：

$H = \Delta ( \, cos(\theta) \, \sigma_z + sin(\theta) \, \sigma_x  \, )$.

$\theta$ 定义量子比特态相对 $z$ 轴朝 $x$ 轴的夹角。
同样用 `mesolve` 求解，这里塌缩算符列表为空。

```python
# Angle
theta = 0.2 * np.pi

# Hamiltonian
H = delta * (np.cos(theta) * sigmaz() + np.sin(theta) * sigmax())

# Obtain Time Evolution
tlist = np.linspace(0, 5, 1000)
result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])
```

可使用 `qutip.Bloch` 类在 Bloch 球上展示状态：
既可添加轨迹点，也可添加状态向量。

```python
# Extract expectation values for pauli matrices
exp_sx_circ, exp_sy_circ, exp_sz_circ = result.expect
exp_sx_circ, exp_sy_circ, exp_sz_circ = (
    np.array(exp_sx_circ),
    np.array(exp_sy_circ),
    np.array(exp_sz_circ),
)

# Create Bloch sphere plot
sphere = Bloch()
sphere.add_points([exp_sx_circ, exp_sy_circ, exp_sz_circ], meth="l")
sphere.add_states(psi0)
sphere.show()
```

从图中可见初态在球面上做圆形轨迹演化。
下面继续加入塌缩算符，观察耗散对动力学的影响。

### 量子比特退相位

通过以下塌缩算符引入相位退相干：

$C = \sqrt{\gamma_p} \; \sigma_z$

```python
gamma_phase = 0.5
c_ops = [np.sqrt(gamma_phase) * sigmaz()]

# solve dynamics
result = mesolve(H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()])
exp_sx_dephase, exp_sy_dephase, exp_sz_dephase = result.expect
exp_sx_dephase, exp_sy_dephase, exp_sz_dephase = (
    np.array(exp_sx_dephase),
    np.array(exp_sy_dephase),
    np.array(exp_sz_dephase),
)

# Create Bloch sphere plot
sphere = Bloch()
sphere.add_points([exp_sx_dephase, exp_sy_dephase, exp_sz_dephase], meth="l")
sphere.add_states(psi0)
sphere.show()
```

可通过轨迹半径逐渐减小观察到退相位效应。

### 量子比特弛豫

另一类可研究的耗散是弛豫，对应塌缩算符：

$C = \sqrt{\gamma_r} \sigma_-$

该过程会以速率 $\gamma_r$ 诱导量子比特从激发态自发翻转到基态。
同样可在 Bloch 球上观察其动力学。

```python
gamma_relax = 0.5
c_ops = [np.sqrt(gamma_relax) * sigmam()]

# solve dynamics
result = mesolve(H, psi0, tlist, c_ops, [sigmax(), sigmay(), sigmaz()])
exp_sx_relax, exp_sy_relax, exp_sz_relax = result.expect

# Create Bloch sphere plot
sphere = Bloch()
sphere.add_points([exp_sx_relax, exp_sy_relax, exp_sz_relax], meth="l")
sphere.add_states(psi0)
sphere.show()
```

可以看到圆轨迹逐渐偏向量子比特基态。

### 总结
使用以上方法，你可以模拟任意由主方程描述的耗散量子系统。
同时可借助 Bloch 球可视化量子比特态动力学。

## 环境信息

```python
about()
```

### 测试

```python
assert np.allclose(res.expect[0], sz_analytic, atol=0.05)
assert np.allclose(exp_sz_circ**2 + exp_sy_circ**2 + exp_sx_circ**2, 1.0)
assert np.all(
    np.diff(exp_sx_dephase**2 + exp_sy_dephase**2 + exp_sz_dephase**2) <= 0
)
```
