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

# 讲座 0 - QuTiP 入门

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

本讲座系列由 J.R. Johansson 开发，原始讲义笔记可在[这里](https://github.com/jrjohansson/qutip-lectures)获取。

这是为适配当前 QuTiP 版本而做的轻度修改版讲义。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些内容。本讲及其他教程笔记也收录在 [QuTiP 教程页面](https://qutip.org/tutorials.html)。

```python
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor, thermal_dm, anim_matrix_histogram,
                   anim_fock_distribution)
# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

%matplotlib inline
```

## 简介

QuTiP 是一个用于量子系统计算与数值模拟的 Python 软件包。

它提供了对多种量子对象的表示与计算能力，包括态矢量（波函数）、bra/ket/密度矩阵、单体与复合系统的量子算符，以及用于构造主方程的超算符。

它还包含多种量子系统时间演化求解器，可处理薛定谔方程、冯诺依曼方程、主方程、Floquet 形式、量子蒙特卡洛轨迹，以及随机薛定谔/主方程的实验相关实现。

更多信息可见项目网站 [qutip.org](https://qutip.org) 与
[QuTiP 文档](https://qutip.readthedocs.io/en/latest/index.html)。

### 安装

你可以直接通过 `pip` 安装 QuTiP：

`pip install qutip`

更多安装细节请参考其 [GitHub 仓库](https://github.com/qutip/qutip)。


## 量子对象类：`qobj`

`Qobj` 类是 QuTiP 的核心，用于表示量子对象（如量子态与算符）。

`Qobj` 类包含描述量子系统所需的关键信息，例如矩阵表示、复合结构与维度信息。

```python
Image(filename="images/qobj.png")
```

### 创建与检查量子对象


可以通过 `Qobj` 构造函数创建新的量子对象，例如：

```python
q = Qobj([[1], [0]])

q
```

这里我们将 Python 列表作为构造参数传入。列表中的数据用于构建量子对象的矩阵表示，其它属性默认也会由这些数据自动推断。

我们可以通过以下类方法查看 `Qobj` 实例的属性：

```python
# the dimension, or composite Hilbert state space structure
q.dims
```

```python
# the shape of the matrix data representation
q.shape
```

```python
# the matrix data itself. in sparse matrix format.
q.data
```

```python
# get the dense matrix representation
q.full()
```

```python
# some additional properties
q.isherm, q.type
```

### 使用 `Qobj` 实例进行计算

基于 `Qobj` 实例，我们可以进行代数运算，并调用多种类方法完成常见操作：

```python
sy = Qobj([[0, -1j], [1j, 0]])  # the sigma-y Pauli operator

sy
```

```python
sz = Qobj([[1, 0], [0, -1]])  # the sigma-z Pauli operator

sz
```

```python
# some arithmetic with quantum objects

H = 1.0 * sz + 0.1 * sy

print("Qubit Hamiltonian = \n")
H
```

下面是使用 `Qobj` 方法修改量子对象的示例：

```python
# The hermitian conjugate
sy.dag()
```

```python
# The trace
H.tr()
```

```python
# Eigen energies
H.eigenenergies()
```

`Qobj` 类的完整方法和属性列表可见 [QuTiP 文档](https://qutip.readthedocs.io/en/latest/index.html)，也可直接在 Python 中使用 `help(Qobj)` 或 `dir(Qobj)`。


## 量子态与算符

通常我们不需要手工从矩阵表示出发调用构造函数来创建 `Qobj`。更常见的是直接使用 QuTiP 内置函数来生成常用量子态和算符。下面给出一些示例：

### 态矢量

```python
# Fundamental basis states (Fock states of oscillator modes)

N = 2  # number of states in the Hilbert space
n = 1  # the state that will be occupied

basis(N, n)  # equivalent to fock(N, n)
```

```python
fock(4, 2)  # another example
```

```python
# a coherent state
coherent(N=10, alpha=1.0)
```

### 密度矩阵

```python
# a fock state as density matrix
fock_dm(5, 2)  # 5 = hilbert space size, 2 = state that is occupied
```

```python
# coherent state as density matrix
coherent_dm(N=8, alpha=1.0)
```

```python
# thermal state
n = 1  # average number of thermal photons
thermal_dm(8, n)
```

### 算符


#### 量子比特（两能级系统）算符

```python
# Pauli sigma x
sigmax()
```

```python
# Pauli sigma y
sigmay()
```

```python
# Pauli sigma z
sigmaz()
```

#### 谐振子算符

```python
#  annihilation operator

destroy(N=8)  # N = number of fock states included in the Hilbert space
```

```python
# creation operator

create(N=8)  # equivalent to destroy(5).dag()
```

```python
# the position operator is easily constructed from the annihilation operator
a = destroy(8)

x = a + a.dag()

x
```

#### 用 `Qobj` 实例可以检验一些经典对易关系：

```python
def commutator(op1, op2):
    return op1 * op2 - op2 * op1
```

$[a, a^1] = 1$

```python
a = destroy(5)

commutator(a, a.dag())
```

**注意：** 结果并不是单位算符！原因在于我们对希尔伯特空间做了截断。只要动力学中不涉及截断后的最高 Fock 态，这通常没有问题；否则，这种截断近似会带来误差。


$[x,p] = i$

```python
x = (a + a.dag()) / np.sqrt(2)
p = -1j * (a - a.dag()) / np.sqrt(2)
```

```python
commutator(x, p)
```

这里同样受希尔伯特空间截断影响，但除此之外结果是正常的。


再来看一些 Pauli 自旋关系

$[\sigma_x, \sigma_y] = 2i \sigma_z$

```python
commutator(sigmax(), sigmay()) - 2j * sigmaz()
```

$-i \sigma_x \sigma_y \sigma_z = \mathbf{1}$

```python
-1j * sigmax() * sigmay() * sigmaz()
```

$\sigma_x^2 = \sigma_y^2 = \sigma_z^2 = \mathbf{1}$

```python
sigmax() ** 2 == sigmay() ** 2 == sigmaz() ** 2 == qeye(2)
```

## 复合系统

在多数情形下，我们关心的是耦合量子系统，例如耦合量子比特、量子比特与腔（振子模）耦合等。

在 QuTiP 中，为此类系统定义态和算符时，通常通过 `tensor` 函数构造复合系统的 `Qobj`。

例如考虑双量子比特系统。若要构造只作用于第一个量子比特、且对第二个量子比特不作用的 Pauli $\sigma_z$（即 $\sigma_z \otimes \mathbf{1}$），可写为：

```python
sz1 = tensor(sigmaz(), qeye(2))

sz1
```

我们可以很容易验证该双量子比特算符确实满足预期性质：

```python
psi1 = tensor(basis(N, 1), basis(N, 0))  # excited first qubit
psi2 = tensor(basis(N, 0), basis(N, 1))  # excited second qubit
```

```python
# this should not be true,
# because sz1 should flip the sign of the excited state of psi1
sz1 * psi1 == psi1
```

```python
# this should be true, because sz1 should leave psi2 unaffected
sz1 * psi2 == psi2
```

上面使用了 `qeye(N)` 生成包含 `N` 个量子态的单位算符。若要在第二个量子比特上构造同类算符，可以这样做：

```python
sz2 = tensor(qeye(2), sigmaz())

sz2
```

注意 `tensor` 参数顺序，这会导致 `sz1` 与 `sz2` 的矩阵表示不同。

用同样的方法，我们可以构造形如 $\sigma_x \otimes \sigma_x$ 的耦合项：

```python
tensor(sigmax(), sigmax())
```

现在可以构造耦合双量子比特哈密顿量的 `Qobj` 表示：$H = \epsilon_1 \sigma_z^{(1)} + \epsilon_2 \sigma_z^{(2)} + g \sigma_x^{(1)}\sigma_x^{(2)}$

```python
epsilon = [1.0, 1.0]
g = 0.1

sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

H = epsilon[0] * sz1 + epsilon[1] * sz2 + g * tensor(sigmax(), sigmax())

H
```

若要构造不同类型的复合系统，只需调整传给 `tensor` 的算符集合（它可接受任意数量的算符，以支持多组分系统）。

例如，量子比特-腔系统的 Jaynes-Cumming 哈密顿量：

$H = \omega_c a^\dagger a - \frac{1}{2}\omega_a \sigma_z + g (a \sigma_+ + a^\dagger \sigma_-)$

```python
wc = 1.0  # cavity frequency
wa = 1.0  # qubit/atom frenqency
g = 0.1  # coupling strength

# cavity mode operator
a = tensor(destroy(5), qeye(2))

# qubit/atom operators
sz = tensor(qeye(5), sigmaz())  # sigma-z operator
sm = tensor(qeye(5), destroy(2))  # sigma-minus operator

# the Jaynes-Cumming Hamiltonian
H = wc * a.dag() * a - 0.5 * wa * sz + g * (a * sm.dag() + a.dag() * sm)

H
```

注意

$a \sigma_+ = (a \otimes \mathbf{1}) (\mathbf{1} \otimes \sigma_+)$

因此下面两种写法是等价的：

```python
a = tensor(destroy(3), qeye(2))
sp = tensor(qeye(3), create(2))

a * sp
```

```python
tensor(destroy(3), create(2))
```

## 幺正动力学

在 QuTiP 中，量子系统的幺正演化可通过 `mesolve` 计算。

`mesolve` 是 Master-equation solver 的缩写（原本用于耗散动力学）。但若未提供描述耗散的塌缩算符，它会退化为幺正演化求解：初态为态矢量时对应薛定谔方程，初态为密度矩阵时对应冯诺依曼方程。

QuTiP 的演化求解器会返回 `Odedata` 类型结果，其中包含对应演化问题的解。

例如，考虑哈密顿量 $H = \sigma_x$、初态为 $\left|1\right>$（在 $\sigma_z$ 基底下）的量子比特，其演化可按如下方式计算：

```python
# Hamiltonian
H = sigmax()

# initial state
psi0 = basis(2, 0)

# list of times for which the solver should store the state vector
tlist = np.linspace(0, 10, 100)

result = mesolve(H, psi0, tlist, [])
```

```python
result
```

`result` 对象中包含按 `tlist` 给定时刻采样得到的波函数列表。

```python
len(result.states)
```

```python
result.states[-1]  # the finial state
```

你可以将态的时间演化可视化。`anim_matrix_histogram` 用于展示其矩阵元素。下方动画显示状态周期变化，并且 $\left|1\right>$ 的系数为实部主导、$\left|0\right>$ 的系数为虚部主导。

```python
fig, ani = anim_matrix_histogram(result, limits=[0, 1],
                                 bar_style='abs', color_style='phase')
# close an auto-generated plot and animation
plt.close()
ani
```

### 期望值

给定态矢量或密度矩阵（或它们的列表）时，可用 `expect` 函数计算算符期望值。

```python
expect(sigmaz(), result.states[-1])
```

```python
expect(sigmaz(), result.states)
```

```python
fig, axes = plt.subplots(1, 1)

axes.plot(tlist, expect(sigmaz(), result.states))

axes.set_xlabel(r"$t$", fontsize=20)
axes.set_ylabel(r"$\left<\sigma_z\right>$", fontsize=20);
```

如果我们只关心期望值，可以将待观测算符列表传入 `mesolve`，求解器会自动计算并将结果保存在返回的 `Odedata` 对象中。\

例如，要求求解器计算 $\sigma_x, \sigma_y, \sigma_z$ 的期望值：

```python
result = mesolve(H, psi0, tlist, [], e_ops=[sigmax(), sigmay(), sigmaz()])
```

现在，这些期望值可在 `result.expect[0]`、`result.expect[1]` 和 `result.expect[2]` 中获取：

```python
fig, axes = plt.subplots(1, 1)

axes.plot(tlist, result.expect[2], label=r"$\left<\sigma_z\right>$")
axes.plot(tlist, result.expect[1], label=r"$\left<\sigma_y\right>$")
axes.plot(tlist, result.expect[0], label=r"$\left<\sigma_x\right>$")

axes.set_xlabel(r"$t$", fontsize=20)
axes.legend(loc=2);
```

## 耗散动力学

要在模型中引入耗散，只需在调用 `mesolve` 时提供塌缩算符列表。

塌缩算符描述了系统与环境的相互作用方式。

例如，考虑一个量子谐振子，其哈密顿量为

$H = \hbar\omega a^\dagger a$

并以弛豫速率 $\kappa$ 向环境失去光子。描述该过程的塌缩算符是

$\sqrt{\kappa} a$

因为 $a$ 是该振子的光子湮灭算符。

在 QuTiP 中，这个问题可以这样实现：

```python
w = 1.0  # oscillator frequency
kappa = 0.1  # relaxation rate
a = destroy(10)  # oscillator annihilation operator
rho0 = fock_dm(10, 5)  # initial state, fock state with 5 photons
H = w * a.dag() * a  # Hamiltonian

# A list of collapse operators
c_ops = [np.sqrt(kappa) * a]
```

```python
tlist = np.linspace(0, 50, 100)

# request that the solver return the expectation value
# of the photon number state operator a.dag() * a
result = mesolve(H, rho0, tlist, c_ops, [a.dag() * a],
                 options={"store_states": True})
```

```python
fig, axes = plt.subplots(1, 1)
axes.plot(tlist, result.expect[0])
axes.set_xlabel(r"$t$", fontsize=20)
axes.set_ylabel(r"Photon number", fontsize=16);
```

`anim_fock_distribution` 可用于可视化各时刻的概率分布。

```python
fig, ani = anim_fock_distribution(result)
# close an auto-generated plot and animation
plt.close()
ani
```

### 安装信息

```python
about()
```
