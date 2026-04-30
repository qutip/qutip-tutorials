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

# 第 12 讲 - 衰减到压缩真空场


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (Bloch, about, basis, liouvillian, mesolve, n_thermal,
                   sigmam, sigmap, sigmax, sigmay, sigmaz, spost, spre)

%matplotlib inline
```

## 引言

这里遵循 Breuer 与 Petruccione《The theory of open quantum systems》第 3.4.3 - 3.4.4 节，讨论二能级系统向压缩真空环境衰减时的主方程：

$\frac{d}{dt}\rho = \gamma_0(N+1)\left(\sigma_-\rho(t)\sigma_+ - \frac{1}{2}\sigma_+\sigma_-\rho(t) - \frac{1}{2}\rho(t)\sigma_+\sigma_-\right)$

$ + \gamma_0 N \left(\sigma_+\rho(t)\sigma_- - \frac{1}{2}\sigma_-\sigma_+\rho(t) - \frac{1}{2}\rho(t)\sigma_-\sigma_+\right)$

$ -\gamma_0 M \sigma_+\rho(t)\sigma_+ -\gamma_0 M^* \sigma_-\rho(t)\sigma_-$

其中参数 $N$ 与 $M$ 描述环境模的温度与压缩：

$\displaystyle N = N_{\rm th} ({\cosh}^2 r + {\sinh}^2 r) + \sinh^2 r$

$\displaystyle M = - \cosh r \sinh r e^{i\theta} (2 N_{\rm th} + 1)$

该方程也可改写成标准 Lindblad 形式：

$\frac{d}{dt}\rho = \gamma_0\left(C\rho(t)C^\dagger - \frac{1}{2}C^\dagger C\rho(t) - \frac{1}{2}\rho(t)C^\dagger C\right)$

其中 $C = \sigma_-\cosh r + \sigma_+ \sinh r e^{i\theta}$。

下面我们用 QuTiP 数值求解这两种主方程，并可视化其动力学。



### 问题参数

```python
w0 = 1.0 * 2 * np.pi
gamma0 = 0.05
```

```python
# the temperature of the environment in frequency units
w_th = 0.0 * 2 * np.pi
```

```python
# the number of average excitations in the
# environment mode w0 at temperature w_th
Nth = n_thermal(w0, w_th)

Nth
```

#### 描述热浴压缩的参数

```python
# squeezing parameter for the environment
r = 1.0
theta = 0.1 * np.pi
```

```python
N = Nth * (np.cosh(r) ** 2 + np.sinh(r) ** 2) + np.sinh(r) ** 2

N
```

```python
M = -np.cosh(r) * np.sinh(r) * np.exp(-1j * theta) * (2 * Nth + 1)

M
```

```python
# Check, should be zero according to Eq. 3.261 in Breuer and Petruccione
abs(M) ** 2 - (N * (N + 1) - Nth * (Nth + 1))
```

### 算符、哈密顿量与初态

```python
sm = sigmam()
sp = sigmap()
```

```python
H = (
    -0.5 * w0 * sigmaz()
)  # by adding the hamiltonian here, so we move back to the schrodinger picture
```

```python
c_ops = [np.sqrt(gamma0 * (N + 1)) * sm, np.sqrt(gamma0 * N) * sp]
```

先构造 Liouvillian 的标准部分，对应幺正演化加上前述第一个主方程中的前两项：

```python
L0 = liouvillian(H, c_ops)

L0
```

接着手工构造环境压缩引起的 Liouvillian 项。该项不是标准 Lindblad 形式，因此不能直接用 QuTiP 的 `liouvillian` 函数。

```python
Lsq = -gamma0 * M * spre(sp) * spost(sp) - gamma0 * \
      M.conj() * spre(sm) * spost(sm)

Lsq
```

于是总 Liouvillian 为

```python
L = L0 + Lsq

L
```

### 演化

现在可用 QuTiP 的 `mesolve` 数值求解主方程：

```python
tlist = np.linspace(0, 50, 1000)
```

```python
# start in the qubit superposition state
psi0 = (2j * basis(2, 0) + 1 * basis(2, 1)).unit()
```

```python
e_ops = [sigmax(), sigmay(), sigmaz()]
```

```python
result1 = mesolve(L, psi0, tlist, [], e_ops=e_ops)
```

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(result1.times, result1.expect[0], "r",
        label=r"$\langle\sigma_x\rangle$")
ax.plot(result1.times, result1.expect[1], "g",
        label=r"$\langle\sigma_y\rangle$")
ax.plot(result1.times, result1.expect[2], "b",
        label=r"$\langle\sigma_z\rangle$")

sz_ss_analytical = -1 / (2 * N + 1)
ax.plot(
    result1.times,
    sz_ss_analytical * np.ones(len(result1.times)),
    "k--",
    label=r"$\langle\sigma_z\rangle_s$ analytical",
)


ax.set_ylabel(r"$\langle\sigma_z\rangle$", fontsize=16)
ax.set_xlabel("time", fontsize=16)
ax.legend()
ax.set_ylim(-1, 1);
```

```python
b = Bloch()
b.add_points(result1.expect, meth="l")
b.show()
```

### Lindblad 形式的等价主方程

另一种主方程是标准 Lindblad 形式，可直接用 `mesolve` 求解：

```python
c_ops = [np.sqrt(gamma0) *
         (sm * np.cosh(r) + sp * np.sinh(r) * np.exp(1j * theta))]
```

```python
result2 = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
```

并可验证其结果确实一致：

```python
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(result2.times, result2.expect[0], "r",
        label=r"$\langle\sigma_x\rangle$")
ax.plot(result2.times, result2.expect[1], "g",
        label=r"$\langle\sigma_y\rangle$")
ax.plot(result2.times, result2.expect[2], "b",
        label=r"$\langle\sigma_z\rangle$")

sz_ss_analytical = -1 / (2 * N + 1)
ax.plot(
    result2.times,
    sz_ss_analytical * np.ones(len(result2.times)),
    "k--",
    label=r"$\langle\sigma_z\rangle_s$ analytical",
)


ax.set_ylabel(r"$\langle\sigma_z\rangle$", fontsize=16)
ax.set_xlabel("time", fontsize=16)
ax.legend()
ax.set_ylim(-1, 1);
```

### 比较两种主方程形式

```python
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))

axes[0].plot(
    result1.times, result1.expect[0], "r",
    label=r"$\langle\sigma_x\rangle$ - me"
)
axes[0].plot(
    result2.times,
    result2.expect[0],
    "b--",
    label=r"$\langle\sigma_x\rangle$ - me lindblad",
)
axes[0].legend()
axes[0].set_ylim(-1, 1)

axes[1].plot(
    result1.times, result1.expect[1], "r",
    label=r"$\langle\sigma_y\rangle$ - me"
)
axes[1].plot(
    result2.times,
    result2.expect[1],
    "b--",
    label=r"$\langle\sigma_y\rangle$ - me lindblad",
)
axes[1].legend()
axes[1].set_ylim(-1, 1)

axes[2].plot(
    result1.times, result1.expect[2], "r",
    label=r"$\langle\sigma_y\rangle$ - me"
)
axes[2].plot(
    result2.times,
    result2.expect[2],
    "b--",
    label=r"$\langle\sigma_y\rangle$ - me lindblad",
)
axes[2].legend()
axes[2].set_ylim(-1, 1)
axes[2].set_xlabel("time", fontsize=16);
```

### 比较衰减到真空与压缩真空

```python
# for vacuum:
r = 0
theta = 0.0
c_ops = [np.sqrt(gamma0) *
         (sm * np.cosh(r) + sp * np.sinh(r) * np.exp(1j * theta))]
```

```python
result1 = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
```

```python
# for squeezed vacuum:
r = 1.0
theta = 0.0
c_ops = [np.sqrt(gamma0) *
         (sm * np.cosh(r) + sp * np.sinh(r) * np.exp(1j * theta))]
```

```python
result2 = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops)
```

```python
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 9))

axes[0].plot(
    result1.times, result1.expect[0], "r",
    label=r"$\langle\sigma_x\rangle$ - vacuum"
)
axes[0].plot(
    result2.times,
    result2.expect[0],
    "b",
    label=r"$\langle\sigma_x\rangle$ - squeezed vacuum",
)
axes[0].legend()
axes[0].set_ylim(-1, 1)

axes[1].plot(
    result1.times, result1.expect[1], "r",
    label=r"$\langle\sigma_y\rangle$ - vacuum"
)
axes[1].plot(
    result2.times,
    result2.expect[1],
    "b",
    label=r"$\langle\sigma_y\rangle$ - squeezed vacuum",
)
axes[1].legend()
axes[1].set_ylim(-1, 1)

axes[2].plot(
    result1.times, result1.expect[2], "r",
    label=r"$\langle\sigma_z\rangle$ - vacuum"
)
axes[2].plot(
    result2.times,
    result2.expect[2],
    "b",
    label=r"$\langle\sigma_z\rangle$ - squeezed vacuum",
)
axes[2].legend()
axes[2].set_ylim(-1, 1)
axes[2].set_xlabel("time", fontsize=16);
```

由此可见，衰减到压缩真空比衰减到普通真空更快。


### 软件版本

```python
about()
```
