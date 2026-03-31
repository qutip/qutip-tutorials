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

# 讲座 7 - 双量子比特 iSWAP 门与过程层析

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

本系列讲座由 J.R. Johansson 开发，原始讲义 notebook 可在[这里](https://github.com/jrjohansson/qutip-lectures)查看。

这里是为适配当前 QuTiP 版本而稍作修改的版本。
你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲座。
本讲座及其他教程 notebook 的索引页见 [QuTiP Tutorial 网页](https://qutip.org/tutorials.html)。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, propagator, qeye, qpt, qpt_plot_combined,
                   sigmam, sigmax, sigmay, sigmaz, spost, spre, tensor)

%matplotlib inline
```

### 简介

考虑一种实现双量子比特 iSWAP 门的简单方案：在时间 $T=\pi/4g$ 内，
系统受如下耦合哈密顿量作用

$\displaystyle H = g \left(\sigma_x\otimes\sigma_x + \sigma_y\otimes\sigma_y\right)$

其中 $g$ 为耦合强度。在理想条件下，该耦合可在两个量子比特间实现 $i$-SWAP 门。

下面我们将求解该哈密顿量下双量子比特的动力学，
并观察加入退相干后的退化效应。
同时使用过程层析来可视化该量子门。


### 参数

```python
g = 1.0 * 2 * np.pi  # coupling strength
g1 = 0.75  # relaxation rate
g2 = 0.25  # dephasing rate
n_th = 1.5  # bath temperature

T = np.pi / (4 * g)
```

### 哈密顿量与初态

```python
H = g * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))
psi0 = tensor(basis(2, 1), basis(2, 0))
```

### 塌缩算符

定义塌缩算符列表，用以描述量子比特与环境（假设互不相关）之间的耦合。

```python
c_ops = []

# qubit 1 collapse operators
sm1 = tensor(sigmam(), qeye(2))
sz1 = tensor(sigmaz(), qeye(2))
c_ops.append(np.sqrt(g1 * (1 + n_th)) * sm1)
c_ops.append(np.sqrt(g1 * n_th) * sm1.dag())
c_ops.append(np.sqrt(g2) * sz1)

# qubit 2 collapse operators
sm2 = tensor(qeye(2), sigmam())
sz2 = tensor(qeye(2), sigmaz())
c_ops.append(np.sqrt(g1 * (1 + n_th)) * sm2)
c_ops.append(np.sqrt(g1 * n_th) * sm2.dag())
c_ops.append(np.sqrt(g2) * sz2)
```

### 过程层析基底

```python
op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * 2
op_label = [["i", "x", "y", "z"]] * 2
```

### 理想演化（无耗散、完美定时等）

```python
# calculate the propagator for the ideal gate
U_psi = (-1j * H * T).expm()
```

```python
# propagator in superoperator form
U_ideal = spre(U_psi) * spost(U_psi.dag())
```

```python
# calculate the process tomography chi matrix from the superoperator propagator
chi = qpt(U_ideal, op_basis)
```

```python
fig = plt.figure(figsize=(8, 6))
fig = qpt_plot_combined(chi, op_label, fig=fig)
```

### 耗散演化

```python
# dissipative gate propagator
U_diss = propagator(H, T, c_ops)
```

```python
# calculate the process tomography chi matrix for the dissipative propagator
chi = qpt(U_diss, op_basis)
```

```python
fig = plt.figure(figsize=(8, 6))
fig = qpt_plot_combined(chi, op_label, fig=fig)
```

### 软件版本：

```python
about()
```
