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

# 主方程求解器：自旋链动力学


Authors: J.R. Johansson and P.D. Nation

Modifications: C. Staufenbiel (2022)

### 简介

本教程将模拟一条自旋链（也称海森堡模型）：
系统由处在磁场中的 $N$ 个 $\frac{1}{2}$ 自旋/量子比特组成，
每个自旋可与其最近邻相互作用。
该模型常用于研究磁性系统。

这里考虑的一维海森堡模型可通过 Bethe Ansatz 精确求解
（即可精确计算其哈密顿量谱）。
### 导入

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, expect, mesolve, qeye, sigmax, sigmay, sigmaz,
                   tensor)

%matplotlib inline
```

### 系统设置
自旋链哈密顿量由自旋间相互作用项和磁场项组成。
我们假设磁场沿自旋的 $z$ 轴，其在第 $n$ 个自旋处的强度为 $h_n$，
因此磁场可随自旋位置变化。

自旋在三个方向均可相互作用。
因此定义三个系数 $J_x^{(n)}, J_y^{(n)}, J_z^{(n)}$，
其中第 $n$ 个系数描述自旋 $n$ 与自旋 $(n+1)$ 的相互作用。
海森堡模型哈密顿量可写为：

$\displaystyle H = - \frac{1}{2}\sum_n^N h_n \sigma_z(n) - \frac{1}{2} \sum_n^{N-1} [ J_x^{(n)} \sigma_x(n) \sigma_x(n+1) + J_y^{(n)} \sigma_y(n) \sigma_y(n+1) +J_z^{(n)} \sigma_z(n) \sigma_z(n+1)]$

下面定义系统大小、初态和相互作用系数。
初态取第一自旋为 *up*，其余自旋为 *down*。
另外取均匀磁场，并设 $J_x = J_y = J_z$。
你可自行修改这些设置来模拟不同自旋链。

```python
# Set the system parameters
N = 5

# initial state
state_list = [basis(2, 1)] + [basis(2, 0)] * (N - 1)
psi0 = tensor(state_list)

# Energy splitting term
h = 2 * np.pi * np.ones(N)

# Interaction coefficients
Jx = 0.2 * np.pi * np.ones(N)
Jy = 0.2 * np.pi * np.ones(N)
Jz = 0.2 * np.pi * np.ones(N)
```

对每个量子比特，我们构造算符 $\sigma_i$：
它是“各位为恒等算符，仅该量子比特位置为 $\sigma_i$”的张量积。
随后可用这些张量积算符与上面系数构造哈密顿量。

```python
# Setup operators for individual qubits
sx_list, sy_list, sz_list = [], [], []
for i in range(N):
    op_list = [qeye(2)] * N
    op_list[i] = sigmax()
    sx_list.append(tensor(op_list))
    op_list[i] = sigmay()
    sy_list.append(tensor(op_list))
    op_list[i] = sigmaz()
    sz_list.append(tensor(op_list))

# Hamiltonian - Energy splitting terms
H = 0
for i in range(N):
    H -= 0.5 * h[i] * sz_list[i]

# Interaction terms
for n in range(N - 1):
    H += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
    H += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
    H += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]
```

### 时间演化
可使用 `qutip.mesolve` 进行仿真。
这里不传入塌缩算符与期望算符，因此函数返回各时间步的量子态。
为便于后续处理，我们将这些态转成密度矩阵。

```python
times = np.linspace(0, 100, 200)
result = mesolve(H, psi0, times, [], [])
# Convert states to density matrices
states = [s * s.dag() for s in result.states]
```

为可视化自旋链动力学，我们计算每个量子比特上 $\sigma_z$ 的期望值。
为了更清晰，仅绘制第一个与最后一个量子比特。
可见第一个自旋初始期望值为 $-1$，最后一个为 $+1$，对应我们设置的初态。
随时间演化可观察到自旋动量在各自旋间持续传递。

```python
# Expectation value
exp_sz = np.array(expect(states, sz_list))

# Plot the expecation value
plt.plot(times, exp_sz[:, 0], label=r"$\langle \sigma_z^{0} \rangle$")
plt.plot(times, exp_sz[:, -1], label=r"$\langle \sigma_z^{-1} \rangle$")
plt.legend(loc="lower right")
plt.xlabel("Time"), plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.title("Dynamics of spin chain");
```

### 退相干

主方程求解器 `qutip.mesolve` 允许定义塌缩算符来描述耗散过程。
这里我们通过如下塌缩算符加入所有自旋的退相位：

$C = \sum_{i=1}^N \; \sqrt{\gamma_i} \, \sigma_z(i)$

对该含耗散系统进行仿真。
从图中可见 $\sigma_z$ 期望值会弛豫到常值，这是系统相干性损失导致的。

```python
# dephasing rate
gamma = 0.02 * np.ones(N)

# collapse operators
c_ops = [np.sqrt(gamma[i]) * sz_list[i] for i in range(N)]

# evolution
result = result = mesolve(H, psi0, times, c_ops, [])

# Expectation value
exp_sz_dephase = expect(sz_list, result.states)

# Plot the expecation value
plt.plot(times, exp_sz_dephase[0], label=r"$\langle \sigma_z^{0} \rangle$")
plt.plot(times, exp_sz_dephase[-1], label=r"$\langle \sigma_z^{-1} \rangle$")
plt.legend()
plt.xlabel("Time"), plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.title("Dynamics of spin chain with qubit dephasing");
```

### 环境信息

```python
about()
```

### 测试

```python
assert np.allclose(np.array(exp_sz_dephase)[:, -1], 0.6, atol=0.01)
```
