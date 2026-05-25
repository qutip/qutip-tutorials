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

# 讲座 8 - 绝热扫描

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

本系列讲座由 J.R. Johansson 开发，原始讲义 notebook 可在[这里](https://github.com/jrjohansson/qutip-lectures)查看。

这里是为适配当前 QuTiP 版本而稍作修改的版本。
你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲座。
本讲座及其他教程 notebook 的索引页见 [QuTiP Tutorial 网页](https://qutip.org/tutorials.html)。


## 简介

在绝热量子计算中，先制备哈密顿量 $H_0$ 的易制备基态，
然后逐步将哈密顿量变换到 $H_1$，并使 $H_1$ 的基态编码困难问题解。
例如可写为

$\displaystyle H(t) = \lambda(t) H_0 + (1 - \lambda(t)) H_1$

其中当 $t$ 从 $0$ 演化到 $t_{\rm final}$ 时，$\lambda(t)$ 从 0 变化到 1。

若该变换足够慢（满足绝热条件），
系统演化将保持在基态。

若从 $H_0$ 到 $H_1$ 的变换过快，
系统会从基态激发，绝热算法失败。

本 notebook 研究一个自旋哈密顿量的动力学：
从一个基态易制备的简单哈密顿量，
逐步变换为具有复杂基态的随机自旋哈密顿量。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (QobjEvo, about, basis, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor)

%matplotlib inline
```

### 参数

```python
N = 6  # number of spins
M = 20  # number of eigenenergies to plot

# array of spin energy splittings and coupling strengths (random values).
h = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))
Jz = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))
Jx = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))
Jy = 1.0 * 2 * np.pi * (1 - 2 * np.random.rand(N))

# increase taumax to get make the sweep more adiabatic
taumax = 5.0
taulist = np.linspace(0, taumax, 100)
```

### 预计算算符

```python
# pre-allocate operators
si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sy
    sy_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))
```

### 构造初态

```python
psi_list = [basis(2, 0) for n in range(N)]
psi0 = tensor(psi_list)
H0 = 0
for n in range(N):
    H0 += -0.5 * 2.5 * sz_list[n]
```

### 构造哈密顿量

```python
# energy splitting terms
H1 = 0
for n in range(N):
    H1 += -0.5 * h[n] * sz_list[n]

H1 = 0
for n in range(N - 1):
    # interaction terms
    H1 += -0.5 * Jx[n] * sx_list[n] * sx_list[n + 1]
    H1 += -0.5 * Jy[n] * sy_list[n] * sy_list[n + 1]
    H1 += -0.5 * Jz[n] * sz_list[n] * sz_list[n + 1]

# the time-dependent hamiltonian in list-function format
args = {"t_max": max(taulist)}
h_t = [
    [H0, lambda t, args: (args["t_max"] - t) / args["t_max"]],
    [H1, lambda t, args: t / args["t_max"]],
]

# transform Hamiltonian to QobjEvo
h_t = QobjEvo(h_t, args=args)
```

### 时间演化

```python
#
# callback function for each time-step
#
evals_mat = np.zeros((len(taulist), M))
P_mat = np.zeros((len(taulist), M))

idx = [0]


def process_rho(tau, psi):

    # evaluate the Hamiltonian with gradually switched on interaction
    H = h_t(tau)

    # find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0], :] = np.real(evals)

    # find the overlap between the eigenstates and psi
    for n, eket in enumerate(ekets):
        P_mat[idx[0], n] = abs((eket.dag() * psi)) ** 2

    idx[0] += 1
```

```python
# Evolve the system, request the solver to call process_rho at each time step.

mesolve(h_t, psi0, taulist, [], process_rho, args)
```

## 结果可视化

绘制能级以及对应占据概率（用能级线宽编码概率）。

```python
# rc('font', family='serif')
# rc('font', size='10')

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

#
# plot the energy eigenvalues
#

# first draw thin lines outlining the energy spectrum
for n in range(len(evals_mat[0, :])):
    ls, lw = ("b", 1) if n == 0 else ("k", 0.25)
    axes[0].plot(taulist / max(taulist), evals_mat[:, n] / (2 * np.pi),
                 ls, lw=lw)

# second, draw line that encode the occupation probability of each state in
# its linewidth. thicker line => high occupation probability.
for idx in range(len(taulist) - 1):
    for n in range(len(P_mat[0, :])):
        lw = 0.5 + 4 * P_mat[idx, n]
        if lw > 0.55:
            axes[0].plot(
                np.array([taulist[idx], taulist[idx + 1]]) / taumax,
                np.array([evals_mat[idx, n],
                          evals_mat[idx + 1, n]]) / (2 * np.pi),
                "r",
                linewidth=lw,
            )

axes[0].set_xlabel(r"$\tau$")
axes[0].set_ylabel("Eigenenergies")
axes[0].set_title(
    "Energyspectrum (%d lowest values) of a chain of %d spins.\n " % (M, N)
    + "The occupation probabilities are encoded in the red line widths."
)

#
# plot the occupation probabilities for the few lowest eigenstates
#
for n in range(len(P_mat[0, :])):
    if n == 0:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n], "r", linewidth=2)
    else:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n])

axes[1].set_xlabel(r"$\tau$")
axes[1].set_ylabel("Occupation probability")
axes[1].set_title(
    "Occupation probability of the %d lowest " % M
    + "eigenstates for a chain of %d spins" % N
)
axes[1].legend(("Ground state",));
```

### 软件版本：

```python
about()
```
