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

# 第 3A 讲 - Dicke 模型

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, destroy, entropy_vn, expect, hinton, jmat, ptrace,
                   qeye, steadystate, tensor, wigner)

%matplotlib inline
```

## 引言

Dicke 哈密顿量由一个腔模与 $N$ 个自旋-1/2 与腔耦合组成：

<center>
$\displaystyle H_D = \omega_0 \sum_{i=1}^N \sigma_z^{(i)} + \omega a^\dagger a + \sum_{i}^N \frac{\lambda}{\sqrt{N}}(a + a^\dagger)(\sigma_+^{(i)}+\sigma_-^{(i)})$

$\displaystyle H_D = \omega_0 J_z + \omega a^\dagger a +  \frac{\lambda}{\sqrt{N}}(a + a^\dagger)(J_+ + J_-)$
</center>
    
其中 $J_z$ 与 $J_\pm$ 是长度为 $j=N/2$ 的赝自旋的集体角动量算符：

<center>
$\displaystyle J_z = \sum_{i=1}^N \sigma_z^{(i)}$

$\displaystyle J_\pm = \sum_{i=1}^N \sigma_\pm^{(i)}$
</center>

### 参考文献

 * [R.H. Dicke, Phys. Rev. 93, 99-110 (1954)](https://journals.aps.org/pr/abstract/10.1103/PhysRev.93.99)


## 在 QuTiP 中搭建问题

```python
w = 1.0
w0 = 1.0

g = 1.0
gc = np.sqrt(w * w0) / 2  # critical coupling strength

kappa = 0.05
gamma = 0.15
```

```python
M = 16
N = 4
j = N / 2
n = 2 * j + 1

a = tensor(destroy(M), qeye(int(n)))
Jp = tensor(qeye(M), jmat(j, "+"))
Jm = tensor(qeye(M), jmat(j, "-"))
Jz = tensor(qeye(M), jmat(j, "z"))

H0 = w * a.dag() * a + w0 * Jz
H1 = 1.0 / np.sqrt(N) * (a + a.dag()) * (Jp + Jm)
H = H0 + g * H1

H
```

### 哈密顿量结构

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
hinton(H, ax=ax);
```

## 计算基态随腔-自旋耦合强度变化

```python
g_vec = np.linspace(0.01, 1.0, 20)

# Ground state and steady state for the Hamiltonian: H = H0 + g * H1
psi_gnd_list = [(H0 + g * H1).groundstate()[1] for g in g_vec]
```

## 腔基态占据概率

```python
n_gnd_vec = expect(a.dag() * a, psi_gnd_list)
Jz_gnd_vec = expect(Jz, psi_gnd_list)
```

```python
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(12, 4))

axes[0].plot(g_vec, n_gnd_vec, "b", linewidth=2, label="cavity occupation")
axes[0].set_ylim(0, max(n_gnd_vec))
axes[0].set_ylabel("Cavity gnd occ. prob.", fontsize=16)
axes[0].set_xlabel("interaction strength", fontsize=16)

axes[1].plot(g_vec, Jz_gnd_vec, "b", linewidth=2, label="cavity occupation")
axes[1].set_ylim(-j, j)
axes[1].set_ylabel(r"$\langle J_z\rangle$", fontsize=16)
axes[1].set_xlabel("interaction strength", fontsize=16)

fig.tight_layout()
```

## 腔 Wigner 函数和 Fock 分布随耦合强度变化

```python
psi_gnd_sublist = psi_gnd_list[::4]

xvec = np.linspace(-7, 7, 200)

fig_grid = (3, len(psi_gnd_sublist))
fig = plt.figure(figsize=(3 * len(psi_gnd_sublist), 9))

for idx, psi_gnd in enumerate(psi_gnd_sublist):

    # trace out the cavity density matrix
    rho_gnd_cavity = ptrace(psi_gnd, 0)

    # calculate its wigner function
    W = wigner(rho_gnd_cavity, xvec, xvec)

    # plot its wigner function
    ax = plt.subplot2grid(fig_grid, (0, idx))
    ax.contourf(xvec, xvec, W, 100)

    # plot its fock-state distribution
    ax = plt.subplot2grid(fig_grid, (1, idx))
    ax.bar(np.arange(0, M), np.real(rho_gnd_cavity.diag()),
           color="blue", alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, M)

# plot the cavity occupation probability in the ground state
ax = plt.subplot2grid(fig_grid, (2, 0), colspan=fig_grid[1])
ax.plot(g_vec, n_gnd_vec, "r", linewidth=2, label="cavity occupation")
ax.set_xlim(0, max(g_vec))
ax.set_ylim(0, max(n_gnd_vec) * 1.2)
ax.set_ylabel("Cavity gnd occ. prob.", fontsize=16)
ax.set_xlabel("interaction strength", fontsize=16)

for g in g_vec[::4]:
    ax.plot([g, g], [0, max(n_gnd_vec) * 1.2], "b:", linewidth=2.5)
```

### 自旋与腔之间的熵/纠缠

```python
entropy_tot = np.zeros(g_vec.shape)
entropy_cavity = np.zeros(g_vec.shape)
entropy_spin = np.zeros(g_vec.shape)

for idx, psi_gnd in enumerate(psi_gnd_list):

    rho_gnd_cavity = ptrace(psi_gnd, 0)
    rho_gnd_spin = ptrace(psi_gnd, 1)

    entropy_tot[idx] = entropy_vn(psi_gnd, 2)
    entropy_cavity[idx] = entropy_vn(rho_gnd_cavity, 2)
    entropy_spin[idx] = entropy_vn(rho_gnd_spin, 2)
```

```python
fig, axes = plt.subplots(1, 1, figsize=(12, 6))
axes.plot(
    g_vec, entropy_tot, "k", g_vec, entropy_cavity, "b", g_vec,
    entropy_spin, "r--"
)

axes.set_ylim(0, 1.5)
axes.set_ylabel("Entropy of subsystems", fontsize=16)
axes.set_xlabel("interaction strength", fontsize=16)

fig.tight_layout()
```

# 不同 N 下，熵随耦合强度变化

### 参考文献

* [Lambert et al., Phys. Rev. Lett. 92, 073602 (2004)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.92.073602).

```python
def calculate_entropy(M, N, g_vec):

    j = N / 2.0
    n = 2 * j + 1

    # setup the hamiltonian for the requested hilbert space sizes
    a = tensor(destroy(M), qeye(int(n)))
    Jp = tensor(qeye(M), jmat(j, "+"))
    Jm = tensor(qeye(M), jmat(j, "-"))
    Jz = tensor(qeye(M), jmat(j, "z"))

    H0 = w * a.dag() * a + w0 * Jz
    H1 = 1.0 / np.sqrt(N) * (a + a.dag()) * (Jp + Jm)

    # Ground state and steady state for the Hamiltonian: H = H0 + g * H1
    psi_gnd_list = [(H0 + g * H1).groundstate()[1] for g in g_vec]

    entropy_cavity = np.zeros(g_vec.shape)
    entropy_spin = np.zeros(g_vec.shape)

    for idx, psi_gnd in enumerate(psi_gnd_list):

        rho_gnd_cavity = ptrace(psi_gnd, 0)
        rho_gnd_spin = ptrace(psi_gnd, 1)

        entropy_cavity[idx] = entropy_vn(rho_gnd_cavity, 2)
        entropy_spin[idx] = entropy_vn(rho_gnd_spin, 2)

    return entropy_cavity, entropy_spin
```

```python
g_vec = np.linspace(0.2, 0.8, 60)
N_vec = [4, 8, 12, 16, 24, 32]
MM = 25

fig, axes = plt.subplots(1, 1, figsize=(12, 6))

for NN in N_vec:

    entropy_cavity, entropy_spin = calculate_entropy(MM, NN, g_vec)

    axes.plot(g_vec, entropy_cavity, "b", label="N = %d" % NN)
    axes.plot(g_vec, entropy_spin, "r--")

axes.set_ylim(0, 1.75)
axes.set_ylabel("Entropy of subsystems", fontsize=16)
axes.set_xlabel("interaction strength", fontsize=16)
axes.legend();
```

# 含耗散腔：用稳态代替基态

```python
# average number thermal photons in the bath coupling to the resonator
n_th = 0.25

c_ops = [np.sqrt(kappa * (n_th + 1)) * a, np.sqrt(kappa * n_th) * a.dag()]
# c_ops = [sqrt(kappa) * a, sqrt(gamma) * Jm]
```

## 计算稳态随腔-自旋耦合强度变化

```python
g_vec = np.linspace(0.01, 1.0, 20)

# Ground state for the Hamiltonian: H = H0 + g * H1
rho_ss_list = [steadystate(H0 + g * H1, c_ops) for g in g_vec]
```

## 腔稳态占据概率

```python
# calculate the expectation value of the number of photons in the cavity
n_ss_vec = expect(a.dag() * a, rho_ss_list)
```

```python
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))

axes.plot(g_vec, n_gnd_vec, "b", linewidth=2, label="cavity groundstate")
axes.plot(g_vec, n_ss_vec, "r", linewidth=2, label="cavity steadystate")
axes.set_ylim(0, max(n_ss_vec))
axes.set_ylabel("Cavity occ. prob.", fontsize=16)
axes.set_xlabel("interaction strength", fontsize=16)
axes.legend(loc=0)

fig.tight_layout()
```

## 腔 Wigner 函数和 Fock 分布随耦合强度变化

```python
rho_ss_sublist = rho_ss_list[::4]

xvec = np.linspace(-6, 6, 200)

fig_grid = (3, len(rho_ss_sublist))
fig = plt.figure(figsize=(3 * len(rho_ss_sublist), 9))

for idx, rho_ss in enumerate(rho_ss_sublist):

    # trace out the cavity density matrix
    rho_ss_cavity = ptrace(rho_ss, 0)

    # calculate its wigner function
    W = wigner(rho_ss_cavity, xvec, xvec)

    # plot its wigner function
    ax = plt.subplot2grid(fig_grid, (0, idx))
    ax.contourf(xvec, xvec, W, 100)

    # plot its fock-state distribution
    ax = plt.subplot2grid(fig_grid, (1, idx))
    ax.bar(np.arange(0, M), np.real(rho_ss_cavity.diag()), color="blue",
           alpha=0.6)
    ax.set_ylim(0, 1)

# plot the cavity occupation probability in the ground state
ax = plt.subplot2grid(fig_grid, (2, 0), colspan=fig_grid[1])
ax.plot(g_vec, n_gnd_vec, "b", linewidth=2, label="cavity groundstate")
ax.plot(g_vec, n_ss_vec, "r", linewidth=2, label="cavity steadystate")
ax.set_xlim(0, max(g_vec))
ax.set_ylim(0, max(n_ss_vec) * 1.2)
ax.set_ylabel("Cavity gnd occ. prob.", fontsize=16)
ax.set_xlabel("interaction strength", fontsize=16)

for g in g_vec[::4]:
    ax.plot([g, g], [0, max(n_ss_vec) * 1.2], "b:", linewidth=5)
```

## 熵

```python
entropy_tot = np.zeros(g_vec.shape)
entropy_cavity = np.zeros(g_vec.shape)
entropy_spin = np.zeros(g_vec.shape)

for idx, rho_ss in enumerate(rho_ss_list):

    rho_gnd_cavity = ptrace(rho_ss, 0)
    rho_gnd_spin = ptrace(rho_ss, 1)

    entropy_tot[idx] = entropy_vn(rho_ss, 2)
    entropy_cavity[idx] = entropy_vn(rho_gnd_cavity, 2)
    entropy_spin[idx] = entropy_vn(rho_gnd_spin, 2)
```

```python
fig, axes = plt.subplots(1, 1, figsize=(12, 6))

axes.plot(g_vec, entropy_tot, "k", label="total")
axes.plot(g_vec, entropy_cavity, "b", label="cavity")
axes.plot(g_vec, entropy_spin, "r--", label="spin")

axes.set_ylabel("Entropy of subsystems", fontsize=16)
axes.set_xlabel("interaction strength", fontsize=16)
axes.legend(loc=0)
fig.tight_layout()
```

### 软件版本

```python
about()
```

```python

```
