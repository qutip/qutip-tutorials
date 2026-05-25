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

# 第 3B 讲 - 超强耦合区的类 Jaynes-Cummings 模型


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, destroy, entropy_vn, expect, mesolve, ptrace,
                   qeye, tensor, wigner)

%matplotlib inline
```

<!-- #region -->
# 引言

在 Jaynes-Cummings 模型中，原子与腔场之间的偶极相互作用通常被假设为弱，从而可采用旋波近似（RWA）。
当耦合强度增大时，RWA 不再成立；在非常强耦合下，原子-腔系统基态会出现有趣性质。

用 QuTiP 探索该现象时，可考虑哈密顿量

### $H = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger + a)(\sigma_- + \sigma_+)$.

注意这里相互作用项没有做 RWA。若做 RWA，则哈密顿量为

### $H_{\rm RWA} = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger\sigma_- + a\sigma_+)$.

本 notebook 计算哈密顿量 $H$ 的基态随耦合强度 $g$ 的变化（可尝试将 `use_rwa = True` 改用 $H_{\rm RWA}$）。

当 $g$ 相比 $H$ 中其他能标都很大时，称为超强耦合区（ultrastrong coupling），近年来研究非常活跃。参考文献如下。


References:

 * [P. Nataf et al., Phys. Rev. Lett. 104, 023601 (2010)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.023601)

 * [J. Casanova et al., Phys. Rev. Lett. 105, 26360 (2010)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.263603).

 * [S. Ashhab et al., Phys. Rev. A 81, 042311 (2010)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.81.042311)
<!-- #endregion -->

<!-- #region -->
### 问题参数


这里采用单位 $\hbar = 1$： 
<!-- #endregion -->

```python
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency

N = 15  # number of cavity fock states
use_rwa = False
```

### 设置算符与哈密顿量

```python
# operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))

na = sm.dag() * sm  # atom
nc = a.dag() * a  # cavity

# decoupled Hamiltonian
H0 = wc * a.dag() * a + wa * sm.dag() * sm

# interaction Hamiltonian
if use_rwa:
    H1 = a.dag() * sm + a * sm.dag()
else:
    H1 = (a.dag() + a) * (sm + sm.dag())
```

## 计算基态随耦合强度的变化

```python
g_vec = np.linspace(0, 2.0, 101) * 2 * np.pi  # coupling strength vector

psi_list = []

for g in g_vec:

    H = H0 + g * H1

    # find the groundstate and its energy
    gnd_energy, gnd_state = H.groundstate()

    # store the ground state
    psi_list.append(gnd_state)
```

计算上述基态中的腔与原子激发概率：

```python
na_expt = expect(na, psi_list)  # qubit  occupation probability
nc_expt = expect(nc, psi_list)  # cavity occupation probability
```

绘制腔与原子基态占据概率随耦合强度变化。注意在大耦合强度（超强耦合区，$g > \omega_a,\omega_c$）下，基态同时包含光子与原子激发。

```python
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))

axes.plot(g_vec / (2 * np.pi), nc_expt, "r", linewidth=2, label="cavity")
axes.plot(g_vec / (2 * np.pi), na_expt, "b", linewidth=2, label="atom")
axes.set_ylabel("Occupation probability", fontsize=16)
axes.set_xlabel("coupling strength", fontsize=16)
axes.legend(loc=0)

fig.tight_layout()
```

# 绘制腔模 Wigner 函数随耦合强度变化

```python
g_idx = np.where([g_vec == 2 * np.pi * g for
                  g in [0.0, 0.5, 1.0, 1.5, 2.0]])[1]
psi_sublist = []
for idx in g_idx:
    psi_sublist.append(psi_list[idx])

xvec = np.linspace(-5, 5, 200)

fig_grid = (2, len(psi_sublist) * 2)
fig = plt.figure(figsize=(3 * len(psi_sublist), 6))

for idx, psi in enumerate(psi_sublist):
    rho_cavity = ptrace(psi, 0)
    W = wigner(rho_cavity, xvec, xvec)
    ax = plt.subplot2grid(fig_grid, (0, 2 * idx), colspan=2)
    ax.contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-0.125, 0.125),
        cmap=plt.get_cmap("RdBu"),
    )
    ax.set_title(r"$g = %.1f$" % (g_vec[g_idx][idx] / (2 * np.pi)),
                 fontsize=16)

# plot the cavity occupation probability in the ground state
ax = plt.subplot2grid(fig_grid, (1, 1), colspan=(fig_grid[1] - 2))
ax.plot(g_vec / (2 * np.pi), nc_expt, label="Cavity")
ax.plot(g_vec / (2 * np.pi), na_expt, label="Atom excited state")
ax.legend(loc=0)
ax.set_xlabel("coupling strength")
ax.set_ylabel("Occupation probability");
```

## 用原子/腔熵衡量纠缠

```python
entropy_cavity = np.zeros(g_vec.shape)
entropy_atom = np.zeros(g_vec.shape)

for idx, psi in enumerate(psi_list):

    rho_cavity = ptrace(psi, 0)
    entropy_cavity[idx] = entropy_vn(rho_cavity, 2)

    rho_atom = ptrace(psi, 1)
    entropy_atom[idx] = entropy_vn(rho_atom, 2)
```

```python
fig, axes = plt.subplots(1, 1, figsize=(12, 6))
axes.plot(g_vec / (2 * np.pi), entropy_cavity, "b", label="cavity",
          linewidth=2)
axes.plot(g_vec / (2 * np.pi), entropy_atom, "r--", label="atom", linewidth=2)
axes.set_ylim(0, 1)
axes.set_ylabel("entropy", fontsize=16)
axes.set_xlabel("coupling strength", fontsize=16)
axes.legend(loc=0);
```

## 初始腔激发态的动力学

```python
H = H0 + 1.0 * 2 * np.pi * H1

psi0 = tensor(basis(N, 1), basis(2, 0))
```

```python
tlist = np.linspace(0, 20, 1000)
output = mesolve(H, psi0, tlist, [], e_ops=[a.dag() * a, sm.dag() * sm])
```

```python
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))

axes.plot(tlist, np.real(output.expect[0]), "r", linewidth=2, label="cavity")
axes.plot(tlist, np.real(output.expect[1]), "b", linewidth=2, label="atom")
axes.legend(loc=0)

fig.tight_layout()
```

### 腔模 Fock 分布与 Wigner 函数随时间变化

```python
tlist = np.linspace(0, 0.35, 8)
output = mesolve(H, psi0, tlist, [])
```

```python
rho_ss_sublist = output.states  # [::4]

xvec = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, len(rho_ss_sublist),
                         figsize=(2 * len(rho_ss_sublist), 4))

for idx, rho_ss in enumerate(rho_ss_sublist):

    # trace out the cavity density matrix
    rho_ss_cavity = ptrace(rho_ss, 0)

    # calculate its wigner function
    W = wigner(rho_ss_cavity, xvec, xvec)

    # plot its wigner function
    axes[0, idx].contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-0.25, 0.25),
        cmap=plt.get_cmap("RdBu"),
    )

    # plot its fock-state distribution
    axes[1, idx].bar(
        np.arange(0, N), np.real(rho_ss_cavity.diag()), color="blue", alpha=0.6
    )
    axes[1, idx].set_ylim(0, 1)
    axes[1, idx].set_xlim(0, N)
```

### 加入少量耗散后的同样分析

```python
kappa = 0.25
```

```python
tlist = np.linspace(0, 20, 1000)
output = mesolve(H, psi0, tlist, [np.sqrt(kappa) * a],
                 e_ops=[a.dag() * a, sm.dag() * sm])
```

```python
fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))
axes.plot(tlist, output.expect[0], "r", linewidth=2, label="cavity")
axes.plot(tlist, output.expect[1], "b", linewidth=2, label="atom")
axes.legend(loc=0);
```

```python
tlist = np.linspace(0, 10, 8)
output = mesolve(H, psi0, tlist, [np.sqrt(kappa) * a])
```

```python
xvec = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, len(output.states),
                         figsize=(2 * len(output.states), 4))

for idx, rho_ss in enumerate(output.states):

    # trace out the cavity density matrix
    rho_ss_cavity = ptrace(rho_ss, 0)

    # calculate its wigner function
    W = wigner(rho_ss_cavity, xvec, xvec)

    # plot its wigner function
    axes[0, idx].contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-0.25, 0.25),
        cmap=plt.get_cmap("RdBu"),
    )

    # plot its fock-state distribution
    axes[1, idx].bar(
        np.arange(0, N), np.real(rho_ss_cavity.diag()), color="blue", alpha=0.6
    )
    axes[1, idx].set_ylim(0, 1)
    axes[1, idx].set_xlim(0, N)
```

### 存在耗散且从理想基态出发时，熵随时间变化

```python
tlist = np.linspace(0, 30, 50)

psi0 = H.groundstate()[1]

output = mesolve(H, psi0, tlist, [np.sqrt(kappa) * a])
```

```python
entropy_tot = np.zeros(tlist.shape)
entropy_cavity = np.zeros(tlist.shape)
entropy_atom = np.zeros(tlist.shape)

for idx, rho in enumerate(output.states):

    entropy_tot[idx] = entropy_vn(rho, 2)

    rho_cavity = ptrace(rho, 0)
    entropy_cavity[idx] = entropy_vn(rho_cavity, 2)

    rho_atom = ptrace(rho, 1)
    entropy_atom[idx] = entropy_vn(rho_atom, 2)
```

```python
fig, axes = plt.subplots(1, 1, figsize=(12, 6))
axes.plot(tlist, entropy_tot, "k", label="total", linewidth=2)
axes.plot(tlist, entropy_cavity, "b", label="cavity", linewidth=2)
axes.plot(tlist, entropy_atom, "r--", label="atom", linewidth=2)
axes.set_ylabel("entropy", fontsize=16)
axes.set_xlabel("coupling strength", fontsize=16)
axes.set_ylim(0, 1.5)
axes.legend(loc=0);
```

### 软件版本

```python
about()
```

```python

```
