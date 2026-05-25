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

# 第 11 讲 - 超导 Josephson 电荷量子比特

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, about, plot_energy_levels, ket2dm, mesolve

%matplotlib inline
```

### 引言

Josephson 电荷量子比特的哈密顿量为

$\displaystyle H = \sum_n 4 E_C (n_g - n)^2 \left|n\right\rangle\left\langle n\right| - \frac{1}{2}E_J\sum_n\left(\left|n+1\right\rangle\left\langle n\right| + \left|n\right\rangle\left\langle n+1\right| \right)$

其中 $E_C$ 是充电能，$E_J$ 是 Josephson 能，$\left| n\right\rangle$ 是岛上含 $n$ 个 Cooper 对的电荷态。


#### 参考文献

 * [J. Koch et al, Phys. Rev. A 76, 042319 (2007)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319)
 * [Y.A. Pashkin et al, Quantum Inf Process 8, 55 (2009)](http://dx.doi.org/10.1007/s11128-009-0101-5)


### 辅助函数

下面会反复在不同参数下生成电荷量子比特哈密顿量，并绘制本征能级，因此先定义两个辅助函数。

```python
def hamiltonian(Ec, Ej, N, ng):
    """
    Return the charge qubit hamiltonian as a Qobj instance.
    """
    m = np.diag(4 * Ec * (np.arange(-N, N + 1) - ng) ** 2) + 0.5 * Ej * (
        np.diag(-np.ones(2 * N), 1) + np.diag(-np.ones(2 * N), -1)
    )
    return Qobj(m)
```

```python
def plot_energies(ng_vec, energies, ymax=(20, 3)):
    """
    Plot energy levels as a function of bias parameter ng_vec.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for n in range(len(energies[0, :])):
        axes[0].plot(ng_vec, energies[:, n])
    axes[0].set_ylim(-2, ymax[0])
    axes[0].set_xlabel(r"$n_g$", fontsize=18)
    axes[0].set_ylabel(r"$E_n$", fontsize=18)

    for n in range(len(energies[0, :])):
        axes[1].plot(
            ng_vec,
            (energies[:, n] - energies[:, 0]) /
            (energies[:, 1] - energies[:, 0]),
        )
    axes[1].set_ylim(-0.1, ymax[1])
    axes[1].set_xlabel(r"$n_g$", fontsize=18)
    axes[1].set_ylabel(r"$(E_n-E_0)/(E_1-E_0)$", fontsize=18)
    return fig, axes
```

```python
def visualize_dynamics(result, ylabel):
    """
    Plot the evolution of the expectation values stored in result.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(result.times, result.expect[0])

    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel(r"$t$", fontsize=16);
```

### 电荷量子比特区间（charge qubit regime）

```python
N = 10
Ec = 1.0
Ej = 1.0
```

```python
ng_vec = np.linspace(-4, 4, 200)

energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies);
```

```python
ng_vec = np.linspace(-1, 1, 200)

energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(7.5, 3.0));
```

### 中间区间

```python
ng_vec = np.linspace(-4, 4, 200)
```

```python
Ec = 1.0
Ej = 5.0
```

```python
energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(50, 3));
```

```python
Ec = 1.0
Ej = 10.0
```

```python
energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(50, 3));
```

### Transmon 区间

```python
Ec = 1.0
Ej = 50.0
```

```python
energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(50, 3));
```

可以看到：至少对最低几个能级而言，能级劈裂几乎与栅偏置 $n_g$ 无关，因此器件对电荷噪声不敏感。
但与此同时，最低两能级与更高能级不再明显分离（系统更像谐振子）。
不过仍保留一定非谐性，只要能控制向高能级泄漏，仍可作为量子比特使用。


## 聚焦最低两个能级

回到 charge 区间，再看最低几条能级：

```python
N = 10
Ec = 1.0
Ej = 1.0
```

```python
ng_vec = np.linspace(-1, 1, 200)
```

```python
energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
```

```python
plot_energies(ng_vec, energies, ymax=(10, 3));
```

可见在 $n_g = 0.5$ 附近，最低两能级与高能级有较好分离：

```python
ng_vec = np.linspace(0.25, 0.75, 200)
energies = np.array([hamiltonian(Ec, Ej, N, ng).eigenenergies()
                     for ng in ng_vec])
plot_energies(ng_vec, energies, ymax=(10, 1.1));
```

将系统调到 $n_g = 0.5$，详细查看哈密顿量与本征态：

```python
H = hamiltonian(Ec, Ej, N, 0.5)
```

```python
H
```

```python
evals, ekets = H.eigenstates()
```

本征能量已按序排列：

```python
evals
```

在最低两个本征态中，只有两个态具有显著权重：

```python
ekets[0].full() > 0.1
```

```python
abs(ekets[1].full()) > 0.1
```

用这两个孤立本征态定义量子比特基：

```python
psi_g = ekets[0]  # basis(2, 0)
psi_e = ekets[1]  # basis(2, 1)

# psi_g = basis(2, 0)
# psi_e = basis(2, 1)
```

对应 Pauli 矩阵：

```python
sx = psi_g * psi_e.dag() + psi_e * psi_g.dag()
```

```python
sz = psi_g * psi_g.dag() - psi_e * psi_e.dag()
```

以及有效量子比特哈密顿量

```python
evals[1] - evals[0]
```

```python
H0 = 0.5 * (evals[1] - evals[0]) * sz

A = 0.25  # some driving amplitude
Hd = 0.5 * A * sx  # obtained by driving ng(t),
# but now H0 is in the eigenbasis so the drive becomes a sigma_x
```

此时系统中仍保留很多不参与动力学的高能级，但它们目前还在哈密顿量中。

```python
qubit_evals = H0.eigenenergies()

qubit_evals - qubit_evals[0]
```

```python
fig = plt.figure(figsize=(4, 2))
plot_energy_levels([H0, Hd], fig=fig);
```

设想还可以施加 $\sigma_x$ 型驱动（例如外场）：

```python
Heff = [H0, [Hd, "sin(wd*t)"]]

args = {"wd": (evals[1] - evals[0])}
```

看一下量子比特初始在基态时的 Rabi 振荡：

```python
psi0 = psi_g
```

```python
tlist = np.linspace(0.0, 100.0, 500)
result = mesolve(Heff, psi0, tlist, [], e_ops=[ket2dm(psi_e)], args=args)
```

```python
visualize_dynamics(result, r"$\rho_{ee}$");
```

可以看到动力学几乎只在选出的两态内进行，向其他能级的泄漏很小。

与其在计算中保留所有不活跃态，我们可以将其消除，得到真正的二能级系统。

```python
np.where(abs(ekets[0].full().flatten()) > 0.1)[0]
```

```python
np.where(abs(ekets[1].full().flatten()) > 0.1)[0]
```

```python
keep_states = np.where(abs(ekets[1].full().flatten()) > 0.1)[0]
```

```python
H0 = Qobj(H0.full()[keep_states, :][:, keep_states])

H0
```

```python
Hd = Qobj(Hd.full()[keep_states, :][:, keep_states])

Hd
```

再看能级图，就只剩期望的两个能级。

```python
fig = plt.figure(figsize=(4, 2))
plot_energy_levels([H0, Hd], fig=fig);
```

```python
Heff = [H0, [Hd, "sin(wd*t)"]]

args = {"wd": (evals[1] - evals[0])}
```

```python
psi0 = Qobj(psi0.full()[keep_states, :])
```

```python
psi_e = Qobj(psi_e.full()[keep_states, :])
```

```python
tlist = np.linspace(0.0, 100.0, 500)
result = mesolve(Heff, psi0, tlist, [], e_ops=[ket2dm(psi_e)], args=args)
```

```python
visualize_dynamics(result, r"$\rho_{ee}$");
```

### 软件版本

```python
about()
```
