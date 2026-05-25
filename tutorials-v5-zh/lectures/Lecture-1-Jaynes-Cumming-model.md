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

# 第 1 讲 - Jaynes-Cummings 模型中的真空 Rabi 振荡


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, destroy, mesolve, ptrace, qeye,
                   tensor, wigner, anim_wigner)
# set a parameter to see animations in line
from matplotlib import rc
rc('animation', html='jshtml')

%matplotlib inline
```

# 引言

Jaynes-Cummings 模型是量子光-物质相互作用最简单的模型：描述单个二能级原子与单个电磁腔模的相互作用。该系统哈密顿量（偶极相互作用形式）为

### $H = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger + a)(\sigma_- + \sigma_+)$

在旋波近似下为

### $H_{\rm RWA} = \hbar \omega_c a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g(a^\dagger\sigma_- + a\sigma_+)$

其中 $\omega_c$ 与 $\omega_a$ 分别是腔与原子频率，$g$ 是耦合强度。

<!-- #region -->
### 问题参数


这里采用单位 $\hbar = 1$：
<!-- #endregion -->

```python
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
kappa = 0.005  # cavity dissipation rate
gamma = 0.05  # atom dissipation rate
N = 15  # number of cavity fock states
n_th_a = 0.0  # avg number of thermal bath excitation
use_rwa = True

tlist = np.linspace(0, 25, 101)
```

### 设置算符、哈密顿量与初态

```python
# initial state
psi0 = tensor(basis(N, 0), basis(2, 1))  # start with an excited atom

# operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))

# Hamiltonian
if use_rwa:
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() * sm + a * sm.dag())
else:
    H = wc * a.dag() * a + wa * sm.dag() * sm + \
        g * (a.dag() + a) * (sm + sm.dag())
```

### 构造描述耗散的塌缩算符列表

```python
c_ops = []

# cavity relaxation
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a)

# cavity excitation, if temperature > 0
rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a.dag())

# qubit relaxation
rate = gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sm)
```

### 演化系统

这里使用 Lindblad 主方程求解器演化系统，并通过把 `[a.dag()*a, sm.dag()*sm]` 作为第五个参数传入，要求求解器返回算符 $a^\dagger a$ 和 $\sigma_+\sigma_-$ 的期望值。

```python
output = mesolve(H, psi0, tlist, c_ops, e_ops=[a.dag() * a, sm.dag() * sm])
```

## 可视化结果

这里绘制腔与原子的激发概率（这些期望值由上面的 `mesolve` 计算得到）。可以清楚看到能量在腔与原子之间相干往返传递。

```python
n_c = output.expect[0]
n_a = output.expect[1]

fig, axes = plt.subplots(1, 1, figsize=(10, 6))

axes.plot(tlist, n_c, label="Cavity")
axes.plot(tlist, n_a, label="Atom excited state")
axes.legend(loc=0)
axes.set_xlabel("Time")
axes.set_ylabel("Occupation probability")
axes.set_title("Vacuum Rabi oscillations");
```

## 腔模 Wigner 函数

除了腔和原子的激发概率，我们还可能关心随时间变化的 Wigner 函数。Wigner 函数能帮助我们理解谐振器状态的量子性质。

在 QuTiP 中计算 Wigner 函数时，先重新做一次演化且不指定期望值算符，这样求解器会返回各时刻系统密度矩阵列表。

```python
output = mesolve(H, psi0, tlist, c_ops)
```

现在 `output.states` 就是 `tlist` 所给时刻的系统密度矩阵列表：

```python
output
```

```python
type(output.states)
```

```python
len(output.states)
```

```python
# indexing the list with -1 results in the last element in the list
output.states[-1]
```

现在看原子处于基态附近的时刻：$t = \{5, 15, 25\}$（见上图）。

对每个时刻都要做三步：

 1. 找到对应时刻的系统密度矩阵。
 2. 对原子做偏迹，得到腔的约化密度矩阵。
 3. 计算并可视化该约化密度矩阵的 Wigner 函数。

```python
# find the indices of the density matrices for the times we are interested in
t_idx = np.where([tlist == t for t in [0.0, 5.0, 15.0, 25.0]])[1]
tlist[t_idx]
```

```python
# get a list density matrices
rho_list = [output.states[i] for i in t_idx]
```

```python
# loop over the list of density matrices

xvec = np.linspace(-3, 3, 200)

fig, axes = plt.subplots(1, len(rho_list), sharex=True,
                         figsize=(3 * len(rho_list), 3))

for idx, rho in enumerate(rho_list):

    # trace out the atom from the density matrix, to obtain
    # the reduced density matrix for the cavity
    rho_cavity = ptrace(rho, 0)

    # calculate its wigner function
    W = wigner(rho_cavity, xvec, xvec)

    # plot its wigner function
    axes[idx].contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-0.25, 0.25),
        cmap=plt.get_cmap("RdBu"),
    )

    axes[idx].set_title(r"$t = %.1f$" % tlist[t_idx][idx], fontsize=16)
```

在 $t =0$ 时，腔处于基态。于 $t = 5, 15, 25$ 时，腔在该真空 Rabi 振荡过程中达到最大占据。可注意到在 $t=5$ 和 $t=15$ 时，Wigner 函数出现负值，表明系统处于真正的量子态。而在 $t=25$，Wigner 函数不再出现负值，因此可视为经典态。

此外，`qutip.anim_wigner` 很适合可视化振荡过程。它内部会计算 Wigner 函数，因此只需传入约化密度矩阵。还可传选项，例如 `projection='3d'` 会生成三维图，便于观察振荡。

```python
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=0)

rho_cavity = list()

xvec = np.linspace(-3, 3, 150)

for idx, rho in enumerate(output.states):
    rho_cavity.append(ptrace(rho, 0))

fig, ani = anim_wigner(rho_cavity, xvec, xvec, projection='3d',
                       colorbar=True, fig=fig, ax=ax)

# close an auto-generated plot and animation
plt.close()
ani
```

### 同一结果的另一种视图

```python
t_idx = np.where([tlist == t for t in [0.0, 5.0, 10, 15, 20, 25]])[1]
rho_list = [output.states[i] for i in t_idx]

fig_grid = (2, len(rho_list) * 2)
fig = plt.figure(figsize=(2.5 * len(rho_list), 5))

for idx, rho in enumerate(rho_list):
    rho_cavity = ptrace(rho, 0)
    W = wigner(rho_cavity, xvec, xvec)
    ax = plt.subplot2grid(fig_grid, (0, 2 * idx), colspan=2)
    ax.contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-0.25, 0.25),
        cmap=plt.get_cmap("RdBu"),
    )
    ax.set_title(r"$t = %.1f$" % tlist[t_idx][idx], fontsize=16)

# plot the cavity occupation probability in the ground state
ax = plt.subplot2grid(fig_grid, (1, 1), colspan=(fig_grid[1] - 2))
ax.plot(tlist, n_c, label="Cavity")
ax.plot(tlist, n_a, label="Atom excited state")
ax.legend()
ax.set_xlabel("Time")
ax.set_ylabel("Occupation probability");
```

### 软件版本

```python
about()
```

```python

```
