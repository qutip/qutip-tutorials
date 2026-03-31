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

# 第 2B 讲 - 单原子激光


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
# setup the matplotlib graphics library and configure it to show
# figures inline in the notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import (about, basis, destroy, expect, mesolve,
                   ptrace, qeye, sigmax, steadystate, tensor, wigner)

%matplotlib inline
```

# 引言与模型

考虑单个原子与单个腔模耦合，如下图所示。当原子激发率 $\Gamma$ 超过弛豫率时，原子可出现布居反转；若再与腔耦合，原子就能作为腔内光子泵浦源。

```python
Image(filename="images/schematic-lasing-model.png")
```

该模型的相干动力学由哈密顿量描述：

$H = \hbar \omega_0 a^\dagger a + \frac{1}{2}\hbar\omega_a\sigma_z + \hbar g\sigma_x(a^\dagger + a)$

其中 $\omega_0$ 为腔能级劈裂，$\omega_a$ 为原子能级劈裂，$g$ 为原子-腔耦合强度。

除相干动力学外，还包含以下非相干过程： 

1. 腔的弛豫与热激发（速率相关于 $\kappa$）；
2. 原子激发（泵浦）过程，速率 $\Gamma$。

模型对应 Lindblad 主方程：

$$\frac{d}{dt}\rho = -i[H, \rho] + \Gamma\left(\sigma_+\rho\sigma_- - \frac{1}{2}\sigma_-\sigma_+\rho - \frac{1}{2}\rho\sigma_-\sigma_+\right)
+ \kappa (1 + n_{\rm th}) \left(a\rho a^\dagger - \frac{1}{2}a^\dagger a\rho - \frac{1}{2}\rho a^\dagger a\right)
+ \kappa n_{\rm th} \left(a^\dagger\rho a - \frac{1}{2}a a^\dagger \rho - \frac{1}{2}\rho a a^\dagger\right)$$

（取单位 $\hbar = 1$）。

References:

 * [Yi Mu, C.M. Savage, Phys. Rev. A 46, 5944 (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.46.5944)

 * [D.A. Rodrigues, J. Imbers, A.D. Armour, Phys. Rev. Lett. 98, 067204 (2007)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.067204)

 * [S. Ashhab, J.R. Johansson, A.M. Zagoskin, F. Nori, New J. Phys. 11, 023030 (2009)](http://dx.doi.org/10.1088/1367-2630/11/2/023030)


### 问题参数

```python
w0 = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength

kappa = 0.04  # cavity dissipation rate
gamma = 0.00  # atom dissipation rate
Gamma = 0.35  # atom pump rate

N = 50  # number of cavity fock states
n_th_a = 0.0  # avg number of thermal bath excitation

tlist = np.linspace(0, 150, 101)
```

### 设置算符、哈密顿量与初态

```python
# initial state
psi0 = tensor(basis(N, 0), basis(2, 0))  # start without excitations

# operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
sx = tensor(qeye(N), sigmax())

# Hamiltonian
H = w0 * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() + a) * sx
```

### 构造耗散塌缩算符列表

```python
# collapse operators
c_ops = []

rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a)

rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a.dag())

rate = gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sm)

rate = Gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sm.dag())
```

### 演化系统

这里用 Lindblad 主方程求解器演化系统，并将 `[a.dag()*a, sm.dag()*sm]` 作为第五个参数传入，返回算符 $a^\dagger a$ 与 $\sigma_+\sigma_-$ 的期望值。

```python
opt = {'nsteps': 2000}  # allow extra time-steps
output = mesolve(H, psi0, tlist, c_ops, e_ops=[a.dag() * a, sm.dag() * sm],
                 options=opt)
```

## 结果可视化

绘制腔与原子的激发概率（由上面的 `mesolve` 计算得到）。

```python
n_c = output.expect[0]
n_a = output.expect[1]

fig, axes = plt.subplots(1, 1, figsize=(8, 6))

axes.plot(tlist, n_c, label="Cavity")
axes.plot(tlist, n_a, label="Atom excited state")
axes.set_xlim(0, 150)
axes.legend(loc=0)
axes.set_xlabel("Time")
axes.set_ylabel("Occupation probability");
```

## 稳态：腔 Fock 分布与 Wigner 函数

```python
rho_ss = steadystate(H, c_ops)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

xvec = np.linspace(-5, 5, 200)

rho_cavity = ptrace(rho_ss, 0)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()

axes[1].contourf(
    xvec,
    xvec,
    W,
    100,
    norm=mpl.colors.Normalize(-wlim, wlim),
    cmap=plt.get_cmap("RdBu"),
)
axes[1].set_xlabel(r"Im $\alpha$", fontsize=18)
axes[1].set_ylabel(r"Re $\alpha$", fontsize=18)

axes[0].bar(np.arange(0, N), np.real(rho_cavity.diag()), color="blue",
            alpha=0.6)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, N)
axes[0].set_xlabel("Fock number", fontsize=18)
axes[0].set_ylabel("Occupation probability", fontsize=18);
```

## 腔 Fock 分布与 Wigner 函数随时间变化

```python
tlist = np.linspace(0, 25, 5)
output = mesolve(H, psi0, tlist, c_ops,
                 options={'nsteps': 5000})
```

```python
rho_ss_sublist = output.states

xvec = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, len(rho_ss_sublist),
                         figsize=(3 * len(rho_ss_sublist), 6))

for idx, rho_ss in enumerate(rho_ss_sublist):

    # trace out the cavity density matrix
    rho_ss_cavity = ptrace(rho_ss, 0)

    # calculate its wigner function
    W = wigner(rho_ss_cavity, xvec, xvec)

    # plot its wigner function
    wlim = abs(W).max()
    axes[0, idx].contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-wlim, wlim),
        cmap=plt.get_cmap("RdBu"),
    )
    axes[0, idx].set_title(r"$t = %.1f$" % tlist[idx])

    # plot its fock-state distribution
    axes[1, idx].bar(np.arange(0, N), np.real(rho_ss_cavity.diag()),
                     color="blue", alpha=0.8)
    axes[1, idx].set_ylim(0, 1)
    axes[1, idx].set_xlim(0, 15)
```

## 稳态平均腔光子数随泵浦率变化

References:

 * [S. Ashhab, J.R. Johansson, A.M. Zagoskin, F. Nori, New J. Phys. 11, 023030 (2009)](http://dx.doi.org/10.1088/1367-2630/11/2/023030)

```python
def calulcate_avg_photons(N, Gamma):

    # collapse operators
    c_ops = []

    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a)

    rate = kappa * n_th_a
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a.dag())

    rate = gamma
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * sm)

    rate = Gamma
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * sm.dag())

    # Ground state and steady state for the Hamiltonian: H = H0 + g * H1
    rho_ss = steadystate(H, c_ops)

    # cavity photon number
    n_cavity = expect(a.dag() * a, rho_ss)

    # cavity second order coherence function
    g2_cavity = expect(a.dag() * a.dag() * a * a, rho_ss) / (n_cavity**2)

    return n_cavity, g2_cavity
```

```python
Gamma_max = 2 * (4 * g**2) / kappa
Gamma_vec = np.linspace(0.1, Gamma_max, 50)

n_avg_vec = []
g2_vec = []

for Gamma in Gamma_vec:
    n_avg, g2 = calulcate_avg_photons(N, Gamma)
    n_avg_vec.append(n_avg)
    g2_vec.append(g2)
```

```python
fig, axes = plt.subplots(1, 1, figsize=(12, 6))

axes.plot(
    Gamma_vec * kappa / (4 * g**2),
    n_avg_vec,
    color="blue",
    alpha=0.6,
    label="numerical",
)

axes.set_xlabel(r"$\Gamma\kappa/(4g^2)$", fontsize=18)
axes.set_ylabel(r"Occupation probability $\langle n \rangle$",
                fontsize=18)
axes.set_xlim(0, 2);
```

```python
fig, axes = plt.subplots(1, 1, figsize=(12, 6))

axes.plot(Gamma_vec * kappa / (4 * g**2), g2_vec, color="blue", alpha=0.6,
          label="numerical")

axes.set_xlabel(r"$\Gamma\kappa/(4g^2)$", fontsize=18)
axes.set_ylabel(r"$g^{(2)}(0)$", fontsize=18)
axes.set_xlim(0, 2)
axes.text(0.1, 1.1, "Lasing regime", fontsize=16)
axes.text(1.5, 1.8, "Thermal regime", fontsize=16);
```

<!-- #region -->
这里可以看到，当 $\Gamma\kappa/(4g^2) > 1$ 时，激光过程被抑制。 


下面看两种情况下的 Fock 分布：
$\Gamma\kappa/(4g^2) = 0.5$（激光区）与 $\Gamma\kappa/(4g^2) = 1.5$（抑制区）。
<!-- #endregion -->

### 情况 1：$\Gamma\kappa/(4g^2) = 0.5$

```python
Gamma = 0.5 * (4 * g**2) / kappa
```

```python
c_ops = [
    np.sqrt(kappa * (1 + n_th_a)) * a,
    np.sqrt(kappa * n_th_a) * a.dag(),
    np.sqrt(gamma) * sm,
    np.sqrt(Gamma) * sm.dag(),
]

rho_ss = steadystate(H, c_ops)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

xvec = np.linspace(-10, 10, 200)

rho_cavity = ptrace(rho_ss, 0)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()
axes[1].contourf(
    xvec,
    xvec,
    W,
    100,
    norm=mpl.colors.Normalize(-wlim, wlim),
    cmap=plt.get_cmap("RdBu"),
)
axes[1].set_xlabel(r"Im $\alpha$", fontsize=18)
axes[1].set_ylabel(r"Re $\alpha$", fontsize=18)

axes[0].bar(np.arange(0, N), np.real(rho_cavity.diag()), color="blue",
            alpha=0.6)
axes[0].set_xlabel(r"$n$", fontsize=18)
axes[0].set_ylabel(r"Occupation probability", fontsize=18)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, N);
```

### 情况 2：$\Gamma\kappa/(4g^2) = 1.5$

```python
Gamma = 1.5 * (4 * g**2) / kappa
```

```python
c_ops = [
    np.sqrt(kappa * (1 + n_th_a)) * a,
    np.sqrt(kappa * n_th_a) * a.dag(),
    np.sqrt(gamma) * sm,
    np.sqrt(Gamma) * sm.dag(),
]

rho_ss = steadystate(H, c_ops)
```

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

xvec = np.linspace(-10, 10, 200)

rho_cavity = ptrace(rho_ss, 0)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()
axes[1].contourf(
    xvec,
    xvec,
    W,
    100,
    norm=mpl.colors.Normalize(-wlim, wlim),
    cmap=plt.get_cmap("RdBu"),
)
axes[1].set_xlabel(r"Im $\alpha$", fontsize=18)
axes[1].set_ylabel(r"Re $\alpha$", fontsize=18)

axes[0].bar(np.arange(0, N), np.real(rho_cavity.diag()), color="blue",
            alpha=0.6)
axes[0].set_xlabel(r"$n$", fontsize=18)
axes[0].set_ylabel(r"Occupation probability", fontsize=18)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, N);
```

泵浦率 $\Gamma$ 过大将破坏激光过程，即“反阈值”现象。


### 软件版本

```python
about()
```
