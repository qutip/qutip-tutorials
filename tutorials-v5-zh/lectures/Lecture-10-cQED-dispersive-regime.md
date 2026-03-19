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

# 第 10 讲 - 色散区的腔 QED

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, coherent, correlation_2op_1t,
                   destroy, expect, mesolve, ptrace, qeye, sigmax, sigmaz,
                   spectrum_correlation_fft, tensor, wigner)

%matplotlib inline
```

# 引言

一个量子比特-谐振器系统可由如下哈密顿量描述：

$\displaystyle H = \omega_r a^\dagger a - \frac{1}{2} \omega_q \sigma_z + g (a^\dagger + a) \sigma_x$

其中 $\omega_r$ 和 $\omega_q$ 分别是谐振器与量子比特的裸频率，$g$ 为偶极相互作用强度。 

当谐振器与量子比特远离共振，即 $\Delta \gg g$（$\Delta = \omega_r-\omega_q$ 为失谐，例如 $\omega_r \gg \omega_q$）时，系统处于色散区。

在色散区，系统可由如下有效哈密顿量表示：

$\displaystyle H = \omega_r a^\dagger a - \frac{1}{2}\omega_q \sigma_z + \chi (a^\dagger a  + 1/2) \sigma_z$

其中 $\chi = g^2/\Delta$。最后一项可视为“依赖量子比特态的谐振器频率修正”，或等价地“依赖谐振器态的量子比特频率修正”。

在 D. I. Schuster 等人的经典实验中，研究者通过监测与谐振器耦合的量子比特，成功分辨了微波谐振器中的光子数态。本笔记展示如何在 QuTiP 中数值模拟这类系统。

### 参考文献

 * [D. I. Schuster et al., Resolving photon number states in a superconducting circuit, Nature 445, 515 (2007)](http://dx.doi.org/10.1038/nature05461)


## 参数

```python
N = 20

wr = 2.0 * 2 * np.pi  # resonator frequency
wq = 3.0 * 2 * np.pi  # qubit frequency
chi = 0.025 * 2 * np.pi  # parameter in the dispersive hamiltonian

delta = abs(wr - wq)  # detuning
g = np.sqrt(delta * chi)  # coupling strength that is consistent with chi
```

```python
# compare detuning and g, the first should be much larger than the second
delta / (2 * np.pi), g / (2 * np.pi)
```

```python
# cavity operators
a = tensor(destroy(N), qeye(2))
nc = a.dag() * a
xc = a + a.dag()

# atomic operators
sm = tensor(qeye(N), destroy(2))
sz = tensor(qeye(N), sigmaz())
sx = tensor(qeye(N), sigmax())
nq = sm.dag() * sm
xq = sm + sm.dag()

Id = tensor(qeye(N), qeye(2))
```

```python
# dispersive hamiltonian
H = wr * (a.dag() * a + Id / 2.0) + (wq / 2.0) * sz + chi * \
    (a.dag() * a + Id / 2) * sz
```

尝试改变谐振器初态，观察下文频谱如何反映这里设定的光子分布。

```python
# psi0 = tensor(coherent(N, sqrt(6)), (basis(2,0)+basis(2,1)).unit())
```

```python
# psi0 = tensor(thermal_dm(N, 3), ket2dm(basis(2,0)+basis(2,1))).unit()
```

```python
psi0 = tensor(coherent(N, np.sqrt(4)), (basis(2, 0) + basis(2, 1)).unit())
```

## 时间演化

```python
tlist = np.linspace(0, 250, 1000)
```

```python
res = mesolve(H, psi0, tlist, [], options={'nsteps': 5000})
```

### 激发数

可以看到两者几乎不交换能量，因为它们彼此远失谐。

```python
nc_list = expect(nc, res.states)
nq_list = expect(nq, res.states)
```

```python
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4))

ax.plot(tlist, nc_list, "r", linewidth=2, label="cavity")
ax.plot(tlist, nq_list, "b--", linewidth=2, label="qubit")
ax.set_ylim(0, 7)
ax.set_ylabel("n", fontsize=16)
ax.set_xlabel("Time (ns)", fontsize=16)
ax.legend()

fig.tight_layout()
```

### 谐振器正交分量


不过，谐振器正交分量会快速振荡。

```python
xc_list = expect(xc, res.states)
```

```python
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4))

ax.plot(tlist, xc_list, "r", linewidth=2, label="cavity")
ax.set_ylabel("x", fontsize=16)
ax.set_xlabel("Time (ns)", fontsize=16)
ax.legend()

fig.tight_layout()
```

### 谐振器关联函数

```python
tlist = np.linspace(0, 100, 1000)
```

```python
corr_vec = correlation_2op_1t(H, psi0, tlist, [], a.dag(), a)
```

```python
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4))

ax.plot(tlist, np.real(corr_vec), "r", linewidth=2, label="resonator")
ax.set_ylabel("correlation", fontsize=16)
ax.set_xlabel("Time (ns)", fontsize=16)
ax.legend()
ax.set_xlim(0, 50)
fig.tight_layout()
```

### 谐振器频谱

```python
w, S = spectrum_correlation_fft(tlist, corr_vec)
```

```python
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(w / (2 * np.pi), abs(S))
ax.set_xlabel(r"$\omega$", fontsize=18)
ax.set_xlim(wr / (2 * np.pi) - 0.5, wr / (2 * np.pi) + 0.5);
```

这里可以看到：由于量子比特处于 0/1 叠加态，谐振器峰发生上下分裂与偏移。还可验证分裂正好为预期的 $2\chi$：

```python
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot((w - wr) / chi, abs(S))
ax.set_xlabel(r"$(\omega-\omega_r)/\chi$", fontsize=18)
ax.set_xlim(-2, 2);
```

### 量子比特关联函数

```python
corr_vec = correlation_2op_1t(H, psi0, tlist, [], sx, sx)
```

```python
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(12, 4))

ax.plot(tlist, np.real(corr_vec), "r", linewidth=2, label="qubit")
ax.set_ylabel("correlation", fontsize=16)
ax.set_xlabel("Time (ns)", fontsize=16)
ax.legend()
ax.set_xlim(0, 50)
fig.tight_layout()
```

### 量子比特频谱

量子比特频谱具有很有趣的结构：可直接读出谐振器模中的光子分布。

```python
w, S = spectrum_correlation_fft(tlist, corr_vec)
```

```python
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(w / (2 * np.pi), abs(S))
ax.set_xlabel(r"$\omega$", fontsize=18)
```

把频谱平移并按 $2\chi$ 缩放后更清晰：

```python
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot((w - wq - chi) / (2 * chi), abs(S))
ax.set_xlabel(r"$(\omega - \omega_q - \chi)/2\chi$", fontsize=18)
ax.set_xlim(-0.5, N);
```

与腔的 Fock 态分布对比：

```python
rho_cavity = ptrace(res.states[-1], 0)
```

```python
fig, axes = plt.subplots(1, 1, figsize=(9, 3))

axes.bar(np.arange(0, N) - 0.4, np.real(rho_cavity.diag()), color="blue",
         alpha=0.6)
axes.set_ylim(0, 1)
axes.set_xlim(-0.5, N)
axes.set_xticks(np.arange(0, N))
axes.set_xlabel("Fock number", fontsize=12)
axes.set_ylabel("Occupation probability", fontsize=12);
```

再看腔模 Wigner 函数可知：与量子比特发生色散相互作用后，腔模不再是单一相干态，而成为相干态叠加。

```python
fig, axes = plt.subplots(1, 1, figsize=(6, 6))

xvec = np.linspace(-5, 5, 200)
W = wigner(rho_cavity, xvec, xvec)
wlim = abs(W).max()

axes.contourf(
    xvec,
    xvec,
    W,
    100,
    norm=mpl.colors.Normalize(-wlim, wlim),
    cmap=plt.get_cmap("RdBu"),
)
axes.set_xlabel(r"Im $\alpha$", fontsize=18)
axes.set_ylabel(r"Re $\alpha$", fontsize=18);
```

### 软件版本

```python
about()
```
