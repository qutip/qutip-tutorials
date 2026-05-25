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

# 讲座 16 - Wigner 函数图集


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

本系列讲座由 J.R. Johansson 开发，原始讲义 notebook 可在[这里](https://github.com/jrjohansson/qutip-lectures)查看。

这里是为适配当前 QuTiP 版本而稍作修改的版本。
你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲座。
本讲座及其他教程 notebook 的索引页见 [QuTiP Tutorial 网页](https://qutip.org/tutorials.html)。

```python
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from qutip import (about, basis, coherent, coherent_dm, displace, fock, ket2dm,
                   plot_wigner, squeeze, thermal_dm)

%matplotlib inline
```

## 简介


## 参数

```python
N = 20
```

```python
def plot_wigner_2d_3d(psi):
    xvec = np.linspace(-6, 6, 200)
    yvec = np.linspace(-6, 6, 200)

    fig = plt.figure(figsize=(17, 8))

    ax = fig.add_subplot(1, 2, 1)
    plot_wigner(psi, xvec=xvec, yvec=yvec, fig=fig, ax=ax)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    plot_wigner(psi, xvec=xvec, yvec=yvec, projection="3d", fig=fig, ax=ax)

    plt.close(fig)
    return fig
```

## 真空态：$\left|0\right>$

```python
psi = basis(N, 0)
plot_wigner_2d_3d(psi)
```

## 热态

```python
psi = thermal_dm(N, 2)
plot_wigner_2d_3d(psi)
```

## 相干态：$\left|\alpha\right>$

```python
psi = coherent(N, 2.0)
plot_wigner_2d_3d(psi)
```

```python
psi = coherent(N, -1.0)
plot_wigner_2d_3d(psi)
```

## 相干态叠加

```python
psi = (coherent(N, -2.0) + coherent(N, 2.0)) / np.sqrt(2)
plot_wigner_2d_3d(psi)
```

```python
psi = (coherent(N, -2.0) - coherent(N, 2.0)) / np.sqrt(2)
plot_wigner_2d_3d(psi)
```

```python
psi = (coherent(N, -2.0) + coherent(N, -2j) + coherent(N, 2j)
       + coherent(N, 2.0)).unit()
plot_wigner_2d_3d(psi)
```

```python
psi = (coherent(N, -2.0) + coherent(N, -1j) + coherent(N, 1j)
       + coherent(N, 2.0)).unit()
plot_wigner_2d_3d(psi)
```

```python
NN = 8

fig, axes = plt.subplots(NN, 1, figsize=(5, 5 * NN),
                         sharex=True, sharey=True)
for n in range(NN):
    psi = sum(
        [coherent(N, 2 * np.exp(2j * np.pi * m / (n + 2)))
         for m in range(n + 2)]
    ).unit()
    plot_wigner(psi, fig=fig, ax=axes[n])

    # if n < NN - 1:
    #    axes[n].set_ylabel("")
```

### 相干态混合

```python
psi = (coherent_dm(N, -2.0) + coherent_dm(N, 2.0)) / np.sqrt(2)
plot_wigner_2d_3d(psi)
```

## Fock 态：$\left|n\right>$

```python
for n in range(6):
    psi = basis(N, n)
    display(plot_wigner_2d_3d(psi))
```

## Fock 态叠加

```python
NN = MM = 5

fig, axes = plt.subplots(NN, MM, figsize=(18, 18),
                         sharex=True, sharey=True)
for n in range(NN):
    for m in range(MM):
        psi = (fock(N, n) + fock(N, m)).unit()
        plot_wigner(psi, fig=fig, ax=axes[n, m])
        if n < NN - 1:
            axes[n, m].set_xlabel("")
        if m > 0:
            axes[n, m].set_ylabel("")
```

## 压缩真空态

```python
psi = squeeze(N, 0.5) * basis(N, 0)
display(plot_wigner_2d_3d(psi))

psi = squeeze(N, 0.75j) * basis(N, 0)
display(plot_wigner_2d_3d(psi))

psi = squeeze(N, -1) * basis(N, 0)
display(plot_wigner_2d_3d(psi))
```

### 压缩真空叠加态

```python
psi = (squeeze(N, 0.75j) * basis(N, 0) - squeeze(N, -0.75j)
       * basis(N, 0)).unit()
display(plot_wigner_2d_3d(psi))
```

### 压缩真空混合态

```python
psi = (
    ket2dm(squeeze(N, 0.75j) * basis(N, 0)) +
    ket2dm(squeeze(N, -0.75j) * basis(N, 0))
).unit()
display(plot_wigner_2d_3d(psi))
```

## 位移压缩真空态

```python
psi = displace(N, 2) * squeeze(N, 0.75) * basis(N, 0)
display(plot_wigner_2d_3d(psi))
```

### 两个位移压缩态叠加

```python
psi = (
    displace(N, -1) * squeeze(N, 0.75) * basis(N, 0)
    - displace(N, 1) * squeeze(N, -0.75) * basis(N, 0)
).unit()
display(plot_wigner_2d_3d(psi))
```

## 版本信息

```python
about()
```
