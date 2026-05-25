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

# 第 14 讲 - Kerr 非线性

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

这套讲义由 J.R. Johansson 开发。原始讲义 notebook 在[这里](https://github.com/jrjohansson/qutip-lectures)。

当前版本在原讲义基础上做了小幅修改，以适配 QuTiP 的当前发布版。你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲义。本讲及其他教程 notebook 可在 [QuTiP 教程页面](https://qutip.org/tutorials.html)查看索引。

```python
from base64 import b64encode

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from qutip import (about, coherent, destroy, expect, isket, ket2dm, mesolve,
                   num, variance, wigner)

%matplotlib inline
```

## 引言

Kerr 效应描述了非线性介质中电磁量子场的自相互作用。单模量子场可由如下有效哈密顿量描述：

$\displaystyle H = \frac{1}{2}\chi (a^\dagger)^2a^2$

其中 $\chi$ 与三阶非线性极化率相关。Kerr 效应是量子光学中典型的非线性来源之一。

本笔记将展示如何在 QuTiP 中搭建该模型，并观察该哈密顿量下演化态的一些有趣性质。


## 参数

```python
N = 15
chi = 1 * 2 * np.pi  # Kerr-nonlinearity
tlist = np.linspace(0, 1.0, 51)  # time
```

```python
# operators: the annihilation operator of the field
a = destroy(N)

# and we'll also need the following operators in calculation of
# expectation values when visualizing the dynamics
n = num(N)
x = a + a.dag()
p = -1j * (a - a.dag())
```

```python
# the Kerr Hamiltonian
H = 0.5 * chi * a.dag() * a.dag() * a * a
```

## 绘图函数

先定义一些可视化动力学的函数，后续会用到。

```python
def plot_expect_with_variance(N, op_list, op_title, states):
    """
    Plot the expectation value of an operator (list of operators)
    with an envelope that describes the operators variance.
    """

    fig, axes = plt.subplots(1, len(op_list), figsize=(14, 3))

    for idx, op in enumerate(op_list):

        e_op = expect(op, states)
        v_op = variance(op, states)

        axes[idx].fill_between(
            tlist, e_op - np.sqrt(v_op), e_op + np.sqrt(v_op), color="green",
            alpha=0.5
        )
        axes[idx].plot(tlist, e_op)
        axes[idx].set_xlabel("Time")
        axes[idx].set_title(op_title[idx])
        axes[idx].set_xlim(0, max(tlist))

    return fig, axes
```

```python
def plot_wigner(rho, fig=None, ax=None):
    """
    Plot the Wigner function and the Fock state distribution given a density
    matrix for a harmonic oscillator mode.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    if isket(rho):
        rho = ket2dm(rho)

    xvec = np.linspace(-7.5, 7.5, 200)

    W = wigner(rho, xvec, xvec)
    wlim = abs(W).max()

    ax.contourf(
        xvec,
        xvec,
        W,
        100,
        norm=mpl.colors.Normalize(-wlim, wlim),
        cmap=mpl.colormaps["RdBu"],
    )
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)

    return fig, ax
```

```python
def plot_fock_distribution_vs_time(tlist, states, fig=None, ax=None):

    Z = np.zeros((len(tlist), states[0].shape[0]))

    for state_idx, state in enumerate(states):
        Z[state_idx, :] = np.real(ket2dm(state).diag())

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    Y, X = np.meshgrid(tlist, range(states[0].shape[0]))
    p = ax.pcolor(
        X,
        Y,
        Z.T,
        norm=mpl.colors.Normalize(0, 0.5),
        cmap=mpl.colormaps["RdBu"],
        edgecolors="k",
    )
    ax.set_xlabel(r"$N$", fontsize=16)
    ax.set_ylabel(r"$t$", fontsize=16)

    cb = fig.colorbar(p)
    cb.set_label("Probability")

    return fig, ax
```

```python
def display_embedded_video(filename):
    video = open(filename, "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = '<video controls alt="test" \
                src="data:video/x-m4v;base64,{0}">'.format(
        video_encoded
    )
    return HTML(video_tag)
```

## 相干态

下面看相干态在 Kerr 哈密顿量作用下如何演化。

```python
# we start with a coherent state with alpha=2.0
psi0 = coherent(N, 2.0)
```

```python
# and evolve the state under the influence of the hamiltonian.
# by passing an empty list as expecation value operators argument,
# we get the full state of the system in result.states
result = mesolve(H, psi0, tlist, [])
```

先看光子数算符 $n$ 与 $x,p$ 两个正交分量的期望值和方差如何随时间变化：

```python
plot_expect_with_variance(N, [n, x, p], [r"n", r"x", r"p"], result.states);
```

注意平均光子数 $\langle n \rangle$ 及其方差保持常数，这暗示 Fock 分布保持不变。由上图还能看出，演化过程中 $x$ 与 $p$ 的方差会随时间变化。

为验证光子分布确实与时间无关，可将 Fock 分布画成时间函数：

```python
plot_fock_distribution_vs_time(tlist, result.states);
```

因此 Fock 分布恒定，但状态的 Wigner 函数仍会随时间变化。为了更直观看到 Wigner 函数动力学，我们做一个短动画，展示从 $t=0$ 到演化末时刻的变化。

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 8))


def update(n):
    plot_wigner(result.states[n], fig=fig, ax=ax)
    return ax.artists


anim = animation.FuncAnimation(fig, update, frames=len(result.states),
                               blit=True)

anim.save("animation-kerr-coherent-state.mp4", fps=10, writer="ffmpeg")

plt.close(fig)
```

```python
display_embedded_video("animation-kerr-coherent-state.mp4")
```

可以看到一个有趣现象：动力学是周期性的，而且这里恰好演化了一个完整周期，所以末态与初态相同。

在中间时刻会出现很有意思的结构。例如在半个周期后，状态看起来非常像相干态叠加形成的猫态。

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 8))


def update(n):
    plot_wigner(result.states[n], fig=fig, ax=ax)
    return ax.artists


anim = animation.FuncAnimation(
    fig, update, frames=int(len(result.states) / 2 + 1), blit=True
)

anim.save("animation-kerr-coherent-state-half-period.mp4",
          fps=10, writer="ffmpeg")

plt.close(fig)
```

```python
display_embedded_video("animation-kerr-coherent-state-half-period.mp4")
```

确实如此：相干态 $|\alpha\rangle$ 在满足 $\chi t = \pi$ 的时刻演化后，会得到猫态

$\psi = \frac{1}{\sqrt{2}}\left(e^{i\pi/4}|-i\alpha\rangle + e^{-i\pi/4}|i\alpha\rangle\right)$

（见 Walls and Milburn, Quantum Optics, p91）

```python
psi = (
    np.exp(1j * np.pi / 4) * coherent(N, -2.0j)
    + np.exp(-1j * np.pi / 4) * coherent(N, 2.0j)
).unit()
```

```python
plot_wigner(psi);
```

### 软件版本

```python
about()
```

### 致谢


感谢 Sander Konijnenberg 指出猫态解析表达式中的一个错误。
