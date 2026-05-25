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

# 讲座 9 - 量子谐振子的压缩态

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

本系列讲座由 J.R. Johansson 开发，原始讲义 notebook 可在[这里](https://github.com/jrjohansson/qutip-lectures)查看。

这里是为适配当前 QuTiP 版本而稍作修改的版本。
你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲座。
本讲座及其他教程 notebook 的索引页见 [QuTiP Tutorial 网页](https://qutip.org/tutorials.html)。

```python
from base64 import b64encode

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from qutip import (about, basis, coherent, destroy, displace, expect, mesolve,
                   num, plot_fock_distribution, plot_wigner, squeeze, variance)

%matplotlib inline
```

## 简介

在量子力学中，每次对可观测量（对应厄米算符）的测量结果都是随机的，
并遵循某种概率分布。
算符期望值是大量测量结果的平均值，
而标准差刻画测量结果的不确定度。

这种不确定性是量子力学的内禀属性，无法被彻底消除。
海森堡不确定关系给出了不对易算符对的不确定性下界。
例如 $x$ 与 $p$ 满足对易关系 $[x, p] = i\hbar$，
因此总有 $(\Delta x) (\Delta p) >= \hbar/2$。

若某个态满足

$(\Delta x) (\Delta p) = \hbar/2$

称为最小不确定态。
若某态满足例如

$(\Delta x)^2 < \hbar/2$

则称其为压缩态。
此时为了满足海森堡关系，
$(\Delta p)^2$ 必须相应大于 $\hbar/2(\Delta x)^2$。
也就是说，降低某个算符（如 $x$）的方差到最小不确定极限以下时，
必然会放大与其不对易算符（如 $p$）的方差。

对谐振模式而言，压缩 $x$ 或 $p$ 通常称为“正交分量压缩”，
也是最常见的压缩类型。

本 QuTiP notebook 中，我们将观察在不同压缩初态下，
单模谐振子的正交分量算符 $x$、$p$ 的期望值与方差如何随时间演化。


## 参数

```python
N = 35
w = 1 * 2 * np.pi  # oscillator frequency
tlist = np.linspace(0, 2, 31)  # periods
```

```python
# operators
a = destroy(N)
n = num(N)
x = (a + a.dag()) / np.sqrt(2)
p = -1j * (a - a.dag()) / np.sqrt(2)
```

```python
# the quantum harmonic oscillator Hamiltonian
H = w * a.dag() * a
```

```python
c_ops = []

# uncomment to see how things change when disspation is included
# c_ops = [np.sqrt(0.25) * a]
```

## 绘图函数

因为后续会对多种不同初态重复同类计算与可视化，
这里先定义几个可复用函数。

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
            alpha=0.5)
        axes[idx].plot(tlist, e_op, label="expectation")
        axes[idx].set_xlabel("Time")
        axes[idx].set_title(op_title[idx])

    return fig, axes
```

```python
def display_embedded_video(filename):
    video = open(filename, "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = '<video controls alt="test"                 src="data:video/x-m4v;base64,{0}">'.format(
        video_encoded
    )
    return HTML(video_tag)
```

## 相干态

作为参考，先看相干态的时间演化。

```python
psi0 = coherent(N, 2.0)
```

```python
result = mesolve(H, psi0, tlist, c_ops)
```

```python
plot_expect_with_variance(N, [n, x, p], [r"$n$", r"$x$", r"$p$"],
                          result.states);
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))


def update(n):
    axes[0].cla()
    plot_fock_distribution(result.states[n], fig=fig, ax=axes[0])
    plot_wigner(result.states[n], fig=fig, ax=axes[1])
    return axes[0].artists + axes[1].artists


anim = animation.FuncAnimation(fig, update, frames=len(result.states),
                               blit=True)

anim.save("/tmp/animation-coherent-state.mp4", fps=10, writer="ffmpeg")

plt.close(fig)
```

```python
display_embedded_video("/tmp/animation-coherent-state.mp4")
```

## 压缩真空态

```python
psi0 = squeeze(N, 1.0) * basis(N, 0)
```

```python
result = mesolve(H, psi0, tlist, c_ops)
```

```python
plot_expect_with_variance(N, [n, x, p], [r"$n$", r"$x$", r"$p$"],
                          result.states);
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))


def update(n):
    axes[0].cla()
    plot_fock_distribution(result.states[n], fig=fig, ax=axes[0])
    plot_wigner(result.states[n], fig=fig, ax=axes[1])
    return axes[0].artists + axes[1].artists


anim = animation.FuncAnimation(fig, update, frames=len(result.states),
                               blit=True)

anim.save("/tmp/animation-squeezed-vacuum.mp4", fps=10, writer="ffmpeg")

plt.close(fig)
```

```python
display_embedded_video("/tmp/animation-squeezed-vacuum.mp4")
```

## 压缩相干态

```python
psi0 = (
    displace(N, 2) * squeeze(N, 1.0) * basis(N, 0)
)  # first squeeze vacuum and then displace
```

```python
result = mesolve(H, psi0, tlist, c_ops)
```

```python
plot_expect_with_variance(N, [n, x, p], [r"$n$", r"$x$", r"$p$"],
                          result.states);
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))


def update(n):
    axes[0].cla()
    plot_fock_distribution(result.states[n], fig=fig, ax=axes[0])
    plot_wigner(result.states[n], fig=fig, ax=axes[1])
    return axes[0].artists + axes[1].artists


anim = animation.FuncAnimation(fig, update, frames=len(result.states),
                               blit=True)

anim.save("/tmp/animation-squeezed-coherent-state.mp4", fps=10,
          writer="ffmpeg")

plt.close(fig)
```

```python
display_embedded_video("/tmp/animation-squeezed-coherent-state.mp4")
```

### 软件版本

```python
about()
```
