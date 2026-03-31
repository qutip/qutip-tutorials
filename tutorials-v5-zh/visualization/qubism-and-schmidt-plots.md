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

# Qubism 可视化


by [Piotr Migdal](http://migdal.wikidot.com/), June 2014

QuTiP 更多信息见 http://qutip.org。

Qubism 参考资料：
* J. Rodriguez-Laguna, P. Migdal, M. Ibanez Berganza, M. Lewenstein, G. Sierra,
  [Qubism: self-similar visualization of many-body wavefunctions](http://dx.doi.org/10.1088/1367-2630/14/5/053028), New J. Phys. 14 053028 (2012), [arXiv:1112.3560](http://arxiv.org/abs/1112.3560),
* [视频摘要](https://www.youtube.com/watch?v=8fPAzOziTZo),
* [GitHub 上的 C++ 与 Mathematica 代码](https://github.com/stared/qubism)。

本文介绍 `plot_schmidt` 与 `plot_qubism` 两个绘图函数，以及 `complex_array_to_rgb`，并展示其应用。



```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (Qobj, about, complex_array_to_rgb, jmat, ket, plot_qubism,
                   plot_schmidt, qeye, sigmax, sigmay, sigmaz, tensor)

%matplotlib inline
```

## 颜色


在量子力学中，复数与实数一样自然。

在介绍具体绘图前，先看 `complex_array_to_rgb` 如何把 $z = x + i y$ 映射到颜色。
有两种主题：`theme='light'` 与 `theme='dark'`。两者都用色相表示相位：正数偏红，负数偏青。

关于复函数着色的更深入讨论，可参考 Emilia Petrisor 的 IPython Notebook：
[Visualizing complex-valued functions with Matplotlib and Mayavi](http://nbviewer.jupyter.org/github/empet/Math/blob/master/DomainColoring.ipynb)。

```python
compl_circ = np.array(
    [
        [(x + 1j * y) if x ** 2 + y**2 <= 1 else 0j
            for x in np.arange(-1, 1, 0.005)]
        for y in np.arange(-1, 1, 0.005)
    ]
)

fig = plt.figure(figsize=(6, 3))
for i, theme in enumerate(["light", "dark"]):
    ax = plt.subplot(1, 2, i + 1)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.imshow(
        complex_array_to_rgb(compl_circ, rmax=1, theme=theme),
        extent=(-1, 1, -1, 1)
    )
plt.tight_layout()
```

## Schmidt 图


展示纠缠最直观的方法之一，是把波函数画成两个变量的函数。
如果图像可分离成两个变量的乘积，则态是直积态；否则就是纠缠态。

将波函数写为矩阵 $|\psi\rangle_{ij}$ 正是 [Schmidt 分解](http://en.wikipedia.org/wiki/Schmidt_decomposition) 的关键步骤，
因此这种图称为 Schmidt 图。

考虑两种态：

* 纠缠态：单态 $|\psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}$；
* 直积态：$(|01\rangle - |00\rangle)/\sqrt{2}$。

它们看起来可能相似，但后者可分解为 $|0\rangle(|1\rangle - |0\rangle)/\sqrt{2}$。

```python
singlet = (ket("01") - ket("10")).unit()
separable = (ket("01") - ket("00")).unit()
```

```python
fig = plt.figure(figsize=(2, 2))
plot_schmidt(singlet, fig=fig);
```

```python
fig = plt.figure(figsize=(2, 2))
plot_schmidt(separable, fig=fig);
```

可见：对可分离态，图像是 `x` 与 `y` 坐标的乘积；对 singlet 态则不是。

再看两个 singlet 的直积：$|\psi^-\rangle|\psi^-\rangle$。
`plot_schmidt` 默认按粒子数均分划分系统。

（为了颜色更明显，这里额外乘上虚数单位。）

```python
fig = plt.figure(figsize=(2, 2))
plot_schmidt(1j * tensor([singlet, singlet]), fig=fig);
```

可见它仍是乘积结构，因为相对于“前 2 个粒子 vs 后 2 个粒子”的划分，该态是直积态。

如果把粒子重排成 $|\psi^-\rangle_{23}|\psi^-\rangle_{41}$ 呢？

```python
fig = plt.figure(figsize=(2, 2))
plot_schmidt(1j * tensor([singlet, singlet]).permute([1, 2, 3, 0]),
             fig=fig);
```

这时就出现纠缠。

`plot_schmidt` 允许指定其他划分。参数 `splitting` 决定列方向包含多少粒子。
一般地，它可处理粒子数不同、每个粒子维度也可能不同的系统。

例如：

```python
fig = plt.figure(figsize=(4, 2))
plot_schmidt(
    1j * tensor([singlet, singlet]),
    splitting=1,
    labels_iteration=(1, 3),
    fig=fig
);
```

## Qubism 图

```python
fig = plt.figure(figsize=(8, 4))
for i in [1, 2]:
    ax = plt.subplot(1, 2, i)
    plot_qubism(0 * ket("0000"), legend_iteration=i, grid_iteration=i,
                fig=fig, ax=ax)
```

也就是说，以以下前缀开头的振幅会映射到：

* $|00\rangle$：左上象限；
* $|01\rangle$：右上象限；
* $|10\rangle$：左下象限；
* $|11\rangle$：右下象限。

然后对后续粒子递归应用同样规则。例如：

```python
state = (
    ket("0010")
    + 0.5 * ket("1111")
    + 0.5j * ket("0101")
    - 1j * ket("1101")
    - 0.2 * ket("0110")
)
fig = plt.figure(figsize=(4, 4))
plot_qubism(state, fig=fig);
```

若要明确查看振幅到图中区域的映射：

```python
fig = plt.figure(figsize=(4, 4))
plot_qubism(state, legend_iteration=2, fig=fig);
```

也可切换深色主题（例如配黑底幻灯片）：

```python
fig = plt.figure(figsize=(4, 4))
plot_qubism(state, legend_iteration=2, theme="dark", fig=fig);
```

Qubism 最重要的性质是递归结构，可平滑扩展到更多粒子。
例如画 `k` 个 singlet 的张量积：$|\psi^-\rangle^{\otimes k}$：

```python
fig = plt.figure(figsize=(15, 3))
for k in range(1, 6):
    ax = plt.subplot(1, 5, k)
    plot_qubism(tensor([singlet] * k), fig=fig, ax=ax)
```

当然，如果波函数可手写，图像本身价值有限。
下面看如何可视化基态。
在此之前先定义一些函数，便于构造平移不变哈密顿量。

```python
def spinchainize(op, n, bc="periodic"):

    if isinstance(op, list):
        return sum([spinchainize(each, n, bc=bc) for each in op])

    k = len(op.dims[0])
    d = op.dims[0][0]

    expanded = tensor([op] + [qeye(d)] * (n - k))

    if bc == "periodic":
        shifts = n
    elif bc == "open":
        shifts = n - k + 1

    shifteds = [
        expanded.permute([(i + j) % n for i in range(n)])
        for j in range(shifts)
    ]

    return sum(shifteds)


def gs_of(ham):
    gval, gstate = ham.groundstate()
    return gstate
```

例如考虑 $N$ 个粒子的如下哈密顿量（[Majumdar-Ghosh 模型](http://en.wikipedia.org/wiki/Majumdar%E2%80%93Ghosh_Model) 的推广）：

$$H = \sum_{i=1}^N \vec{S}_i \cdot \vec{S}_{i+1} + J \sum_{i=1}^N \vec{S}_i \cdot \vec{S}_{i+2},$$

其中 $\vec{S}_i = \tfrac{1}{2} (\sigma^x, \sigma^y, \sigma^z)$ 是自旋算符（$\sigma$ 为 [Pauli 矩阵](http://en.wikipedia.org/wiki/Pauli_matrices)）。

还可选两种边界条件：

* 周期边界（periodic）：自旋链成环（$N+1 \equiv 1$ 且 $N+2 \equiv 2$）；
* 开边界（open）：自旋链成线（去掉涉及 $N+1$ 与 $N+2$ 的项）。

```python
heis = sum([tensor([pauli] * 2) for pauli in [sigmax(), sigmay(), sigmaz()]])
heis2 = sum(
    [tensor([pauli, qeye(2), pauli])
     for pauli in [sigmax(), sigmay(), sigmaz()]]
)

N = 10
Js = [0.0, 0.5, 1.0]

fig = plt.figure(figsize=(2 * len(Js), 4.4))

for b in [0, 1]:
    for k, J in enumerate(Js):
        ax = plt.subplot(2, len(Js), b * len(Js) + k + 1)

        if b == 0:
            spinchain = spinchainize([heis, J * heis2], N, bc="periodic")
        elif b == 1:
            spinchain = spinchainize([heis, J * heis2], N, bc="open")

        plot_qubism(gs_of(spinchain), ax=ax)

        if k == 0:
            if b == 0:
                ax.set_ylabel("periodic BC", fontsize=16)
            else:
                ax.set_ylabel("open BC", fontsize=16)
        if b == 1:
            ax.set_xlabel("$J={0:.1f}$".format(J), fontsize=16)

plt.tight_layout()
```

Qubism 不局限于 qubit，也支持其他维度（如 qutrit）。

考虑自旋-1 粒子的 [AKLT 模型](http://en.wikipedia.org/wiki/AKLT_Model)：

$$H = \sum_{i=1}^N \vec{S}_i \cdot \vec{S}_{i+1} + \tfrac{1}{3} \sum_{i=1}^N (\vec{S}_i \cdot \vec{S}_{i+1})^2.$$

其中 $\vec{S}_i$ 是自旋-1 粒子的 [自旋算符](http://en.wikipedia.org/wiki/Pauli_matrices#Physics)（在 `qutip` 中对应 `jmat(1, 'x')`、`jmat(1, 'y')`、`jmat(1, 'z')`）。

```python
ss = sum([tensor([jmat(1, s)] * 2) for s in ["x", "y", "z"]])
H = spinchainize([ss, (1.0 / 3.0) * ss**2], n=6, bc="periodic")
fig = plt.figure(figsize=(4, 4))
plot_qubism(gs_of(H), fig=fig);
```

qutrit 的 Qubism 用法与 qubit 类似：

```python
fig = plt.figure(figsize=(10, 5))
for i in [1, 2]:
    ax = plt.subplot(1, 2, i)
    plot_qubism(
        0 * ket("0000", dim=3), legend_iteration=i, grid_iteration=i,
        fig=fig, ax=ax
    )
```

此时可解释为：

* 0 对应 $s_z=-1$；
* 1 对应 $s_z=\ \ 0$；
* 2 对应 $s_z=+1$。

虽然 Qubism 最适合平移不变态（尤其要求粒子维度一致），也可用于更一般系统。

另外还有其他相关绘图方案，如 `how='pairs_skewed'`：

```python
fig = plt.figure(figsize=(8, 4))
for i in [1, 2]:
    ax = plt.subplot(1, 2, i)
    plot_qubism(
        0 * ket("0000"),
        how="pairs_skewed",
        legend_iteration=i,
        grid_iteration=i,
        fig=fig,
        ax=ax,
    )
```

该模式更强调铁磁（左侧）与反铁磁（右侧）态。

另一个方案 `how='before_after'`（受[该图](http://commons.wikimedia.org/wiki/File:Ising-tartan.png) 启发）略有不同：
它仍用递归，但从中间粒子开始。例如左上象限对应 $|00\rangle_{N/2,N/2+1}$：

```python
fig = plt.figure(figsize=(8, 4))
for i in [1, 2]:
    ax = plt.subplot(1, 2, i)
    plot_qubism(
        0 * ket("0000"),
        how="before_after",
        legend_iteration=i,
        grid_iteration=i,
        fig=fig,
        ax=ax,
    )
```

它与 Schmidt 图（默认划分）非常接近，唯一区别是 `y` 轴排序（粒子顺序反转）。纠缠性质不变。

那它在同一示例上效果如何？
这里取自旋链（$J=0$ 的 Majumdar-Ghosh，即
$$H = \sum_{i=1}^N \vec{S}_i \cdot \vec{S}_{i+1}$$
用于 qubit）：

```python
heis = sum([tensor([pauli] * 2) for pauli in [sigmax(), sigmay(), sigmaz()]])
N = 10
gs = gs_of(spinchainize(heis, N, bc="periodic"))

fig = plt.figure(figsize=(12, 4))
for i, how in enumerate(["schmidt_plot", "pairs",
                         "pairs_skewed", "before_after"]):
    ax = plt.subplot(1, 4, i + 1)
    if how == "schmidt_plot":
        plot_schmidt(gs, fig=fig, ax=ax)
    else:
        plot_qubism(gs, how=how, fig=fig, ax=ax)
    ax.set_title(how)
plt.tight_layout()
```

## 观察纠缠

```python
product_1 = ket("0000")
product_2 = tensor([(ket("0") + ket("1")).unit()] * 4)
w = (ket("0001") + ket("0010") + ket("0100") + ket("1000")).unit()
dicke_2of4 = (
    ket("0011") + ket("0101") + ket("0110") +
    ket("1001") + ket("1010") + ket("1100")
).unit()
ghz = (ket("0000") + ket("1111")).unit()
```

```python
states = ["product_1", "product_2", "w", "dicke_2of4", "ghz"]
fig = plt.figure(figsize=(2 * len(states), 2))
for i, state_str in enumerate(states):
    ax = plt.subplot(1, len(states), i + 1)
    plot_qubism(eval(state_str), fig=fig, ax=ax)
    ax.set_title(state_str)
plt.tight_layout()
```

对给定划分下，纠缠（更准确地说 Schmidt 秩）等于“不同且非零方块”的个数。
（不允许旋转，但允许整体因子缩放，且允许线性叠加。）

这里划分为前 2 个粒子 vs 后 2 个粒子（由分隔线表示）。

即：
* `product_1`：1 个非零方块，Schmidt 秩 1；
* `product_2`：4 个非零方块，但相同，Schmidt 秩 1；
* `w`：3 个非零方块，其中两个相同，Schmidt 秩 2；
* `dicke_2of4`：4 个非零方块，其中两个相同，Schmidt 秩 3；
* `ghz`：2 个非零方块且互不相同，Schmidt 秩 2。

这些性质与基无关，但在某些基下更容易观察。

作为对比，再看如下直积态：

$$\left( \cos(\theta/2) |0\rangle + \sin(\theta/2) e^{i \varphi} |1\rangle \right)^N $$

```python
def product_state(theta, phi=0, n=1):
    single = Qobj([[np.cos(theta / 2.0)],
                   [np.sin(theta / 2.0) * np.exp(1j * phi)]])
    return tensor([single] * n)


thetas = 0.5 * np.pi * np.array([0.0, 0.5, 0.75, 1.0])
phis = np.pi * np.array([0.0, 0.1, 0.2, 0.3])

fig, axes2d = plt.subplots(nrows=len(phis),
                           ncols=len(thetas), figsize=(6, 6))

for i, row in enumerate(axes2d):
    for j, cell in enumerate(row):
        plot_qubism(
            product_state(thetas[j], phi=phis[i], n=8),
            grid_iteration=1, ax=cell
        )
        if i == len(axes2d) - 1:
            cell.set_xlabel(
                r"$\theta={0:s}\pi$".format(
                    ["0", "(1/4)", "(3/8)", "(1/2)"][j]),
                fontsize=16,
            )
        if j == 0:
            cell.set_ylabel(
                r"$\varphi={0:.1f}\pi$".format(phis[i] / np.pi), fontsize=16
            )

plt.tight_layout()
```

在每张图中，各方块都只差一个因子（由亮度和色调体现）。

你也可以回看前面的图。`grid_iteration=2` 会显示前 4 个粒子 vs 其余粒子的划分。
对 `how='before_after'`，则是中间粒子 vs 其余粒子。


### 版本

```python
about()
```
