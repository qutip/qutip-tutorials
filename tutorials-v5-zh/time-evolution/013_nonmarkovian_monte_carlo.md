---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 非马尔可夫蒙特卡洛求解器：两个物理示例

作者：B. Donvil 与 P. Menczel，2023

### 引言

本教程讨论两个由时间局域非马尔可夫主方程描述的系统，即形式类似 Lindblad、但“速率”可能为负的主方程。
我们将说明这类主方程在物理场景中的来源，以及如何使用 QuTiP 的非马尔可夫蒙特卡洛求解器进行模拟。
该求解器基于 influence martingale（影响鞅）形式，详见参考文献 [\[1, 2\]](#References)。
示例取自参考文献 [\[1\]](#References)：一个处于光子带隙中的二能级原子（基于 [\[3\]](#References)），以及两个互不相互作用但共同耦合到同一环境的量子比特 Redfield 主方程。

量子蒙特卡洛方法的一个优势是模拟易于并行化。
QuTiP 可以与 `mpi4py` 交互，从而利用高性能计算集群的大规模并行能力。
作为*示例 2*的一部分，我们将在[文末](#Monte-Carlo-Simulations-on-Computing-Clusters-via-MPI)演示如何在 MPI 环境下运行该示例。

```python
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

import numpy as np
import qutip as qt

from scipy import special, optimize
from scipy.interpolate import CubicSpline

from collections import Counter
import os
```

<!-- #region -->
### 示例 1：光子带隙中的二能级原子

这里讨论基于参考文献 [\[3\]](#References) 的光子带隙二能级原子主方程。
描述二能级系统与三维周期电介质中辐射场的总哈密顿量为

$$ H = \sum_\lambda \hbar \omega_\lambda a^\dagger_\lambda a_\lambda + \omega \sigma_z + i\hbar \sum_\lambda g_\lambda (a_\lambda^\dagger\sigma_- - \sigma_+ a_\lambda) $$

其中 $a_\lambda, a_\lambda^\dagger$ 是标准升降算符，$\sigma_\pm=(\sigma_x\pm i \sigma_y)/2$，$\sigma_{x,y,z}$ 为 Pauli 矩阵。
求和遍历电磁场模式与偏振 $\lambda=(\vec k,\sigma)$，$\omega$（以及 $\omega_\lambda$）分别表示原子（模式 $\lambda$）的特征频率。
最后，$g_\lambda \propto \cos(\theta)/\omega_\lambda^{1/2}$ 是原子-场耦合常数，其中 $\theta$ 为模式偏振向量与原子偶极矩之间夹角。

我们作如下假设：
- 光子场初始处于真空态。
- 在带隙附近 $k\simeq k_0$，色散关系满足 $\omega_\lambda=\omega_c+A(k-k_0)^2$，其中 $A=\omega_c/k_0^2$ 且 $\omega_c=ck_0$。

在这些条件下，全系统动力学可以解析求解。
时刻 $t$ 的总态写作

$$ |\psi(t)\rangle = e^{-i\omega t} c(t) |1,\{0\}\rangle + \sum_\lambda c_\lambda(t) |0,\{\lambda\}\rangle e^{-i\omega_\lambda t}$$

其中 $|1,\{0\}\rangle$ 表示二能级系统处于激发态、辐射场为真空；$|0,\{\lambda\}\rangle$ 表示二能级系统处于基态且模式 $\lambda$ 被激发。
函数 $c_\lambda(t)$ 定义为
$$c_\lambda(t) = g_\lambda \int_0^t c(\tau) e^{i(\omega_\lambda-\omega)\tau} d\tau$$

而 $c(t)$ 的复杂表达式见下方代码（参考文献 [\[3\]](#References) 式 (2.21)）。
它依赖两个参数

$$ \begin{aligned}
\delta &= \omega-\omega_c \quad \text{and} \\
\beta^{3/2} &= \frac{\omega^2 \omega_c^{3/2} d^2 }{6\pi \epsilon_0 \hbar c^3} ,
\end{aligned} $$
其中 $d$ 是原子偶极矩的绝对值。

最后，可将场模精确迹化 [\[4\]](#References)，得到主方程

$$ \frac{d}{dt} \rho(t) = -2i S(t) [\sigma_+\sigma_-,\rho(t)] + \Gamma(t) \left(\sigma_-\rho(t)\sigma_+ -\frac{1}{2}\{\sigma_+\sigma_-,\rho(t)\}\right) , $$

这里 $\rho(t)$ 是原子的约化密度矩阵。
其中

$$ \begin{aligned}
S(t) &= -2 \operatorname{Im} \frac{\dot{c}(t)}{c(t)} \quad \text{and} \\
\Gamma(t) &= -2 \operatorname{Re} \frac{\dot{c}(t)}{c(t)} .
\end{aligned} $$



##### 系统设置

我们选择模型参数 $\delta$ 与 $\beta$：
<!-- #endregion -->

```python
delta = -1
beta = 1
```

上文引入的若干派生参数与函数 $c(t)$：

```python
Ap = (1 / 2 + 1 / 2 * (1 + 4 / 27 * (delta / beta) ** 3) ** (1 / 2)) ** (1 / 3)
Am = (1 / 2 - 1 / 2 * (1 + 4 / 27 * (delta / beta) ** 3) ** (1 / 2)) ** (1 / 3)
x1 = (Ap + Am) * np.exp(1j * np.pi / 4)
x2 = ((Ap * np.exp(-1j * np.pi / 6) - Am * np.exp(1j * np.pi / 6))
      * np.exp(-1j * np.pi / 4))
x3 = ((Ap * np.exp(1j * np.pi / 6) - Am * np.exp(-1j * np.pi / 6))
      * np.exp(3 * 1j * np.pi / 4))
a1 = x1 / ((x1 - x2) * (x1 - x3))
a2 = x2 / ((x2 - x1) * (x2 - x3))
a3 = x3 / ((x3 - x2) * (x3 - x1))
y1 = (x1**2) ** (1 / 2)
y2 = (x2**2) ** (1 / 2)
y3 = (x3**2) ** (1 / 2)


def c(t):
    return (
        2 * x1 * a1 * np.exp((beta * x1**2 + 1j * delta) * t)
        + a2 * (x2 + y2) * np.exp((beta * x2**2 + 1j * delta) * t)
        - (a1 * y1 * (1 - special.erf((beta * x1**2 * t) ** (1 / 2)))
           * np.exp((beta * x1**2 + 1j * delta) * t))
        - (a2 * y2 * (1 - special.erf((beta * x2**2 * t) ** (1 / 2)))
           * np.exp((beta * x2**2 + 1j * delta) * t))
        - (a3 * y3 * (1 - special.erf((beta * x3**2 * t) ** (1 / 2)))
           * np.exp((beta * x3**2 + 1j * delta) * t))
    )


def cd(t):  # time derivative
    return (
        ((beta * x1**2 + 1j * delta) * 2 * x1 * a1
         * np.exp((beta * x1**2 + 1j * delta) * t))
        + ((beta * x2**2 + 1j * delta) * a2 * (x2 + y2)
           * np.exp((beta * x2**2 + 1j * delta) * t))
        - ((beta * x1**2 + 1j * delta) * a1 * y1
           * (1 - special.erf((beta * x1**2 * t) ** (1 / 2)))
           * np.exp((beta * x1**2 + 1j * delta) * t))
        - ((beta * x2**2 + 1j * delta) * a2 * y2
           * (1 - special.erf((beta * x2**2 * t) ** (1 / 2)))
           * np.exp((beta * x2**2 + 1j * delta) * t))
        - ((beta * x3**2 + 1j * delta) * a3 * y3
           * (1 - special.erf((beta * x3**2 * t) ** (1 / 2)))
           * np.exp((beta * x3**2 + 1j * delta) * t))
        + (a1 * y1 * x1 * np.exp(-beta * t * x1**2)
           * (beta / (t * np.pi + 0.00001)) ** (1 / 2)
           * np.exp((beta * x1**2 + 1j * delta) * t))
        + (a2 * y2 * x2 * np.exp(-beta * t * x2**2)
           * (beta / (t * np.pi + 0.00001)) ** (1 / 2)
           * np.exp((beta * x2**2 + 1j * delta) * t))
        + (a3 * y3 * x3 * np.exp(-beta * t * x3**2)
           * (beta / (t * np.pi + 0.00001)) ** (1 / 2)
           * np.exp((beta * x3**2 + 1j * delta) * t))
    )


def S(t):
    return -2 * np.imag(cd(t) / c(t))


def Gamma(t):
    return -2 * np.real(cd(t) / c(t))
```

定义时间区间。初始时刻做了平移，以避免在 $t=0$ 处出现 $\Gamma(t)$ 的负值（该点不具物理意义）：

```python
ti = optimize.root_scalar(Gamma, bracket=(1.4, 1.5)).root
duration = 10
steps = 100

times = np.linspace(ti, ti + duration, steps + 1)
```

绘制时间区间内的 $\Gamma(t)$ 与 $S(t)$，可以看到 $\Gamma(t)$ 会变为负值：

```python
Gamma_values = Gamma(times)
S_values = S(times)

plt.plot(times - ti, Gamma_values, label=r"$\Gamma(t)$")
plt.plot(times - ti, S_values, label=r"$S(t)$")
plt.xlabel(r"$t$")
plt.legend()
plt.show()
```

为将后续计算加速约 3-4 倍，我们将 $\Gamma(t)$ 与 $S(t)$ 预存为插值函数。

```python
Gamma_int = CubicSpline(times, np.complex128(Gamma_values))
S_int = CubicSpline(times, np.complex128(S_values))
```

##### 蒙特卡洛模拟

我们先设置数值参数（轨迹数、是否并行计算），再定义对应主方程的哈密顿量、跳跃算符和初态。

```python
nmmc_options = {"map": "parallel",
                "norm_steps": 10}  # options specific to nm_mcsolve
options = {"progress_bar": "enhanced"}  # options shared by all solvers
ntraj = 5500

H = 2 * qt.sigmap() * qt.sigmam()
ops_and_rates = [[qt.sigmam(), qt.coefficient(Gamma_int)]]
psi0 = qt.basis(2, 0)
e_ops = [H]
```

与 `mcsolve` 及其 `MCSolver` 类似，我们既可实例化 `NonMarkovianMCSolver` 并调用其 `run` 或 `start`/`step` 方法，也可使用 `nm_mcsolve` 便捷函数。
这里我们显式构建 `NonMarkovianMCSolver`，以验证求解器会自动确保跳跃算符满足完备关系 $\sum_n A_n^\dagger A_n \propto 1$。
可以看到，为满足该关系，系统会额外加入第二个跳跃算符。

```python
solver = qt.NonMarkovianMCSolver(qt.QobjEvo([H, S_int]),
                                 ops_and_rates,
                                 options=(options | nmmc_options))

for L in solver.ops:
    print(L, "\n")

completeness_check = sum(L.dag() * L for L in solver.ops)
with qt.CoreOptions(atol=1e-5):
    assert completeness_check == qt.qeye(2)
```

最后，运行蒙特卡洛模拟：

```python
MCSol = solver.run(psi0, tlist=times, ntraj=ntraj, e_ops=e_ops)
```

##### 与非随机模拟对比

我们将结果与下面精确的 `mesolve` 模拟进行比较：

```python
d_ops = [[qt.lindblad_dissipator(qt.sigmam(), qt.sigmam()), Gamma]]
MESol = qt.mesolve([H, S], psi0, times, d_ops, e_ops=e_ops, options=options)
```

##### 结果

通过两个求解器的 `e_ops` 参数，我们分别得到精确模拟的期望值 $\langle H\rangle$，以及蒙特卡洛模拟中 `ntraj` 条轨迹的平均值。下图同时展示精确解、蒙特卡洛估计以及其误差估计
$$ \textrm{Error}_{\text{MC}} = \sigma / \sqrt{N} $$
其中 $\sigma$ 是各轨迹返回值的标准差（由 QuTiP 自动计算），$N$ 是轨迹数量。

在 influence martingale 方法下，非马尔可夫主方程的每条轨迹并不保持迹守恒。事实上，轨迹上态的迹正是影响鞅。由于其鞅性质，足够多轨迹平均后，迹的期望应为 1。QuTiP 会自动保存平均迹及其标准差；我们在图中读出并展示，平均迹偏离 1 的程度可反映蒙特卡洛收敛质量。

注意此处 `ntraj` 设为 `2500`，以控制笔记本运行时间。将其提高到 `10000` 或更高会改善收敛。

```python
plt.plot(times - ti, MESol.expect[0] / 2, "k-", label="Exact")
plt.plot(times - ti, MCSol.expect[0] / 2, "kx", label="Monte-Carlo")
plt.fill_between(
    times - ti,
    (MCSol.expect[0] - MCSol.std_expect[0] / np.sqrt(ntraj)) / 2,
    (MCSol.expect[0] + MCSol.std_expect[0] / np.sqrt(ntraj)) / 2,
    alpha=0.5,
)

plt.plot(times - ti, np.ones_like(times), "-", color="0.5")
plt.plot(times - ti, MCSol.trace, "x", color="0.5")
plt.fill_between(
    times - ti,
    MCSol.trace - MCSol.std_trace / np.sqrt(ntraj),
    MCSol.trace + MCSol.std_trace / np.sqrt(ntraj),
    alpha=0.5,
)

plt.xlabel(r"$t$")
plt.ylabel(r"$\langle H \rangle\, /\, 2$")
plt.legend()
plt.show()
```

##### 改进采样
作为本示例结尾，我们简要演示 `nm_mcsolve` 中“改进采样（improved sampling）”选项的用法。
为清晰起见，我们仍用同一示例，但缩短时间区间并减少轨迹数量：

```python
times_is = np.linspace(ti, ti + duration / 2, steps + 1)
ntraj_is = int(ntraj / 10)

MCSol_is = solver.run(psi0, tlist=times_is, ntraj=ntraj_is, e_ops=e_ops)
MESol_is = qt.mesolve([H, S], psi0, times_is, d_ops, e_ops, options=options)
```

若统计每条轨迹的塌缩次数，可见约 10% 的轨迹没有发生塌缩：

```python
print(Counter([len(x) for x in MCSol_is.col_times]))
```

这些轨迹彼此完全相同。
因此这类轨迹理论上只需计算一次。
可通过启用“improved sampling”实现：

```python
solver.options = {'improved_sampling': True}
MCSol_is_improved = solver.run(psi0, tlist=times_is, ntraj=ntraj_is, e_ops=e_ops)
print(Counter([len(x) for x in MCSol_is_improved.col_times]))
```

```python
plt.plot(times_is - ti, MESol_is.expect[0] / 2, "k-", label="Exact")
plt.plot(times_is - ti, MCSol_is.expect[0] / 2, "kx", label="MC")
plt.plot(times_is - ti, MCSol_is_improved.expect[0] / 2, "rx", label="MC (improved)")

plt.plot(times_is - ti, np.ones_like(times_is), "k-", alpha=0.5)
plt.plot(times_is - ti, MCSol_is.trace, "kx", alpha=0.5)
plt.plot(times_is - ti, MCSol_is_improved.trace, "rx", alpha=0.5)

plt.xlabel(r"$t$")
plt.ylabel(r"$\langle H \rangle\, /\, 2$")
plt.legend()
plt.show()
```

<!-- #region -->
### 示例 2：双量子比特 Redfield 方程

我们考虑两个量子比特：它们共同耦合到同一个玻色浴，但彼此之间无直接相互作用。
完整模型哈密顿量为

$$ H = \omega_1 \sigma^{(1)}_+\sigma^{(1)}_- + \omega_2 \sigma^{(2)}_+\sigma^{(2)}_- + \sum_k \epsilon_k b_k^\dagger b_k +\sum_k g_k [b_k(\sigma^{(1)}_++\sigma^{(2)}_+)+b_k^\dagger(\sigma^{(1)}_-+\sigma^{(2)}_-)]$$

这里 $b_k, b_k^\dagger$ 为玻色升降算符，$\sigma_\pm^{(j)}=(\sigma_x^{(j)}\pm i\sigma_y^{(j)})/2$，$\sigma_{x,y,z}^{(j)}$ 为作用在第 $j$ 个量子比特上的 Pauli 矩阵。
此外，$\omega_j$ 与 $\epsilon_k$ 分别表示量子比特与浴模的特征频率，$g_k$ 为耦合常数。
总系统初态取为 $\rho_{2q}\otimes |0\rangle\langle 0|$，其中 $\rho_{2q}$ 是双比特初态，$|0\rangle\langle 0|$ 是玻色浴真空态。

按参考文献 [\[5\]](#References) 进行 Born-Markov 近似，可得到量子比特的 Redfield 主方程：
$$ \quad \frac{d}{dt}\rho(t) = -i \sum_{i,j=1}^2 A_{i,j} [\sigma_+^{(j)}\sigma_-^{(i)},\rho(t)] + \sum_{i,j=1}^2B_{i,j}\left( \sigma_-^{(i)}\rho(t)\sigma_+^{(j)} -\frac{1}{2}\{\sigma_+^{(j)}\sigma_-^{(i)},\rho(t)\}\right) . \tag 1 $$
这里引入矩阵
$$ \begin{aligned}
    A &= \begin{pmatrix} \omega_1 + \alpha & \alpha + \frac{\kappa}{2} - i\frac{\gamma_1-\gamma_2}{8} \\ \alpha + \frac{\kappa}{2} - i\frac{\gamma_2-\gamma_1}{8} & \omega_2 + \alpha + \kappa \end{pmatrix} \quad \text{and} \\
    B &= \frac{1}{2}\begin{pmatrix} \gamma_1 & \frac{\gamma_1+\gamma_2}{2} - 2i\kappa \\ \frac{\gamma_1+\gamma_2}{2} + 2i\kappa & \gamma_2 \end{pmatrix} ,
\end{aligned} $$
其中参数 $\alpha$、$\kappa$ 与 $\gamma_{1,2}\ge 0$ 与完整哈密顿量的关系如下：
$$ \begin{aligned}
    \gamma_j &= 2\int_{-\infty}^\infty dt\, \sum_k g_k^2\, e^{i(\omega_j - \epsilon_k)t} , \\
    \alpha &= \operatorname{Im} \int_0^\infty dt\, \sum_k g_k^2\, e^{i(\omega_1 - \epsilon_k)t} , \\
    \kappa &= \operatorname{Im} \int_0^\infty dt\, \sum_k g_k^2\, \Bigl( e^{i(\omega_2 - \epsilon_k)t} - e^{i(\omega_1 - \epsilon_k)t} \Bigr) .
\end{aligned} $$

矩阵 $B$ 是自伴的，因此可由某个酉矩阵 $U$ 对角化：
$$ U^\dagger\, B\, U = \begin{pmatrix} \lambda_{1}&0\\0&\lambda_{2} \end{pmatrix} $$
其中 $\lambda_i = \frac{\gamma_1+\gamma_2}{4}+(-1)^i\sqrt{\frac{\gamma_1^2+\gamma_2^2+8\,\kappa^2}{8}}$。
定义 Lindblad 算符
$$ L_j = \sum_{i=1}^2 \sigma_-^{(i)} U_{ij} $$
则方程 (1) 可写为
$$ \frac{d}{dt}\rho(t) = -i \sum_{i,j=1}^2 A_{i,j} [\sigma_+^{(j)}\sigma_-^{(i)},\rho(t)] + \sum_{i=1}^2\lambda_i\left( L_i\rho(t)L_i^\dagger -\frac{1}{2}\{L_i^\dagger L_i,\rho(t)\}\right) . \tag 2 $$
由此可见，当 $\kappa>0$ 时有 $\lambda_1<0$，因此式 (2) 不再是严格 Lindblad 形式。



##### 系统设置

我们选择模型参数 $\alpha$、$\kappa$、$\gamma_{1,2}$，并计算速率 $\lambda_1$ 与 $\lambda_2$。
<!-- #endregion -->

```python
omeg1 = 0.25
omeg2 = 0.5

gam1 = 1
gam2 = 4
alpha = 3
kappa = 1

lamb1 = (gam1 + gam2) / 4 - np.sqrt((gam1**2 + gam2**2 + 8 * kappa**2) / 8)
lamb2 = (gam1 + gam2) / 4 + np.sqrt((gam1**2 + gam2**2 + 8 * kappa**2) / 8)
```

我们选择时间区间与数值参数。

```python
times2 = np.linspace(0, 2.5, 100)
nmmc_options = {"map": "parallel",
                "keep_runs_results": True,
                "norm_steps": 10}  # options specific to nm_mcsolve
options = {"progress_bar": "enhanced"}  # options shared by all solvers
ntraj = 5000
```

接下来在满足 $L_1^\dagger|0\rangle=|1\rangle$、$L_2^\dagger|0\rangle=|2\rangle$、$L_1^\dagger L_2^\dagger|0\rangle=|3\rangle$ 的基底中定义系统。注意定义哈密顿量时需先旋转到该基底。初态取 $|\psi_0\rangle=\sqrt{0.4}|0\rangle+\sqrt{0.4}|1\rangle+\sqrt{0.2}|2\rangle$。

```python
# Initial state of the system
psi0 = (
    np.sqrt(0.4) * qt.basis(4, 0)
    + np.sqrt(0.4) * qt.basis(4, 1)
    + np.sqrt(0.2) * qt.basis(4, 2)
)

L1 = qt.Qobj(np.array([[0, 0, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 1, 0]])).dag()
L2 = qt.Qobj(np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0]])).dag()

norm1 = np.sqrt(
    1 + (gam2 - gam1 + np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2))**2
    / ((gam1 + gam2) ** 2 + 16 * kappa**2))
norm2 = np.sqrt(
    1 + (gam2 - gam1 - np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2))**2
    / ((gam1 + gam2) ** 2 + 16 * kappa**2))
U = qt.Qobj(np.array([
    [(gam1 - gam2 - np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2))
     / (gam1 + gam2 + 4j*kappa) / norm1,
     (gam1 - gam2 + np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2))
     / (gam1 + gam2 + 4j*kappa) / norm2],
    [1 / norm1, 1 / norm2]]))

# Write \sigma_\pm^{1,2} in terms of L_{1,2} using the explicit definition of U
Udag = U.dag()
sigmam1 = Udag[0, 0] * L1 + Udag[1, 0] * L2
sigmam2 = Udag[0, 1] * L1 + Udag[1, 1] * L2
sigmap1 = sigmam1.dag()
sigmap2 = sigmam2.dag()

H = ((omeg1 + alpha) * sigmap1 * sigmam1
     + (omeg2 + alpha + kappa) * sigmap2 * sigmam2
     + (alpha + kappa / 2 - 1j * (gam1 - gam2) / 8) * sigmap2 * sigmam1
     + (alpha + kappa / 2 - 1j * (gam2 - gam1) / 8) * sigmap1 * sigmam2)
```

##### 蒙特卡洛模拟

```python
eops = [qt.basis(4, 0) * qt.basis(4, 0).dag(),
        qt.basis(4, 1) * qt.basis(4, 1).dag(),
        qt.basis(4, 2) * qt.basis(4, 2).dag()]

MCSol2 = qt.nm_mcsolve(H, psi0, times2, ntraj=ntraj,
                       options=(options | nmmc_options),
                       ops_and_rates=[[L1, lamb1], [L2, lamb2]],
                       e_ops=eops)
```

##### 与非随机模拟对比

```python
d_ops = [lamb1 * qt.lindblad_dissipator(L1, L1),
         lamb2 * qt.lindblad_dissipator(L2, L2)]
MESol2 = qt.mesolve(H, psi0, times2, d_ops,
                    e_ops=eops,
                    options=options)
```

##### 结果

先给出与示例 1 类似的图：展示基态与 $|1\rangle$、$|2\rangle$ 态占据数（在本初态下 $|3\rangle$ 占据始终为零）。实线与叉号分别对应主方程精确解与蒙特卡洛平均，阴影区域表示各量的蒙特卡洛误差 $\text{Error}_\text{MC}$。可以看出该方法在短时与中等时间表现更好，而长时间误差似乎指数增长，后文将分析原因。

```python
plt.plot(times2, MESol2.expect[0], color='C0',
         label=r"$\langle e_0 \mid \rho \mid e_0 \rangle$")
plt.plot(times2, MCSol2.average_expect[0], 'x', color='C0')
plt.fill_between(
    times2,
    MCSol2.average_expect[0] - MCSol2.std_expect[0] / np.sqrt(ntraj),
    MCSol2.average_expect[0] + MCSol2.std_expect[0] / np.sqrt(ntraj),
    alpha=0.2, color='C0'
)

plt.plot(times2, MESol2.expect[1], color='C1',
         label=r"$\langle e_1 \mid \rho \mid e_1 \rangle$")
plt.plot(times2, MCSol2.average_expect[1], 'x', color='C1')
plt.fill_between(
    times2,
    MCSol2.average_expect[1] - MCSol2.std_expect[1] / np.sqrt(ntraj),
    MCSol2.average_expect[1] + MCSol2.std_expect[1] / np.sqrt(ntraj),
    alpha=0.2, color='C1'
)

plt.plot(times2, MESol2.expect[2], color='C2',
         label=r"$\langle e_2 \mid \rho \mid e_2 \rangle$")
plt.plot(times2, MCSol2.average_expect[2], 'x', color='C2')
plt.fill_between(
    times2,
    MCSol2.average_expect[2] - MCSol2.std_expect[2] / np.sqrt(ntraj),
    MCSol2.average_expect[2] + MCSol2.std_expect[2] / np.sqrt(ntraj),
    alpha=0.2, color='C2'
)

plt.plot(times2, np.ones_like(times2), color="0.5",
         label=r"$\operatorname{tr} \rho$")
plt.plot(times2, MCSol2.average_trace, "x", color="0.5")
plt.fill_between(
    times2,
    MCSol2.average_trace - MCSol2.std_trace / np.sqrt(ntraj),
    MCSol2.average_trace + MCSol2.std_trace / np.sqrt(ntraj),
    alpha=0.2, color="0.5"
)

plt.xlabel(r"$t$")
plt.ylabel('Expectation values')
plt.legend()

plt.show()
```

当用 `nm_mcsolve` 计算如 $H$ 的期望值时，求解器会先在每条轨迹上计算该量，再用对应影响鞅 $\mu$ 对轨迹值加权平均。下图展示若干单轨迹上的该量演化（图中未包含鞅权重，因此纵轴标为 $\langle H\rangle/\mu$）。线条阴影表示鞅大小；红色实线是精确 $\langle H\rangle$，也即在足够多轨迹下按权重 $\mu$ 对 $(\langle H\rangle/\mu)$ 加权平均的结果。

图中可见，真正对平均有贡献的轨迹数会快速减少，原因有二：其一，$L_1$ 或 $L_2$ 跳跃会把系统带入基态，而进入基态后轨迹不会再离开；其二，求解器会添加第三个“零跳跃率”通道，一旦发生该通道跳跃，对应轨迹在后续时间中的鞅会被置零。下图通过让轨迹在区间结束前停止来可视化这一点。

若 $\langle H\rangle$ 最终趋于零，贡献轨迹减少并非大问题；但由于存在负速率，系综平均并不收敛到 $|0\rangle\langle0|$。因此剩余有效轨迹的权重（由阴影表示）必须迅速增大以补偿无贡献轨迹，从而因这类轨迹采样不足导致误差快速放大。

注意图中轨迹做了轻微偏移，否则会彼此重叠。

```python
# Adapted from matplotlib tutorial,
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

# --- Settings ---
# Lines are plotted w/ offset, spread is difference between max and min offset
spread = 0.015
N = 150  # Number of trajectories
# ----------------

all_traces = [np.abs(tr) for traj in MCSol2.trajectories[0:N]
              for tr in traj.trace if np.abs(tr) > 0]
min_martingale = min(all_traces)
max_martingale = max(all_traces)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
norm = LogNorm(min_martingale, max_martingale)

for i, traj in enumerate(MCSol2.trajectories[0:N]):
    offset = -spread / 2 + spread * i / N

    traj_defined = np.abs(traj.trace) > 0
    traj_times = times2[traj_defined]
    traj_trace = np.array(traj.trace)[traj_defined]
    traj_ex0 = traj.expect[0][traj_defined] / traj_trace
    traj_ex1 = traj.expect[1][traj_defined] / traj_trace
    traj_ex2 = traj.expect[2][traj_defined] / traj_trace
    traj_exH = omeg1 * traj_ex1 + omeg2 * traj_ex2

    points = np.array([traj_times, traj_exH + offset]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = np.abs(traj_trace[:-1])

    lc = LineCollection(segments, cmap='PuBu', norm=norm)
    lc.set_array(colors)
    line = ax.add_collection(lc)

ax.plot(times2, omeg1 * MESol2.expect[1] + omeg2 * MESol2.expect[2],
        color='red', label=r"$\langle H \rangle$ (exact)")
ax.plot(times2,
        omeg1 * MCSol2.average_expect[1] + omeg2 * MCSol2.average_expect[2],
        color='blue', label=r"$\langle H \rangle$ (MC)")
ax.legend()

fig.colorbar(line, ax=ax, label=r"Martingale $\mu$")
ax.set_xlabel(r"$t$")
ax.set_xlim(times2[0], times2[-1])
ax.set_ylabel(r"$\langle H \rangle\, /\, \mu$")
ax.set_ylim(-0.02, 0.22)
plt.show()
```

<!-- #region -->
### 通过 MPI 在计算集群上进行蒙特卡洛模拟

为改善上例在长时间下的收敛行为，需要更多轨迹。
我们借此演示 QuTiP 的 MPI 能力。
在 QuTiP 侧，只需把 `map` 选项从 `parallel` 改成 `mpi`。
随后 QuTiP 会调用 `mpi4py` 提供的 `MPIPoolExecutor` 并行模拟轨迹。
这类计算通常无法直接在 Jupyter 笔记本内部启动。
下面代码给出了一个可提交到超级计算机作业调度器的独立脚本示例。

```python
# Beginning of `qutip-mpi-example.py` file
import numpy as np
import qutip as qt

# --- SETTINGS ---
# Maximum number of MPI worker processes that can be used
NUM_WORKER_PROCESSES = 500
# Create batches averaged over this number of trajectories
BATCH_SIZE = 1000
# Create this number of batches
# (total number of trajectories is BATCH_SIZE * NUM_BATCHES)
NUM_BATCHES = 500


def setup_system():
    omeg1, omeg2 = 0.25, 0.5
    gam1, gam2, alpha, kappa = 1, 4, 3, 1
    lamb1 = (gam1 + gam2) / 4 - np.sqrt((gam1**2 + gam2**2 + 8 * kappa**2) / 8)
    lamb2 = (gam1 + gam2) / 4 + np.sqrt((gam1**2 + gam2**2 + 8 * kappa**2) / 8)

    L1 = qt.Qobj(np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])).dag()
    L2 = qt.Qobj(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])).dag()
    norm1 = np.sqrt(1 + (gam2 - gam1 + np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2))**2 / ((gam1 + gam2) ** 2 + 16 * kappa**2))
    norm2 = np.sqrt(1 + (gam2 - gam1 - np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2))**2 / ((gam1 + gam2) ** 2 + 16 * kappa**2))
    Udag = qt.Qobj(np.array([
        [(gam1 - gam2 - np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2)) / (gam1 + gam2 + 4j*kappa) / norm1,
         (gam1 - gam2 + np.sqrt(2) * np.sqrt(gam1**2 + gam2**2 + 8*kappa**2)) / (gam1 + gam2 + 4j*kappa) / norm2],
        [1 / norm1,
         1 / norm2]
    ])).dag()
    sigmam1 = Udag[0, 0] * L1 + Udag[1, 0] * L2
    sigmam2 = Udag[0, 1] * L1 + Udag[1, 1] * L2
    sigmap1 = sigmam1.dag()
    sigmap2 = sigmam2.dag()

    H = ((omeg1 + alpha) * sigmap1 * sigmam1
         + (omeg2 + alpha + kappa) * sigmap2 * sigmam2
         + (alpha + kappa / 2 - 1j * (gam1 - gam2) / 8) * sigmap2 * sigmam1
         + (alpha + kappa / 2 - 1j * (gam2 - gam1) / 8) * sigmap1 * sigmam2)
    psi0 = np.sqrt(0.4) * qt.basis(4, 0) + np.sqrt(0.4) * qt.basis(4, 1) + np.sqrt(0.2) * qt.basis(4, 2)
    ops_and_rates = [[L1, lamb1], [L2, lamb2]]

    return H, psi0, ops_and_rates


def main():
    times = np.linspace(0, 10, 250)
    H, psi0, ops_and_rates = setup_system()

    for i in range(NUM_BATCHES):
        result = qt.nm_mcsolve(H, psi0, times, ops_and_rates, ntraj=BATCH_SIZE,
                               options={'store_states': True,
                                        'progress_bar': False,
                                        'map': 'mpi',
                                        'num_cpus': NUM_WORKER_PROCESSES,
                                        'norm_steps': 10}
                              )
        qt.qsave(result, f"./result-{i}")


if __name__ == "__main__":
    main()
```
<!-- #endregion -->

<!-- #region -->
如何在实际环境运行该脚本取决于你的计算基础设施。
可参考 [mpi4py.futures 用户指南](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html)。
本教程作者在超级计算机 [HOKUSAI](https://www.r-ccs.riken.jp/exhibit_contents/SC20/hokusai.html) 上，使用 MPI 标准的 [MPICH](https://www.mpich.org) 实现，并通过下方批处理脚本提交到 SLURM。
在该超算上，使用 501 个 CPU 核心（1 个管理进程 + 500 个工作进程）跨 5 个节点运行，任务约 50 分钟完成，共生成 50 万条轨迹。

```bash
#!/bin/bash
#SBATCH --partition=XXXXX
#SBATCH --account=XXXXX

#SBATCH --nodes=5
#SBATCH --ntasks=501
#SBATCH --mem-per-cpu=1G

#SBATCH --time=0-10:00

source ~/.bashrc

module purge
module load mpi/mpich-x86_64
conda activate qutip-environment

mpirun -np $SLURM_NTASKS -bind-to core python -m mpi4py.futures qutip-mpi-example.py
```

上述代码会生成若干结果文件。
请将这些文件复制到 `mpi_results` 文件夹，然后执行后续代码。
我们先像前面一样绘制平均结果。


```python
# copied from script above
BATCH_SIZE = 1000
times3 = np.linspace(0, 10, 250)

# load all available result files
results_folder_exists = os.path.isdir('mpi_results')
if results_folder_exists:
    batches = []
    while True:
        i = len(batches)
        filename = f'mpi_results/result-{i}'
        if not os.path.exists(f'{filename}.qu'):
            break
        batches.append(qt.qload(filename))
    NUM_BATCHES = len(batches)

# combine result files
if results_folder_exists and NUM_BATCHES > 0:
    combined_result = sum(batches[1:], start=batches[0])
    exact_solution = qt.mesolve(H, psi0, times3, d_ops, options=options)
```

```python
if results_folder_exists and NUM_BATCHES > 0:
    plt.plot(times3, qt.expect(exact_solution.states, eops[0]),
             color='C0', label=r"$\langle e_0 \mid \rho \mid e_0 \rangle$")
    plt.plot(times3, qt.expect(combined_result.states, eops[0]),
             'x', color='C0')

    plt.plot(times3,  qt.expect(exact_solution.states, eops[1]),
             color='C1', label=r"$\langle e_1 \mid \rho \mid e_1 \rangle$")
    plt.plot(times3, qt.expect(combined_result.states, eops[1]),
             'x', color='C1')

    plt.plot(times3, qt.expect(exact_solution.states, eops[2]),
             color='C2', label=r"$\langle e_2 \mid \rho \mid e_2 \rangle$")
    plt.plot(times3, qt.expect(combined_result.states, eops[2]),
             'x', color='C2')

    plt.xlabel(r"$t$")
    plt.xlim((0, 5))
    plt.ylabel('Expectation values')
    plt.ylim((-.1, .7))
    plt.legend()

    plt.show()

else:
    print('No result files found.')
```

接下来进行如下分析，方法采自参考文献 [\[6\]](#References)。

设 $k\le N$，其中 $N$ 为总轨迹数。
设 $I=[0,10]$ 为总时间区间，并定义
$$ I_k = \{ t \in I : \lVert \rho_{\text{MC},k} - \rho_{\text{exact}} \rVert \leq 0.1 \} . $$
其中，$\rho_{\text{MC},k}$ 是仅使用 $k$ 条轨迹得到的估计态。
我们绘制 $\mu(I_k)$ 随 $k$ 的变化关系，其中 $\mu(I_k)$ 是集合 $I_k$ 的测度。

```python
if results_folder_exists and NUM_BATCHES > 0:
    result = {}
    progress = qt.ui.TqdmProgressBar(NUM_BATCHES)

    for k, batch in enumerate(batches):
        # average of the batches up to index k
        if k == 0:
            average = batch
        else:
            average = average + batch

        # how many points are close to the exact solution
        diff = [(mc - exact).norm()
                for mc, exact in zip(average.states, exact_solution.states)]
        good_points = np.count_nonzero(  # count 'True' entries
            np.isclose(diff, np.zeros_like(diff), rtol=0, atol=0.1)
        )

        num_trajectories = (k + 1) * BATCH_SIZE
        result.update({num_trajectories: good_points / len(times3)})
        progress.update()

    progress.finished()
    xval = np.array(list(result.keys()))
    yval = np.array(list(result.values()))

    fit = np.polyfit(np.log(xval), yval, 1)
    print(('Approximate number of trajectories required for convergence until '
           'time t (according to linear fit):\n'
           f'N = {np.exp(-fit[1] / fit[0]):.2f} * '
           f'exp( {1 / fit[0] / times3[-1]:.2f} * t )\n'))

    plt.semilogx(xval, yval, label='Simulation result')
    plt.semilogx(xval, fit[0] * np.log(xval) + fit[1], '--', label='Fit')
    plt.xlabel('Number of trajectories, $k$')
    plt.ylabel(r'$\mu(I_k)$')
    plt.legend()
    plt.show()

else:
    print('No result files found.')
```

### 参考文献

\[1\] [Donvil and Muratore-Ginanneschi, Nat Commun (2022)](https://www.nature.com/articles/s41467-022-31533-8).  
\[2\] [Donvil and Muratore-Ginanneschi, arXiv:2209.08958 \[quant-ph\]](https://arxiv.org/abs/2209.08958).  
\[3\] [John and Quang, Phys. Rev. A (1994)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.50.1764).  
\[4\] Breuer and Petruccione, *The Theory of Open Quantum Systems*.  
\[5\] [Mozgunov and Lidar, Quantum (2020)](https://quantum-journal.org/papers/q-2020-02-06-227/).  
\[6\] [Menczel *et al.*, arXiv:2401.11830 \[quant-ph\]](https://arxiv.org/abs/2401.11830).


### 关于

```python
qt.about()
```

### 测试

```python
# -- first example --

assert np.any(np.array([Gamma(t) for t in times]) < 0)

np.testing.assert_array_less(MESol.expect[0][1:] / 2, 1)
np.testing.assert_array_less(0, MESol.expect[0] / 2)

np.testing.assert_allclose(MCSol.trace, 1, atol=0, rtol=0.25)
np.testing.assert_allclose(MCSol.expect[0], MESol.expect[0], atol=0, rtol=0.25)
```

```python
# -- second example --

MAX_TIME = 1

mc_ex0 = MCSol2.average_expect[0][times2 <= 1]
me_ex0 = MESol2.expect[0][times2 <= 1]
np.testing.assert_allclose(mc_ex0, me_ex0, atol=0.2, rtol=0)

mc_ex1 = MCSol2.average_expect[1][times2 <= 1]
me_ex1 = MESol2.expect[1][times2 <= 1]
np.testing.assert_allclose(mc_ex1, me_ex1, atol=0.2, rtol=0)

mc_ex2 = MCSol2.average_expect[2][times2 <= 1]
me_ex2 = MESol2.expect[2][times2 <= 1]
np.testing.assert_allclose(mc_ex2, me_ex2, atol=0.2, rtol=0)
```
