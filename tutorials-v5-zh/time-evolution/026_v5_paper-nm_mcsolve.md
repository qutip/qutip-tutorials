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

# QuTiPv5 论文示例：非马尔可夫热浴的 Monte Carlo 求解器

Authors: Maximilian Meyer-Moelleringhof (m.meyermoelleringhof@gmail.com), Paul Menczel (paul@menczel.net), Neill Lambert (nwlambert@gmail.com)

## 引言

当量子系统出现非马尔可夫效应时，标准 Lindblad 形式不再适用。
但仍可使用 time-convolutionless（TCL）投影算符，将密度矩阵演化写成时间局域微分方程：

$\dot{\rho} (t) = - \dfrac{i}{\hbar} [H(t), \rho(t)] + \sum_n \gamma_n(t) \mathcal{D}_n(t) [\rho(t)]$

其中

$\mathcal{D}_n[\rho(t)] = A_n \rho(t) A^\dagger_n - \dfrac{1}{2} [A^\dagger_n A_n \rho(t) + \rho(t) A_n^\dagger A_n]$.

方程包含系统哈密顿量 $H(t)$ 与跃迁算符 $A_n$。
与 Lindblad 方程不同，这里的耦合率 $\gamma_n(t)$ 可以为负。

QuTiP v5 引入了非马尔可夫 Monte Carlo 求解器。
它可以把上述一般主方程映射为同一希尔伯特空间上的 Lindblad 方程。
核心是引入所谓“影响鞅（influence martingale）”作为轨迹权重[
1, 2, 3](#References)：

$\mu (t) = \exp \left[ \alpha \int_0^t s(\tau) d \tau \right] \Pi_k \dfrac{\gamma_{n_k} (t_k)}{\Gamma_{n_k} (t_k)}$.

其中乘积遍历轨迹上所有跳跃算符，$n_k$ 为第 $k$ 次跳跃通道，$t_k<t$ 为对应跳跃时刻。
为了得到 Lindblad 形式，引入平移函数

$s(t) = 2 \left| \min \{ 0, \gamma_1(t), \gamma_2(t), ... \} \right|$

使平移后速率 $\Gamma_n (t) = \gamma_n(t) + s(t)$ 非负。
于是得到完全正的 Lindblad 方程

$\dot{\rho}'(t) = - \dfrac{i}{\hbar} [ H(t), \rho'(t) ] + \sum_n \Gamma(t) \mathcal{D}_n[\rho'(t)]$,

并且可用常规 MCWF 方法写成 $\rho'(t) = \mathbb{E} \{\ket{\psi(t)} \bra{\psi(t)}\}$。
这里 $\ket{\psi (t)}$ 是轨迹，$\mathbb{E}$ 是对轨迹系综平均。
最终原始态可重建为
$\rho(t) = \mathbb{E}\{\mu(t) \ket{\psi(t)} \bra{\psi(t)}\}$。

注意：该技术要求跳跃算符满足完备关系
$\sum_n A_n^\dagger A_n = \alpha \mathbb{1}$（$\alpha > 0$）。
QuTiP 的 `nm_mcsolve()` 会自动处理这一条件。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (ExponentialBosonicEnvironment, about, basis, brmesolve,
                   expect, heom, ket2dm, lindblad_dissipator, liouvillian,
                   mesolve, nm_mcsolve, qeye, sigmam, sigmap, sigmax, sigmay,
                   sigmaz)
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
```

## 阻尼 Jaynes-Cummings 模型

为展示 `nm_mcsolve()` 的使用，考虑阻尼 Jaynes-Cummings 模型：二能级原子与阻尼腔模耦合。
该模型可等效为二能级系统耦合到如下功率谱环境[
4](#References)：

$S(\omega) = \dfrac{\lambda \Gamma^2}{(\omega_0 - \Delta - \omega)^2 + \Gamma^2}$,

其中 $\lambda$ 是原子-腔耦合强度，$\omega_0$ 为原子跃迁频率，$\Delta$ 为腔失谐，$\Gamma$ 为谱宽。
在零温且做旋波近似后，二能级原子动力学满足

$\dot{\rho}(t) = \dfrac{A(t)}{2i\hbar} [ \sigma_+ \sigma_-, \rho(t) ] + \gamma (t) \mathcal{D}_- [\rho (t)]$,

其中 $\rho(t)$ 在相互作用绘景下，$\sigma_\pm$ 是升降算符，$\mathcal{D}_-$ 对应 Lindblad 算符 $\sigma_-$，而 $\gamma(t)$ 与 $A(t)$ 分别是下式实部与虚部：

$\gamma(t) + i A(t) = \dfrac{2 \lambda \Gamma \sinh (\delta t / 2)}{\delta \cosh (\delta t / 2) + (\Gamma - i \Delta) \sinh (\delta t/2)}$,

其中 $\delta = [(\Gamma - i \Delta)^2 - 2 \lambda \Gamma]^{1/2}$。

```python
H = sigmap() * sigmam() / 2
initial_state = (basis(2, 0) + basis(2, 1)).unit()
tlist = np.linspace(0, 5, 500)
```

```python
# Constants
gamma0 = 1
lamb = 0.3 * gamma0
Delta = 8 * lamb

# Derived Quantities
delta = np.sqrt((lamb - 1j * Delta) ** 2 - 2 * gamma0 * lamb)
deltaR = np.real(delta)
deltaI = np.imag(delta)
deltaSq = deltaR**2 + deltaI**2
```

```python
# calculate gamma and A
def prefac(t):
    return (
        2
        * gamma0
        * lamb
        / (
            (lamb**2 + Delta**2 - deltaSq) * np.cos(deltaI * t)
            - (lamb**2 + Delta**2 + deltaSq) * np.cosh(deltaR * t)
            - 2 * (Delta * deltaR + lamb * deltaI) * np.sin(deltaI * t)
            + 2 * (Delta * deltaI - lamb * deltaR) * np.sinh(deltaR * t)
        )
    )


def cgamma(t):
    return prefac(t) * (
        lamb * np.cos(deltaI * t)
        - lamb * np.cosh(deltaR * t)
        - deltaI * np.sin(deltaI * t)
        - deltaR * np.sinh(deltaR * t)
    )


def cA(t):
    return prefac(t) * (
        Delta * np.cos(deltaI * t)
        - Delta * np.cosh(deltaR * t)
        - deltaR * np.sin(deltaI * t)
        + deltaI * np.sinh(deltaR * t)
    )
```

```python
_gamma = np.zeros_like(tlist)
_A = np.zeros_like(tlist)
for i in range(len(tlist)):
    _gamma[i] = cgamma(tlist[i])
    _A[i] = cA(tlist[i])

gamma = CubicSpline(tlist, np.complex128(_gamma))
A = CubicSpline(tlist, np.complex128(_A))
```

```python
unitary_gen = liouvillian(H)
dissipator = lindblad_dissipator(sigmam())
```

```python
mc_sol = nm_mcsolve(
    [[H, A]],
    initial_state,
    tlist,
    ops_and_rates=[(sigmam(), gamma)],
    ntraj=1_000,
    options={"map": "parallel"},
    seeds=0,
)
```

## 与其他方法对比

我们将其与标准 `mesolve()`、HEOM 和 Bloch-Redfield 求解器对比。
`mesolve()` 可直接复用 `nm_mcsolve()` 中构造的算符：

```python
me_sol = mesolve([[unitary_gen, A], [dissipator, gamma]], initial_state, tlist)
```

对其余方法，直接采用自旋-玻色子模型与自由储备关联函数

$C(t) = \dfrac{\lambda \Gamma}{2} e^{-i (\omega - \Delta) t - \lambda |t|}$,

它对应上面定义的功率谱。
在薛定谔绘景下使用哈密顿量 $H = \omega_0 \sigma_+ \sigma_-$ 和耦合算符 $Q = \sigma_+ + \sigma_-$。
这里取 $\omega_0 \gg \Delta$ 以保证旋波近似有效。

```python
omega_c = 100
omega_0 = omega_c + Delta

H = omega_0 * sigmap() * sigmam()
Q = sigmap() + sigmam()


def power_spectrum(w):
    return gamma0 * lamb**2 / ((omega_c - w) ** 2 + lamb**2)
```

首先，对 HEOM，先把关联函数分成实部和虚部后即可直接求解：

```python
ck_real = [gamma0 * lamb / 4] * 2
vk_real = [lamb - 1j * omega_c, lamb + 1j * omega_c]
ck_imag = np.array([1j, -1j]) * gamma0 * lamb / 4
vk_imag = vk_real
```

```python
heom_env = ExponentialBosonicEnvironment(ck_real, vk_real, ck_imag, vk_imag)
heom_sol = heom.heomsolve(H, (heom_env, Q), 10, ket2dm(initial_state), tlist)
```

其次，对 Bloch-Redfield，可直接把功率谱作为输入：

```python
br_sol = brmesolve(H, initial_state, tlist, a_ops=[(sigmax(), power_spectrum)])
```

最后，为了和 `nm_mesolve()` 结果对比，把它们转换到相互作用绘景：

```python
Us = [(-1j * H * t).expm() for t in tlist]
heom_states = [U * s * U.dag() for (U, s) in zip(Us, heom_sol.states)]
br_states = [U * s * U.dag() for (U, s) in zip(Us, br_sol.states)]
```

### 绘制时间演化

我们绘制三组对比图来突出不同求解器结果。
灰色区域表示 $\gamma(t)$ 为负的时间段。

```python
root1 = root_scalar(lambda t: cgamma(t), method="bisect", bracket=(1, 2)).root
root2 = root_scalar(lambda t: cgamma(t), method="bisect", bracket=(2, 3)).root
root3 = root_scalar(lambda t: cgamma(t), method="bisect", bracket=(3, 4)).root
root4 = root_scalar(lambda t: cgamma(t), method="bisect", bracket=(4, 5)).root
```

```python
projector = (sigmaz() + qeye(2)) / 2

rho11_me = expect(projector, me_sol.states)
rho11_mc = expect(projector, mc_sol.states)
rho11_br = expect(projector, br_sol.states)
rho11_heom = expect(projector, heom_states)

plt.plot(tlist, rho11_me, "-", color="orange", label="mesolve")
plt.plot(
    tlist[::10],
    rho11_mc[::10],
    "x",
    color="blue",
    label="nm_mcsolve",
)
plt.plot(tlist, rho11_br, "-.", color="gray", label="brmesolve")
plt.plot(tlist, rho11_heom, "--", color="green", label="heomsolve")

plt.xlabel(r"$t\, /\, \lambda^{-1}$")
plt.xlim((-0.2, 5.2))
plt.xticks([0, 2.5, 5], labels=["0", "2.5", "5"])
plt.title(r"$\rho_{11}$")
plt.ylim((0.4376, 0.5024))
plt.yticks([0.44, 0.46, 0.48, 0.5], labels=["0.44", "0.46", "0.48", "0.50"])

plt.axvspan(root1, root2, color="gray", alpha=0.08, zorder=0)
plt.axvspan(root3, root4, color="gray", alpha=0.08, zorder=0)

plt.legend()
plt.show()
```

```python
me_x = expect(sigmax(), me_sol.states)
mc_x = expect(sigmax(), mc_sol.states)
heom_x = expect(sigmax(), heom_states)
br_x = expect(sigmax(), br_states)

me_y = expect(sigmay(), me_sol.states)
mc_y = expect(sigmay(), mc_sol.states)
heom_y = expect(sigmay(), heom_states)
br_y = expect(sigmay(), br_states)
```

```python
# We smooth the HEOM result because it oscillates quickly and gets hard to see
rho01_heom = heom_x * heom_x + heom_y * heom_y
rho01_heom = np.convolve(rho01_heom, np.array([1 / 11] * 11), mode="valid")
heom_tlist = tlist[5:-5]
```

```python
rho01_me = me_x * me_x + me_y * me_y
rho01_mc = mc_x * mc_x + mc_y * mc_y
rho01_br = br_x * br_x + br_y * br_y

plt.plot(tlist, rho01_me, "-", color="orange", label=r"mesolve")
plt.plot(tlist[::10], rho01_mc[::10], "x", color="blue", label=r"nm_mcsolve")
plt.plot(heom_tlist, rho01_heom, "--", color="green", label=r"heomsolve")
plt.plot(tlist, rho01_br, "-.", color="gray", label=r"brmesolve")

plt.xlabel(r"$t\, /\, \lambda^{-1}$")
plt.xlim((-0.2, 5.2))
plt.xticks([0, 2.5, 5], labels=["0", "2.5", "5"])
plt.title(r"$| \rho_{01} |^2$")
plt.ylim((0.8752, 1.0048))
plt.yticks(
    [0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1],
    labels=["0.88", "0.90", "0.92", "0.94", "0.96", "0.98", "1"],
)

plt.axvspan(root1, root2, color="gray", alpha=0.08, zorder=0)
plt.axvspan(root3, root4, color="gray", alpha=0.08, zorder=0)

plt.legend()
plt.show()
```

```python
mart_dev = mc_sol.trace - 1
```

```python
plt.plot(tlist, np.zeros_like(tlist), "-", color="orange", label=r"Zero")
plt.plot(
    tlist[::10],
    1000 * mart_dev[::10],
    "x",
    color="blue",
    label=r"nm_mcsolve",
)

plt.xlabel(r"$t\, /\, \lambda^{-1}$")
plt.xlim((-0.2, 5.2))
plt.xticks([0, 2.5, 5], labels=["0", "2.5", "5"])
plt.title(r"$(\mu - 1)\, /\, 10^{-3}$")
plt.ylim((-5.8, 15.8))
plt.yticks([-5, 0, 5, 10, 15])

plt.axvspan(root1, root2, color="gray", alpha=0.08, zorder=0)
plt.axvspan(root3, root4, color="gray", alpha=0.08, zorder=0)

plt.legend()
plt.show()
```

从这些图可观察到两点。
第一，Bloch-Redfield 结果明显偏离其他方法，说明非马尔可夫效应对动力学影响很强。
第二，在灰色区域（$\gamma(t)<0$）中，原子态会恢复相干性。
尤其在最后一幅图中，平均影响鞅在这些区间会波动，而在其余时间基本保持常数。
它偏离 1 的程度可用于判断仿真收敛质量。



## 参考文献

\[1\] [Donvil and Muratore-Ginanneschi. Nat Commun (2022).](https://www.nature.com/articles/s41467-022-31533-8)

\[2\] [Donvil and Muratore-Ginanneschi. New J. Phys. (2023).](https://dx.doi.org/10.1088/1367-2630/acd4dc)

\[3\] [Donvil and Muratore-Ginanneschi. *Open Systems & Information Dynamics*.](https://www.worldscientific.com/worldscinet/osid)

\[4\] [Breuer and Petruccione *The Theory of Open Quantum Systems*.](https://doi.org/10.1093/acprof:oso/9780199213900.001.0001)

\[5\] [QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)



## 关于

```python
about()
```

## 测试

```python
assert np.allclose(
    rho11_me, rho11_heom, atol=1e-3
), "rho11 of mesolve and heomsolve do not agree."
assert np.allclose(
    rho11_me, rho11_mc, atol=1e-2
), "rho11 of nm_mcsolve deviates from mesolve too much."
assert np.allclose(
    rho01_me[5:-5], rho01_heom, atol=1e-3
), "|rho01|^2 of mesolve and heomsolve do not agree."
assert np.allclose(
    rho01_me, rho01_mc, atol=1e-1
), "|rho01|^2 of nm_mcsolve deviates from mesolve too much."
assert (
    np.max(mart_dev) < 1e-1
), "MC Simulation has not converged well enough. Average infl. mart. > 1e-1"
```
