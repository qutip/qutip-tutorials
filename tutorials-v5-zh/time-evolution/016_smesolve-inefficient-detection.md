---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 随机求解器：混合随机与确定性方程


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from qutip import (
    about,
    coherent,
    destroy,
    fock,
    liouvillian,
    mcsolve,
    mesolve,
    plot_expectation_values,
    smesolve,
)

%matplotlib inline

rcParams["font.family"] = "STIXGeneral"
rcParams["mathtext.fontset"] = "stix"
rcParams["font.size"] = "14"
```

## 直接光子计数检测


这里沿用 Wiseman 与 Milburn《Quantum measurement and control》第 4.8.1 节的示例。

考虑一个以速率 $\kappa$ 泄漏光子的腔。泄漏光子由效率为 $\eta$ 的非理想光子探测器检测。
将“被探测到的光子”和“漏检光子”分别视为不同耗散通道后，主方程为

$\dot\rho = -i[H, \rho] + \mathcal{D}[\sqrt{1-\eta} \sqrt{\kappa} a] + \mathcal{D}[\sqrt{\eta} \sqrt{\kappa}a]$

若只对“检测通道”做随机展开，而把“漏检通道”保留为确定性耗散，可得 [W&M 式 (4.235)]

$d\rho = \mathcal{H}[-iH -\eta\frac{1}{2}a^\dagger a] \rho dt + \mathcal{D}[\sqrt{1-\eta} a] \rho dt + \mathcal{G}[\sqrt{\eta}a] \rho dN(t)$

或

$d\rho = -i[H, \rho] dt + \mathcal{D}[\sqrt{1-\eta} a] \rho dt -\mathcal{H}[\eta\frac{1}{2}a^\dagger a] \rho dt + \mathcal{G}[\sqrt{\eta}a] \rho dN(t)$

其中

$\displaystyle \mathcal{G}[A] \rho = \frac{A\rho A^\dagger}{\mathrm{Tr}[A\rho A^\dagger]} - \rho$

$\displaystyle \mathcal{H}[A] \rho = A\rho + \rho A^\dagger - \mathrm{Tr}[A\rho + \rho A^\dagger] \rho $

且 $dN(t)$ 为泊松增量，满足 $E[dN(t)] = \eta \langle a^\dagger a\rangle (t)$。


### QuTiP 中的写法

光电流随机主方程可写为：

$\displaystyle d\rho(t) = -i[H, \rho] dt + \mathcal{D}[B] \rho dt 
- \frac{1}{2}\mathcal{H}[A^\dagger A] \rho(t) dt 
+ \mathcal{G}[A]\rho(t) d\xi$

前两项对应确定性主方程（Lindblad 形式，塌缩算符为 $B$，即 `c_ops`），而 $A$ 为随机塌缩算符（`sc_ops`）。 

这里 $A = \sqrt{\eta\gamma} a$，$B = \sqrt{(1-\eta)\gamma} $a。

在 QuTiP 中，只要把确定性部分作为 Liouvillian 传入，Monte Carlo 求解器即可求解该方程。

```python
N = 15
w0 = 0.5 * 2 * np.pi
times = np.linspace(0, 15, 150)
dt = times[1] - times[0]
gamma = 0.1

a = destroy(N)

H = w0 * a.dag() * a

rho0 = fock(N, 5)

e_ops = [a.dag() * a, a + a.dag()]
```

### 高效率检测

```python
eta = 0.7
c_ops = [np.sqrt(1 - eta) * np.sqrt(gamma) * a]  # collapse operator B
sc_ops = [np.sqrt(eta) * np.sqrt(gamma) * a]  # stochastic collapse operator A
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
result1 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=1,
)
```

```python
result2 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=10,
)
```

```python
np.array(result2.runs_photocurrent).dtype
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(
    times, result1.expect[0], label=r"Stochastic ME (ntraj = 1)", lw=2
)
axes[0, 0].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 0].set_title("Cavity photon number (ntraj = 1)")
axes[0, 0].legend()

axes[0, 1].plot(
    times, result2.expect[0], label=r"Stochatic ME (ntraj = 10)", lw=2
)
axes[0, 1].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 1].set_title("Cavity photon number (ntraj = 10)")
axes[0, 1].legend()

axes[1, 0].step(times[1:], dt * np.cumsum(result1.photocurrent), lw=2)
axes[1, 0].set_title("Cummulative photon detections (ntraj = 1)")
axes[1, 1].step(times[1:], dt * np.cumsum(result2.photocurrent), lw=2)
axes[1, 1].set_title("Cummulative avg. photon detections (ntraj = 10)")

fig.tight_layout()
```

### 低效率光子检测

```python
eta = 0.1
c_ops = [np.sqrt(1 - eta) * np.sqrt(gamma) * a]  # collapse operator B
sc_ops = [np.sqrt(eta) * np.sqrt(gamma) * a]  # stochastic collapse operator A
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
result1 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=1,
)
```

```python
result2 = mcsolve(
    liouvillian(H, c_ops),
    rho0,
    times,
    sc_ops,
    e_ops=e_ops,
    ntraj=10,
)
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(
    times, result1.expect[0], label=r"Stochastic ME (ntraj = 1)", lw=2
)
axes[0, 0].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 0].set_title("Cavity photon number (ntraj = 1)")
axes[0, 0].legend()

axes[0, 1].plot(
    times, result2.expect[0], label=r"Stochatic ME (ntraj = 10)", lw=2
)
axes[0, 1].plot(times, result_ref.expect[0], label=r"Lindblad ME", lw=2)
axes[0, 1].set_title("Cavity photon number (ntraj = 10)")
axes[0, 1].legend()

axes[1, 0].step(times[1:], dt * np.cumsum(result1.photocurrent[0]), lw=2)
axes[1, 0].set_title("Cummulative photon detections (ntraj = 1)")
axes[1, 1].step(times[1:], dt * np.cumsum(result2.photocurrent[0]), lw=2)
axes[1, 1].set_title("Cummulative avg. photon detections (ntraj = 10)")

fig.tight_layout()
```

## 非理想同相检测


对于低效率同相检测，对主方程检测部分进行随机展开：

$\dot\rho = -i[H, \rho] + \mathcal{D}[\sqrt{1-\eta} \sqrt{\kappa} a] + \mathcal{D}[\sqrt{\eta} \sqrt{\kappa}a]$,

W&M 给出的随机主方程为

$d\rho = -i[H, \rho]dt + \mathcal{D}[\sqrt{1-\eta} \sqrt{\kappa} a] \rho dt 
+
\mathcal{D}[\sqrt{\eta} \sqrt{\kappa}a] \rho dt
+
\mathcal{H}[\sqrt{\eta} \sqrt{\kappa}a] \rho d\xi$

其中 $d\xi$ 是 Wiener 增量。可将其理解为：效率为 $\eta$ 的标准同相检测，加上一个塌缩算符为 $\sqrt{(1-\eta)\kappa} a$ 的确定性耗散过程。或者把两个确定性项并入标准 Lindblad 形式，得到（W&M 采用的形式）

$d\rho = -i[H, \rho]dt + \mathcal{D}[\sqrt{\kappa} a]\rho dt + \sqrt{\eta}\mathcal{H}[\sqrt{\kappa}a] \rho d\xi$

```python
rho0 = coherent(N, np.sqrt(5))
```

### 标准同相检测 + Lindblad 形式确定性耗散

```python
eta = 0.95
c_ops = [np.sqrt(1 - eta) * np.sqrt(gamma) * a]  # collapse operator B
sc_ops = [np.sqrt(eta) * np.sqrt(gamma) * a]  # stochastic collapse operator A
```

```python
result_ref = mesolve(H, rho0, times, c_ops + sc_ops, e_ops)
```

```python
options = {
    "method": "platen",
    "store_measurement": True,
    "map": "parallel",
}

result = smesolve(
    H,
    rho0,
    times,
    c_ops=c_ops,
    sc_ops=sc_ops,
    e_ops=e_ops,
    ntraj=75,
    options=options,
)
```

```python
plot_expectation_values([result, result_ref])
```

```python
fig, ax = plt.subplots(figsize=(8, 4))

M = np.sqrt(eta * gamma)

for m in result.measurement:
    ax.plot(times[1:], m[0, :].real / M, "b", alpha=0.025)

ax.plot(times, result_ref.expect[1], "k", lw=2)

ax.set_ylim(-25, 25)
ax.set_xlim(0, times.max())
ax.set_xlabel("time", fontsize=12)
ax.plot(
    times[1:], np.mean(result.measurement, axis=0)[0, :].real / M, "b", lw=2
)
```

## 版本

```python
about()
```
