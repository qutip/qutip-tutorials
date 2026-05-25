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

# QuTiPv5 论文示例：量子最优控制包

Authors: Maximilian Meyer-Moelleringhof (m.meyermoelleringhof@gmail.com), Boxi Li (etamin1201@gmail.com), Neill Lambert (nwlambert@gmail.com)

量子系统天然对环境与外部扰动敏感。
这使其非常适合做高精度测量，但也让误差与不确定性控制变得困难。
在量子计算场景中，如何找到能实现目标操作的最优控制参数因此成为关键问题。
待优化参数可包括振幅、频率、持续时间、带宽等，并且通常直接依赖具体硬件。

为寻找这些最优控制参数，社区发展出了多种方法。
这里关注三种算法：*gradient ascent pulse engineering*（GRAPE）[
3](#References)、*chopped random basis*（CRAB）[
4](#References) 与 *gradient optimization af analytic controls*（GOAT）[
5](#References)。
前两者在 QuTiPv4 中属于 `QuTiP-QTRL`，后者是 v5 新增算法。
现在这些算法统一纳入新的 `QuTiP-QOC` 包中，并支持通过 JAX 优化技术（JOPT）与 `QuTiP-JAX`[
6](#References) 集成。

```python
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, numpy
from qutip import (about, gates, liouvillian, qeye, sigmam, sigmax, sigmay,
                   sigmaz)
from qutip_qoc import Objective, optimize_pulses

%matplotlib inline
```

## 引言

本示例要在单量子比特上实现 Hadamard 门。
一般来说，量子比特会受到退相干影响，可在 Lindblad 形式下用跃迁算符 $\sigma_{-}$ 描述。

为简化起见，考虑由 $\sigma_x$、$\sigma_y$、$\sigma_z$ 参数化的控制哈密顿量：

$H_c(t) = c_x(t) \sigma_x + c_y(t) \sigma_y + c_z(t) \sigma_z$

其中 $c_x(t)$、$c_y(t)$、$c_z(t)$ 是独立控制参数。
此外，取常量漂移哈密顿量

$H_d = \dfrac{1}{2} (\omega \sigma_z + \delta \sigma_x)$,

其中 $\omega$ 是能级劈裂，$\delta$ 是隧穿速率。
塌缩算符 $C = \sqrt{\gamma} \sigma_-$ 的振幅阻尼率记为 $\gamma$。

```python
# energy splitting, tunneling, amplitude damping
omega = 0.1  # energy splitting
delta = 1.0  # tunneling
gamma = 0.1  # amplitude damping
sx, sy, sz = sigmax(), sigmay(), sigmaz()

Hc = [sx, sy, sz]  # control operator
Hc = [liouvillian(H) for H in Hc]

Hd = 1 / 2 * (omega * sz + delta * sx)  # drift term
Hd = liouvillian(H=Hd, c_ops=[np.sqrt(gamma) * sigmam()])

# combined operator list
H = [Hd, Hc[0], Hc[1], Hc[2]]
```

```python
# objectives for optimization
initial = qeye(2)
target = gates.hadamard_transform()
fid_err = 0.01
```

```python
# pulse time interval
times = np.linspace(0, np.pi / 2, 100)
```

## 实现

### GRAPE 算法

GRAPE 通过最小化不保真度（infidelity）损失函数，衡量最终态或幺正变换与目标的接近程度。
算法从给定的 `guess` 控制脉冲出发，优化等时间间隔的分段常数振幅。
最终目标是达到 `fid_err_targ` 指定的目标不保真度。

```python
res_grape = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters={
        "ctrl_x": {"guess": np.sin(times), "bounds": [-1, 1]},
        "ctrl_y": {"guess": np.cos(times), "bounds": [-1, 1]},
        "ctrl_z": {"guess": np.tanh(times), "bounds": [-1, 1]},
    },
    tlist=times,
    algorithm_kwargs={"alg": "GRAPE", "fid_err_targ": fid_err},
)
```

### CRAB 算法

该算法将控制场展开在一个随机基上，并优化展开系数 $\vec{\alpha}$。
其优势是可在连续时间区间上使用解析控制函数 $c(\vec{\alpha}, t)$，默认是傅里叶展开。
这会把搜索空间缩减为函数参数空间。
通常这些参数可通过直接搜索算法（如 Nelder-Mead）高效求得。
基函数只展开到有限项，且初始展开系数一般随机给出。

```python
n_params = 3  # adjust in steps of 3
alg_args = {"alg": "CRAB", "fid_err_targ": fid_err, "fix_frequency": False}
```

```python
res_crab = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters={
        "ctrl_x": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
        "ctrl_y": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
        "ctrl_z": {
            "guess": [1 for _ in range(n_params)],
            "bounds": [(-1, 1)] * n_params,
        },
    },
    tlist=times,
    algorithm_kwargs=alg_args,
)
```

### GOAT 算法

与 CRAB 类似，GOAT 也使用解析控制函数。
它通过构造耦合运动方程组，在数值前向积分后可得到（时间有序）演化算符对控制参数的导数。
在无约束设置中，GOAT 在收敛与保真度上常优于前述方法。
QuTiP 的实现允许按常规 Python 方式传入任意控制函数及其导数。

```python
def sin(t, c):
    return c[0] * np.sin(c[1] * t)


# derivatives
def grad_sin(t, c, idx):
    if idx == 0:  # w.r.t. c0
        return np.sin(c[1] * t)
    if idx == 1:  # w.r.t. c1
        return c[0] * np.cos(c[1] * t) * t
    if idx == 2:  # w.r.t. time
        return c[0] * np.cos(c[1] * t) * c[1]
```

```python
H = [Hd] + [[hc, sin, {"grad": grad_sin}] for hc in Hc]

bnds = [(-1, 1), (0, 2 * np.pi)]
ctrl_param = {id: {"guess": [1, 0], "bounds": bnds} for id in ["x", "y", "z"]}
```

为加快收敛，QuTiP 在原算法上扩展了“将总演化时间也作为优化变量”的能力，可通过额外时间关键字参数启用：

```python
# treats time as optimization variable
ctrl_param["__time__"] = {
    "guess": times[len(times) // 2],
    "bounds": [times[0], times[-1]],
}
```

```python
# run the optimization
res_goat = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_param,
    tlist=times,
    algorithm_kwargs={
        "alg": "GOAT",
        "fid_err_targ": fid_err,
    },
)
```

### JOT 算法 - JAX 集成

QuTiP 新的 JAX 后端提供自动微分能力，可直接用于新控制框架。
与 QuTiP 的 GOAT 一样，任何解析控制函数都可传入算法。
不同点在于这里由 JAX 自动微分在整个系统演化中自动计算导数，因此无需手动提供导数。
相较前例，只需把控制函数替换为其即时编译（JIT）版本。

```python
@jit
def sin_y(t, d, **kwargs):
    return d[0] * numpy.sin(d[1] * t)


@jit
def sin_z(t, e, **kwargs):
    return e[0] * numpy.sin(e[1] * t)


@jit
def sin_x(t, c, **kwargs):
    return c[0] * numpy.sin(c[1] * t)
```

```python
H = [Hd] + [[Hc[0], sin_x], [Hc[1], sin_y], [Hc[2], sin_z]]
```

```python
res_jopt = optimize_pulses(
    objectives=Objective(initial, H, target),
    control_parameters=ctrl_param,
    tlist=times,
    algorithm_kwargs={
        "alg": "JOPT",
        "fid_err_targ": fid_err,
    },
)
```

## 结果比较

完成全局和局部优化后，可通过 `qoc.Result` 对象比较各算法结果。该对象提供统一优化指标与 `optimized_controls`。

### 脉冲振幅

```python
fig, ax = plt.subplots(1, 3, figsize=(13, 5))

goat_range = times < res_goat.optimized_params[-1]
jopt_range = times < res_jopt.optimized_params[-1]

for i in range(3):
    ax[i].plot(times, res_grape.optimized_controls[i], ":", label="GRAPE")
    ax[i].plot(times, res_crab.optimized_controls[i], "-.", label="CRAB")
    ax[i].plot(
        times[goat_range],
        np.array(res_goat.optimized_controls[i])[goat_range],
        "-",
        label="GOAT",
    )
    ax[i].plot(
        times[jopt_range],
        np.array(res_jopt.optimized_controls[i])[jopt_range],
        "--",
        label="JOPT",
    )

    ax[i].set_xlabel(r"Time $t$")

ax[0].legend(loc=0)
ax[0].set_ylabel(r"Pulse amplitude $c_x(t)$", labelpad=-5)
ax[1].set_ylabel(r"Pulse amplitude $c_y(t)$", labelpad=-5)
ax[2].set_ylabel(r"Pulse amplitude $c_z(t)$", labelpad=-5)

plt.show()
```

### 不保真度与处理时间

```python
print("GRAPE: ", res_grape.fid_err)
print(res_grape.total_seconds, " seconds")
print()
print("CRAB : ", res_crab.fid_err)
print(res_crab.total_seconds, " seconds")
print()
print("GOAT : ", res_goat.fid_err)
print(res_goat.total_seconds, " seconds")
print()
print("JOPT : ", res_jopt.fid_err)
print(res_jopt.total_seconds, " seconds")
```

## 参考文献

[1] [QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)

[2] [QuTiP-QOC Repository](https://github.com/qutip/qutip-qoc)

[3] [Khaneja, et. al, Journal of Magnetic Resonance (2005)](https://www.sciencedirect.com/science/article/pii/S1090780704003696)

[4] [Caneva, et. al, Phys. Rev. A (2011)](https://link.aps.org/doi/10.1103/PhysRevA.84.022326)

[5] [Machnes, et. al, Phys. Rev. Lett. (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.150401)

[6] [QuTiP-JAX Repository](https://github.com/qutip/qutip-jax)



## 关于

```python
about()
```

## 测试

```python
assert (
    res_grape.fid_err < fid_err
), f"GRAPE did not reach the target infidelity of < {fid_err}."
assert (
    res_crab.fid_err < fid_err
), f"CRAB did not reach the target infidelity of < {fid_err}."
assert (
    res_goat.fid_err < fid_err
), f"GOAT did not reach the target infidelity of < {fid_err}."
assert (
    res_jopt.fid_err < fid_err
), f"JOPT did not reach the target infidelity of < {fid_err}."
```
