---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Floquet 求解器

Author: C. Staufenbiel, 2022

### 简介

*Floquet 形式主义*处理周期性时间依赖系统。
对这类问题，Floquet 方法往往比标准主方程求解器 `qutip.mesolve()` 更高效，
并且在周期驱动情形下具有更广泛的适用性。

本 notebook 通过一个示例量子系统，介绍 QuTiP 中 Floquet 形式主义相关求解器的用法。
更详细理论介绍见[文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-floquet.html)。

若想更深入了解 Floquet 形式主义底层函数（`fsesolve` 与 `fmmesolve` 也依赖这些函数），
可参考[*floquet formalism notebook*](012_floquet_formalism.md)。

### 导入

```python
import numpy as np
from qutip import (about, basis, fmmesolve, fsesolve,
                   plot_expectation_values, sigmax, sigmaz)
```

本例考虑一个强驱动双能级系统，其时间依赖哈密顿量为：

$$ H(t) = -\frac{\Delta}{2} \sigma_x - \frac{\epsilon_0}{2} \sigma_z + \frac{A}{2} sin(\omega t) \sigma_z$$

```python
# define constants
delta = 0.2 * 2 * np.pi
eps0 = 2 * np.pi
A = 2.5 * 2 * np.pi
omega = 2 * np.pi

# Non driving hamiltoninan
H0 = -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz()

# Driving Hamiltonian
H1 = [A / 2.0 * sigmaz(), "sin(w*t)"]
args = {"w": omega}

# combined hamiltonian
H = [H0, H1]

# initial state
psi0 = basis(2, 0)
```

### Floquet 薛定谔方程

现在可用 `qutip.fsesolve()` 在 Floquet 形式主义下求解系统薛定谔方程动力学。
其参数与 `qutip.sesolve()` 类似。
可选参数 `T` 用于定义时间依赖项周期。
若不提供 `T`，默认认为给定 `tlist` 跨越一个周期。
因此本教程中我们始终显式传入 `T`。

```python
# period time
T = 2 * np.pi / omega
# simulation time
tlist = np.linspace(0, 2.5 * T, 101)
# simulation
result = fsesolve(H, psi0, tlist, T=T, e_ops=[sigmaz()], args=args)

plot_expectation_values([result], ylabels=["<Z>"]);
```

### Floquet-Markov 主方程

类似 `mesolve()`，我们也可用 Floquet 方法求解含耗散量子系统的主方程，
对应函数为 `fmmesolve()`。
但此时耗散过程由噪声谱密度函数描述。

例如定义线性噪声谱密度：

$$ S(\omega) = \frac{\gamma \cdot \omega}{4 \pi} $$

其中 $\gamma$ 为耗散率。
系统-热浴相互作用由耦合算符描述，这里使用 $\sigma_x$。

每个谱函数可调用对象应接收一个频率 numpy 数组，并返回对应谱密度数组。
传入频率对应 Floquet 准能差，可能为负值。
若希望负频率处功率为零，可将谱密度函数乘上 `(omega > 0)`，
如下代码所示。

```python
# Noise Spectral Density
gamma = 0.5


def noise_spectrum(omega):
    return (omega > 0) * gamma * omega / (4 * np.pi)


# Coupling operator and noise spectrum
c_ops = [sigmax()]
spectra_cb = [noise_spectrum]

# Solve using Fmmesolve
fme_result = fmmesolve(
    H,
    psi0,
    tlist,
    c_ops=c_ops,
    spectra_cb=spectra_cb,
    e_ops=[sigmaz()],
    T=T,
    args=args,
)
```

将结果与 `fsesolve()` 的期望值对比，可观察到耗散动力学。

```python
fig, axes = plot_expectation_values([result, fme_result], ylabels=["<Z>"])
axes[0].legend(['fsesolve', 'fmmesolve'], loc='upper right');
```

### 环境信息

```python
about()
```

### 测试

```python
fme_result_nodis = fmmesolve(
    H,
    psi0,
    tlist,
    c_ops=c_ops,
    spectra_cb=[lambda w: np.zeros_like(w)],
    e_ops=[sigmaz()],
    T=T,
    args=args,
)
```

```python
assert np.allclose(result.expect[0], fme_result_nodis.expect[0], atol=0.1)
assert not np.allclose(fme_result.expect[0],
                       fme_result_nodis.expect[0], atol=0.1)
```
