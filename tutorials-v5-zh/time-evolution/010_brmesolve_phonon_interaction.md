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

# Bloch-Redfield 求解器：声子辅助初始化

Author: K.A. Fischer, Stanford University

本 Jupyter 笔记演示如何使用含时 Bloch-Redfield 主方程求解器，来模拟量子点的声子辅助初始化过程。示例基于 QuTiP（Python 量子工具箱）。核心目标是说明：环境驱动的耗散相互作用可以被利用来将量子点初始化到激发态。该笔记基本遵循这篇工作：<a href="https://arxiv.org/abs/1409.6014">Dissipative preparation of the exciton and biexciton in self-assembled quantum
dots on picosecond time scales</a>, Phys. Rev. B 90, 241404(R) (2014).

QuTiP 更多信息请见项目主页：http://qutip.org/ 

```python
import itertools

import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, fock, parallel_map, sigmam, BRSolver,
                   QobjEvo, coefficient)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

## 引言

量子二能级系统（TLS）是描述光与人工原子（量子点）量子相互作用的最简单模型。论文中的实验与仿真都使用了三能级模型；这里为降低笔记运行时间，仅展示 TLS 模型。

在本示例中，系统由连续模相干态驱动，其偶极相互作用哈密顿量写为

$$ H =\hbar \omega_0 \sigma^\dagger \sigma + \frac{\hbar\Omega(t)}{2}\left( \sigma\textrm{e}^{-i\omega_dt} + \sigma^\dagger \textrm{e}^{i\omega_dt}\right),$$

其中 $\omega_0$ 为系统跃迁频率，$\sigma$ 为系统降算符，$\omega_d$ 为相干态中心频率，$\Omega(t)$ 为驱动强度。

通过旋转参考系变换可以去除显式含时项，从而简化仿真，此时

$$ H_r =\hbar \left(\omega_0-\omega_d\right) \sigma^\dagger \sigma + \frac{\hbar\Omega(t)}{2}\left( \sigma+ \sigma^\dagger \right).$$

另外，量子点处在固体基质中，环境相互作用非常重要。尤其是声学声子与人工原子的耦合会带来显著退相干效应。尽管可用量子光学主方程中的塌缩算符做唯象建模，但这种方式不一定能直接反映系统-环境相互作用的底层物理。

相比之下，Bloch-Redfield 主方程可将量子动力学与底层物理机制直接关联。并且相互作用强度可以从第一性原理导出，能够包含复杂的功率依赖（例如量子点-声子耦合中的情形）。当然，也存在重要的非马尔可夫效应，当前仍在研究中，例如论文 <a href="https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.201305">Limits to coherent scattering and photon coalescence from solid-state quantum emitters</a>, Phys. Rev. B 95, 201305(R) (2017)。


### 问题参数

注意：这里采用单位 $\hbar=1$。

```python
# 脉冲面积
n_Pi = 13

# 驱动强度
Om_list = np.linspace(0.001, n_Pi, 80)
# 激光失谐（meV）
wd_list_e = np.array([-1, 0, 1])
# 激光失谐（角频率）
wd_list = wd_list_e * 1.5
# 仿真时间，tmax 约为 2x FWHM
tlist = np.linspace(0, 50, 40)

# 归一化高斯脉冲包络，能量尺度下约 10ps
t0 = 17 / (2 * np.sqrt(2 * np.log(2)))
pulse_shape = "0.0867 * exp(-(t - 24) ** 2 / (2 * {0} ** 2))".format(t0)
```

### 设置算符、哈密顿量与初态

```python
# 初态
psi0 = fock(2, 1)  # 基态

# 系统降算符
sm = sigmam()

# 哈密顿量分量
H_S = -sm.dag() * sm  # 自能项，随驱动频率变化
H_I = sm + sm.dag()

# 由于脉冲远快于衰减时间，这里忽略自发辐射
c_ops = []
```

下面定义 Bloch-Redfield 求解器中系统-环境耦合相关的项。量子点通过如下色散型电子-声子相互作用与固体环境中的声学声子耦合：

$$ H_\textrm{phonon}=\hbar J(\omega)\sigma^\dagger \sigma,$$

其中 $J(\omega)$ 为耦合谱密度。

```python
# 量子点与声学声子耦合的算符
a_op = sm.dag() * sm

# 该谱是位移高斯与 w^3 的乘积，
# 用于模拟与纵向声学（LA）声子耦合。
# 电子与空穴相互作用会相干叠加。


# 拟合参数 ae/ah
ah = 1.9e-9  # m
ae = 3.5e-9  # m
# GaAs 材料参数
De = 7
Dh = -3.5
v = 5110  # m/s
rho_m = 5370  # kg/m^3
# 其他常数
hbar = 1.05457173e-34  # Js
T = 4.2  # Kelvin, temperature

# 温度依赖相关常数
t1 = 0.6582119
t2 = 0.086173

# J 的通用项
J = (
    "(1.6 * 1e-13 * w**3) / (4 * pi**2 * rho_m * hbar * v**5) * "
    + "(De * exp(-(w * 1e12 * ae * 0.5 / v)**2) - "
    + "Dh * exp(-(w * 1e12 * ah * 0.5 / v)**2))**2"
)

# 正频率项
JT_p = (
    J
    + "* (1 + exp(-w*t1/(T*t2)) / \
          (1-exp(-w*t1/(T*t2))))"
)

# 负频率项
JT_m = (
    "-1.0* "
    + J
    + "* exp(w*t1/(T*t2)) / \
            (1-exp(w*t1/(T*t2)))"
)


# 用变量名定义谱函数
spectra_cb = "(w > 0) * " + JT_p + "+ (w < 0) * " + JT_m

# 对很小的 w 加保护，避免数值问题
spectra_cb = "0 if (w > -1e-4 and w < 1e-4) else " + spectra_cb

# 仅保留 w 为变量，其余常数替换成数值
constants = ["ah", "ae", "De", "Dh", "v", "rho_m", "hbar", "T", "t1", "t2"]
spectra_cb_numerical = spectra_cb
for c in constants:
    # 将常数替换为其数值
    spectra_cb_numerical = spectra_cb_numerical.replace(c, str(eval(c)))
```

## 可视化量子点-声子相互作用谱

$J(\omega)$ 的形状由两部分共同决定：一是声学声子态密度上升带来的增长项；二是量子点有限尺寸导致的高频滚降。

```python
# 频率网格
spec_list = np.linspace(-5, 10, 200)

# 定义快捷名以便 eval 字符串
pi = np.pi
exp = np.exp

# 绘制谱 J(w)
plt.figure(figsize=(8, 5))
plt.plot(spec_list, [eval(spectra_cb.replace("w", str(_))) for _ in spec_list])
plt.xlim(-5, 10)
plt.xlabel("$\\omega$ [THz]")
plt.ylabel("$J(\\omega)$ [THz]")
plt.title("Quantum-dot-phonon interaction spectrum");
```

## 计算脉冲-系统相互作用动力学

Bloch-Redfield 求解器接收列表-字符串格式的含时哈密顿量。我们计算脉冲与系统作用结束时的最终布居，它代表系统被初始化到激发态的概率。

```python
# 计算量子点布居期望值
e_ops = [sm.dag() * sm]


# 初始化 BRSolver
H = QobjEvo([[H_S, 'wd'], [H_I, 'Om * ' + pulse_shape]],
            args={'wd': 0.0, 'Om': 0.0})

spectrum = coefficient(spectra_cb_numerical, args={'w': 0})
solver = BRSolver(H, [[a_op, spectrum]])


# 用于并行计算的回调函数
def brme_step(args):
    # 提取哈密顿量系数
    args = {'wd': args[0], 'Om': args[1]}

    # 运行求解器
    res = solver.run(psi0, tlist, e_ops=e_ops, args=args)

    # 返回脉冲作用后布居
    return res.expect[0][-1]


# 使用 QuTiP 内置并行循环 parallel_map
results = parallel_map(brme_step, list(itertools.product(wd_list, Om_list)))

# 整理为 2D 数组
inv_mat_X = np.array(results).reshape((len(wd_list), len(Om_list)))
```

### 可视化量子点初始化保真度

下面先看失谐 $\omega_d-\omega_L=0$ 时，激发态布居随脉冲面积增大的轨迹。振荡对应被驱动二能级系统中的标准 Rabi 振荡，并会随脉冲面积增大而被类马尔可夫退相干阻尼。该阻尼可在普通量子光学主方程中用功率依赖塌缩算符拟合。

但在非零失谐时，结果会变得高度非平凡，很难用简单塌缩算符建模。Bloch-Redfield 方法的优势正体现在这里：它能在更自然的“缀饰态基底”中，从第一性原理捕捉退相干。

在该基底下，色散型声子诱导退相干会在缀饰态之间驱动布居差。这等价于把系统推向一个耗散型准稳态，使激发态布居以接近 1 的保真度完成初始化。正如原论文讨论的那样，这种初始化对脉冲面积和激光失谐都不太敏感，因此是一种非常鲁棒的量子点激发方案。下面给出 +1 meV 失谐下的示例轨迹。高保真初始化依赖于低温热浴（更偏向声子发射而非吸收）。作为对照，在 -1 meV 失谐下，激发态几乎不被布居。

```python
plt.figure(figsize=(8, 5))

plt.plot(Om_list, inv_mat_X[0])
plt.plot(Om_list, inv_mat_X[1])
plt.plot(Om_list, inv_mat_X[2])

plt.legend(["laser detuning, -1 meV", "laser detuning, 0 meV",
            "laser detuning, +1 meV"], loc=4)

plt.xlim(0, 13)
plt.xlabel("Pulse area [$\\pi$]")
plt.ylabel("Excited state population")
plt.title("Effects of phonon dephasing for different pulse detunings");
```

## 版本信息

```python
about()
```

### 测试

```python
# +1meV 失谐时布居应单调递增
assert np.all(np.diff(inv_mat_X[2]) > 0)

# +1meV 失谐时稳态应接近激发态
assert inv_mat_X[2][-1] > 0.9
# -1meV 失谐时稳态应接近基态
assert inv_mat_X[0][-1] < 0.1
```
