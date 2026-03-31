---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 使用 CRAB 算法计算双量子比特态到态转移控制场


Jonathan Zoller (jonathan.zoller@uni-ulm.de)


本示例演示如何使用控制库中的 `ctrlpulseoptim.optimize_pulse_unitary`
函数求解控制脉冲。
采用 CRAB 算法优化脉冲形状以最小化保真度误差，
等价于将保真度最大化到 1。

本例系统为两个量子比特，可控项是它们之间的相互作用。
目标是实现纯态转移：从 down-down 态到 up-up 态。

用户可通过修改时间片数量、总演化时间等参数进行实验；
也可调整初始脉冲类型、控制边界，
以及控制开关过程的平滑升降（脉冲起始和结束处）。
示例会绘制初始与最终脉冲。

关于这类方法的深入讨论见 [1,2]。

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np
import random

import qutip_qtrl.pulseoptim as cpo
from qutip import Qobj, identity, sigmax, sigmaz, tensor, about

example_name = "2qubitInteract"
%matplotlib inline
```

### 定义物理模型


系统动力学由总哈密顿量控制：
H(t) = H_d + sum(u1(t)*Hc1 + u2(t)*Hc2 + ....)
即时间依赖哈密顿量由常量漂移部分和时间变化控制部分构成。

本例采用 Ising-like 哈密顿量：
漂移项包含随机系数，控制项作用于比特间耦合：

$ \hat{H} = \sum_{i=1}^2 \alpha_i \sigma_x^i + \beta_i \sigma_z^i + u(t) \cdot \sigma_z \otimes \sigma_z $

初态 $\newcommand{\ket}[1]{\left|{#1}\right\rangle} \ket{\psi_0} = \text{U_0}$
与目标态 $\ket{\psi_t} = \text{U_targ}$ 取为：

$ \ket{\psi_0} = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}$

$ \ket{\psi_t} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$

```python
random.seed(20)
alpha = [random.random(), random.random()]
beta = [random.random(), random.random()]

Sx = sigmax()
Sz = sigmaz()

H_d = (
    alpha[0] * tensor(Sx, identity(2))
    + alpha[1] * tensor(identity(2), Sx)
    + beta[0] * tensor(Sz, identity(2))
    + beta[1] * tensor(identity(2), Sz)
)
H_c = [tensor(Sz, Sz)]
# Number of ctrls
n_ctrls = len(H_c)

q1_0 = q2_0 = Qobj([[1], [0]])
q1_targ = q2_targ = Qobj([[0], [1]])

psi_0 = tensor(q1_0, q2_0)
psi_targ = tensor(q1_targ, q2_targ)
```

### 定义时间演化参数


求解演化时，控制幅值在每个时间片内视为常量，
单时间片演化可由 U(t_k)=expm(-i*H(t_k)*dt) 计算。
将所有时间片演化组合后，
得到从 t=0 初态 $\psi_0$ 到 t=evo_time 的 U(T) 近似。

应选择足够小的时间片时长（dt），
使其相对系统动力学足够细。

```python
# Number of time slots
n_ts = 100
# Time allowed for the evolution
evo_time = 18
```

### 设定优化终止条件


每次迭代都会比较计算得到的 U(T) 与目标 U_targ 的保真度。
对幺正系统，通常有
f = normalise(overlap(U(T), U_targ))。
最大保真度为 1，因此误差定义为 fid_err = 1 - fidelity。
当误差低于阈值时视为优化完成。

实践中可能陷入局部极小，或目标误差不可达，
因此需设置最大时间/最大迭代等限制。

该示例使用 CRAB 算法搜索优化系数，
底层搜索器为 Nelder-Mead 单纯形下降法。
当单纯形收缩到足够小，算法会终止。

```python
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
```

### 设定初始脉冲类型


控制幅值需设初始值。
常见做法是各控制、各时间片随机初始化，
但这可能导致优化后脉冲较不平滑。
本例中多数初值都能得到可行解，
因此可以尝试不同初始类型。

```python
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = "DEF"
```

### 设置输出文件扩展名

```python
# Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
```

### 运行优化


这一步执行实际优化。
Nelder-Mead 会在每次迭代中更新系数集合，
以改进当前最差系数组。
更多细节见 [1,2] 及静态搜索方法教材。

算法会在触发任一终止条件时停止。
若结果不理想，可重跑并/或调整待优化系数数目
（这是 CRAB 中非常关键的参数）。

```python
result = cpo.opt_pulse_crab_unitary(
    H_d,
    H_c,
    psi_0,
    psi_targ,
    n_ts,
    evo_time,
    fid_err_targ=fid_err_targ,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    init_coeff_scaling=5.0,
    num_coeffs=5,
    method_params={"xtol": 1e-3},
    guess_pulse_type=None,
    guess_pulse_action="modulate",
    out_file_ext=f_ext,
    gen_stats=True,
)
```

### 输出结果


先输出性能统计，用于查看各环节耗时。
本例中主要耗时通常在传播子计算（组合哈密顿量指数化）。

`final evolution` 输出的是优化终止时完整时间演化的 `Qobj` 表示。

最重要信息在末尾 summary：
给出最终保真度误差与算法终止原因。

```python
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print(
    "Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time))
)
```

### 绘制初始与最终控制幅值


下面绘制随机初始脉冲与最终优化脉冲，
后者可在设定误差范围内实现目标门演化。

```python
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial Control amps")
ax1.set_ylabel("Control amplitude")
ax1.step(
    result.time,
    np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
    where="post",
)

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Amplitudes")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(
    result.time,
    np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
    where="post",
)
plt.tight_layout()
plt.show()
```

### 版本信息

```python
about()
```

### 参考文献


[1] Doria, P., Calarco, T. & Montangero, S.: Optimal Control Technique for Many-Body Quantum Dynamics. Phys. Rev. Lett. 106, 1–? (2011).

[2] Caneva, T., Calarco, T. & Montangero, S.: Chopped random-basis quantum optimization. Phys. Rev. A - At. Mol. Opt. Phys. 84, (2011).
