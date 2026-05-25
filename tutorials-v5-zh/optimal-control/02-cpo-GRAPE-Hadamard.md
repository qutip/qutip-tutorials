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

# 使用 L-BFGS-B 算法计算单量子比特 Hadamard 门控制场


Alexander Pitchford (agp1@aber.ac.uk)


本示例演示如何使用控制库中的 `ctrlpulseoptim.optimize_pulse_unitary`
函数求解控制脉冲。
使用（默认）L-BFGS-B 算法优化脉冲，使保真度误差最小，
等价于使保真度最大并逼近 1。

本例系统是：单量子比特在 z 方向恒定场中演化，
并在 x 方向受可变控制场驱动。
目标演化是 Hadamard 门（忽略全局相位）。

用户可通过修改时间离散参数（时间片数量、总演化时间）进行实验，
也可尝试不同初始脉冲类型。
示例会绘制初始脉冲与优化后脉冲。

关于这类方法更深入的讨论见 [1]。

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulseoptim as cpo
from qutip import gates, identity, sigmax, sigmaz, about

example_name = "Hadamard"

%matplotlib inline
```

### 定义物理模型


系统动力学由总哈密顿量控制：
H(t) = H_d + sum(u1(t)*Hc1 + u2(t)*Hc2 + ....)
其中时间依赖哈密顿量由常量部分（drift）和随时间变化控制部分组成，
后者由控制哈密顿量及其控制幅值函数 $u_j(t)$ 给出。

本例中 drift 是绕 z 轴旋转，
时间变化控制是绕 x 轴旋转。
理论上该系统在忽略全局相位下是完全可控的，
因此可实现任意幺正目标；本例选择 Hadamard 门。

```python
# Drift Hamiltonian
H_d = sigmaz()
# The (single) control Hamiltonian
H_c = [sigmax()]
# start point for the gate evolution
U_0 = identity(2)
# Target for the gate evolution Hadamard gate
U_targ = gates.hadamard_transform(1)
```

### 定义时间演化参数


求解演化时，控制幅值在每个时间片内视为常量，
因此时间片演化可写为 U(t_k)=expm(-i*H(t_k)*dt)。
将各时间片传播子串联可得到从 t=0 的初始算符到
t=evo_time 的 U(T) 近似演化。

应选择足够小的时间片宽度（dt），
使其相对系统动力学足够细。

```python
# Number of time slots
n_ts = 10
# Time allowed for the evolution
evo_time = 10
```

### 设定优化终止条件


每次迭代都会比较 U(T) 与目标 U_targ 的保真度。
对幺正系统通常有：
f = normalise(overlap(U(T), U_targ))
具体归一化细节见 [1] 或源代码。

该定义下最大保真度为 1，
故误差定义为 fid_err = 1 - fidelity。
当误差低于目标阈值时可认为优化完成。

实践中可能出现局部极小，或目标误差不可达，
因此需设置时间/算力上限。

算法利用梯度引导搜索。
若所有梯度平方和低于 `min_grad`，
可视为已到达某个局部极小。

```python
# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
```

### 设定初始脉冲类型


控制幅值必须先给定初值。
通常是各控制、各时间片随机值。
随机初值可能得到较“锯齿/杂乱”的优化脉冲。
本例中多数初值都能找到解，因此可尝试不同类型。

```python
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = "RND"
```

### 设置输出文件扩展名

```python
# Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
```

### 运行优化


该步骤将调用 L-BFGS-B 算法。
每次迭代中，会用精确梯度法计算“各时间片各控制幅值”对保真度误差的梯度
（见 [1]）。
算法据此更新分段控制幅值以降低误差。
随着迭代推进，会形成 Hessian 近似，
从而实现拟二阶牛顿式搜索。

算法会在触发任一终止条件时停止。

```python
result = cpo.optimize_pulse_unitary(
    H_d,
    H_c,
    U_0,
    U_targ,
    n_ts,
    evo_time,
    fid_err_targ=fid_err_targ,
    min_grad=min_grad,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    out_file_ext=f_ext,
    init_pulse_type=p_type,
    gen_stats=True,
)
```

### 输出结果


先输出性能统计，可查看各阶段耗时分解。
统计中的主要时间通常来自保真度与梯度计算。
剩余时间可认为主要由优化器（L-BFGS-B）本身消耗。
在本例中，主要耗时通常在传播子计算（即组合哈密顿量指数化）。

优化终止时的 U(T) 作为“final evolution”输出。
其本质是保存最终演化算符的 `Qobj` 字符串表示。

最关键信息在末尾 summary：
包括最终保真度误差与算法终止原因。

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


下面绘制随机初始脉冲与优化后脉冲，
后者可在指定误差阈值内实现目标门演化。

```python
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
# ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(
    result.time,
    np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
    where="post",
)

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
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


[1] Machnes et.al., DYNAMO - Dynamic Framework for Quantum Optimal Control. arXiv.1011.4874
