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

# 使用 L-BFGS-B 算法计算 Lindbladian 动力学控制场


Christian Arenz (christianarenz.ca@gmail.com), Alexander Pitchford (alex.pitchford@gmail.com)


本示例演示如何使用控制库中的 `ctrlpulseoptim.optimize_pulse`
函数求解控制脉冲。
采用（默认）L-BFGS-B 算法最小化保真度误差，
本例中误差由“Trace difference”范数定义。

这是开放量子系统示例：
单量子比特受振幅阻尼通道作用，目标演化是 Hadamard 门。
一般地，对 $d$ 维系统，Lindbladian 可通过 Liouvillian 超算符表示为
$d^2 \times d^2$ 矩阵。
本例将振幅阻尼的 Lindbladian 写为超算符形式，
控制生成元也转换为超算符；
初始映射与目标映射同样需使用超算符形式。

用户可通过修改 `gamma` 调节振幅阻尼强度。
若阻尼足够弱，则可在给定容差内达到目标保真度。
也可替换 drift 哈密顿量和控制生成元，
以实验可控/不可控配置。

用户还可通过修改时间片数量和总演化时间实验不同时间离散。
可尝试不同初始脉冲类型。
示例会绘制初始与最终脉冲。

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulseoptim as cpo
from qutip import (
    gates,
    identity,
    liouvillian,
    sigmam,
    sigmax,
    sigmay,
    sigmaz,
    sprepost,
    about,
)

example_name = "Lindblad"

%matplotlib inline
```

### 定义物理模型

```python
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Sm = sigmam()
Si = identity(2)
# Hadamard gate
had_gate = gates.hadamard_transform(1)

# Hamiltonian
Del = 0.1  # Tunnelling term
wq = 1.0  # Energy of the 2-level system.
H0 = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

# Amplitude damping#
# Damping rate:
gamma = 0.1
L0 = liouvillian(H0, [np.sqrt(gamma) * Sm])

# sigma X control
LC_x = liouvillian(Sx)
# sigma Y control
LC_y = liouvillian(Sy)
# sigma Z control
LC_z = liouvillian(Sz)

# Drift
drift = L0
# Controls - different combinations can be tried
ctrls = [LC_z, LC_x]
# Number of ctrls
n_ctrls = len(ctrls)

# start point for the map evolution
E0 = sprepost(Si, Si)

# target for map evolution
E_targ = sprepost(had_gate, had_gate)
```

### 定义时间演化参数

```python
# Number of time slots
n_ts = 10
# Time allowed for the evolution
evo_time = 2
```

### 设定优化终止条件

```python
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 30
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
```

### 设定初始脉冲类型

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

```python
# Note that this call will take the defaults
#    dyn_type='GEN_MAT'
# This means that matrices that describe the dynamics are assumed to be
# general, i.e. the propagator can be calculated using:
# expm(combined_dynamics*dt)
#    prop_type='FRECHET'
# and the propagators and their gradients will be calculated using the
# Frechet method, i.e. an exact gradent
#    fid_type='TRACEDIFF'
# and that the fidelity error, i.e. distance from the target, is give
# by the trace of the difference between the target and evolved operators
result = cpo.optimize_pulse(
    drift,
    ctrls,
    E0,
    E_targ,
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

```python
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print(
    "Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time))
)
```

### 绘制初始与最终控制幅值

```python
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(
        result.time,
        np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
        where="post",
    )

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(
        result.time,
        np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
        where="post",
    )
fig1.tight_layout()
```

### 版本信息

```python
about()
```

```python

```
