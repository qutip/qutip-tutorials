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

# 使用 L-BFGS-B 算法计算双量子比特 QFT 门控制场


Alexander Pitchford (agp1@aber.ac.uk)


本示例演示如何通过 `ctrlpulseoptim.create_pulse_optimizer`
创建 `Optimizer` 对象并使用控制库求解控制脉冲。
该接口允许在运行优化前先修改配置。
本例展示：
1) 自定义初始控制脉冲；
2) 复用对象进行不同总演化时间的重复优化。

优化器采用（默认）L-BFGS-B 算法，
目标是最小化保真度误差（等价于最大化保真度至 1）。

本例系统：两个量子比特在 x/y/z 恒定场中，
并在每个比特上施加独立 x/y 控制场。
目标演化是 QFT 门。
可实验以下选项：
 * 演化时间（`evo_times`）
 * 相位选项（`phase_option = SU/PSU`）
 * 传播子计算类型（`prop_type = DIAG/FRECHET`）
 * 保真度度量（`fid_type = UNIT/TRACEDIFF`）

可通过修改时间片时长探索不同离散精度，
并尝试不同初始脉冲类型。
示例会绘制初始与最终脉冲。

该示例假定你已运行 Hadamard 示例，因此不重复基础概念说明。

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulsegen as pulsegen

import qutip_qtrl.pulseoptim as cpo
from qutip import identity, sigmax, sigmay, sigmaz, tensor, about
from qutip_qip.algorithms import qft

example_name = "QFT"

%matplotlib inline
```

### 定义物理模型


注意本例中每个量子比特上都有两个控制通道。

```python
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5 * identity(2)

# Drift Hamiltonian
H_d = 0.5 * (tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
# The (four) control Hamiltonians
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
n_ctrls = len(H_c)
# start point for the gate evolution
U_0 = identity(4)
# Target for the gate evolution - Quantum Fourier Transform gate
U_targ = qft(2)
```

### 定义时间演化参数


本例会尝试多个总演化时间。
按这种方式可迭代估计达到目标保真度所需最短演化时间。

注意这里固定了时间片宽度 `dt`，
因此时间片数会随 `evo_time` 改变。

```python
# Duration of each timeslot
dt = 0.05
# List of evolution times to try
evo_times = [1, 3, 6]
n_evo_times = len(evo_times)
evo_time = evo_times[0]
n_ts = int(float(evo_time) / dt)
# Empty list that will hold the results for each evolution time
results = list()
```

### 设定优化终止条件

```python
# Fidelity error target
fid_err_targ = 1e-5
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
```

### 设定初始脉冲类型


这里使用线性初始脉冲，主要因为通常可得到较平滑最终脉冲。

```python
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = "LIN"
```

### 设置输出文件扩展名

```python
# Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
```

### 创建优化对象


这里与 Hadamard 示例的主要区别是：
使用另一个 pulseoptim 函数先创建对象，
再手动配置物理模型和优化算法。
这种方式灵活性更高（本例用于给不同控制设置不同初始脉冲参数），
且在同一系统上重复多次优化时更高效。

```python
optim = cpo.create_pulse_optimizer(
    H_d,
    H_c,
    U_0,
    U_targ,
    n_ts,
    evo_time,
    amp_lbound=-5.0,
    amp_ubound=5.0,
    fid_err_targ=fid_err_targ,
    min_grad=min_grad,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    optim_method="fmin_l_bfgs_b",
    method_params={"max_metric_corr": 20, "accuracy_factor": 1e8},
    dyn_type="UNIT",
    fid_params={"phase_option": "PSU"},
    init_pulse_type=p_type,
    gen_stats=True,
)

# **** get handles to the other objects ****
optim.test_out_files = 0
dyn = optim.dynamics
dyn.test_out_files = 0
p_gen = optim.pulse_generator
```

### 对不同总演化时间执行优化


下面循环遍历 `evo_times`。
第一轮使用对象创建时传入的时间片参数。
后续轮次则通过 `dyn` 动态重设时间片参数，
然后重新生成初始脉冲并运行优化。

注意：采用这种方式时，
在 `optim.run_optimization()` 前必须调用
`dyn.initialize_controls(init_amps)`。

```python
for i in range(n_evo_times):
    # Generate the tau (duration) and time (cumulative) arrays
    # so that it can be used to create the pulse generator
    # with matching timeslots
    dyn.init_timeslots()
    if i > 0:
        # Create a new pulse generator for the new dynamics
        p_gen = pulsegen.create_pulse_gen(p_type, dyn)

    # Generate different initial pulses for each of the controls
    init_amps = np.zeros([n_ts, n_ctrls])
    if p_gen.periodic:
        phase_diff = np.pi / n_ctrls
        for j in range(n_ctrls):
            init_amps[:, j] = p_gen.gen_pulse(start_phase=phase_diff * j)
    elif isinstance(p_gen, pulsegen.PulseGenLinear):
        for j in range(n_ctrls):
            p_gen.scaling = float(j) - float(n_ctrls - 1) / 2
            init_amps[:, j] = p_gen.gen_pulse()
    elif isinstance(p_gen, pulsegen.PulseGenZero):
        for j in range(n_ctrls):
            p_gen.offset = sf = float(j) - float(n_ctrls - 1) / 2
            init_amps[:, j] = p_gen.gen_pulse()
    else:
        # Should be random pulse
        for j in range(n_ctrls):
            init_amps[:, j] = p_gen.gen_pulse()

    dyn.initialize_controls(init_amps)

    # Save initial amplitudes to a text file
    if f_ext is not None:
        pulsefile = "ctrl_amps_initial_" + f_ext
        dyn.save_amps(pulsefile)
        print("Initial amplitudes output to file: " + pulsefile)

    print("***********************************")
    print("\n+++++++++++++++++++++++++++++++++++")
    print("Starting pulse optimisation for T={}".format(evo_time))
    print("+++++++++++++++++++++++++++++++++++\n")
    result = optim.run_optimization()
    results.append(result)

    # Save final amplitudes to a text file
    if f_ext is not None:
        pulsefile = "ctrl_amps_final_" + f_ext
        dyn.save_amps(pulsefile)
        print("Final amplitudes output to file: " + pulsefile)

    # Report the results
    result.stats.report()
    print("Final evolution\n{}\n".format(result.evo_full_final))
    print("********* Summary *****************")
    print("Final fidelity error {}".format(result.fid_err))
    print("Final gradient normal {}".format(result.grad_norm_final))
    print("Terminated due to {}".format(result.termination_reason))
    print("Number of iterations {}".format(result.num_iter))
    print(
        "Completed in {} HH:MM:SS.US".format(
            datetime.timedelta(seconds=result.wall_time)
        )
    )

    if i + 1 < len(evo_times):
        # reconfigure the dynamics for the next evo time
        evo_time = evo_times[i + 1]
        n_ts = int(float(evo_time) / dt)
        dyn.tau = None
        dyn.evo_time = evo_time
        dyn.num_tslots = n_ts
```

### 绘制初始与最终幅值

```python
fig1 = plt.figure(figsize=(12, 8))
for i in range(n_evo_times):
    # Initial amps
    ax1 = fig1.add_subplot(2, n_evo_times, i + 1)
    ax1.set_title("Init amps T={}".format(evo_times[i]))
    # ax1.set_xlabel("Time")
    ax1.get_xaxis().set_visible(False)
    if i == 0:
        ax1.set_ylabel("Control amplitude")
    for j in range(n_ctrls):
        ax1.step(
            results[i].time,
            np.hstack((results[i].initial_amps[:, j], results[i].initial_amps[-1, j])),
            where="post",
        )

    ax2 = fig1.add_subplot(2, n_evo_times, i + n_evo_times + 1)
    ax2.set_title("Final amps T={}".format(evo_times[i]))
    ax2.set_xlabel("Time")
    # Optimised amps
    if i == 0:
        ax2.set_ylabel("Control amplitude")
    for j in range(n_ctrls):
        ax2.step(
            results[i].time,
            np.hstack((results[i].final_amps[:, j], results[i].final_amps[-1, j])),
            where="post",
        )

plt.tight_layout()
plt.show()
```

### 版本信息

```python
about()
```

```python

```
