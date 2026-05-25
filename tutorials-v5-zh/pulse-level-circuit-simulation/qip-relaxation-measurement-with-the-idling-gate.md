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

# 使用空闲门测量弛豫时间
为了演示退相干噪声的仿真，我们构建一个示例：
把 Ramsey 实验表示为量子电路，并在带噪声的 `Processor` 上运行。
Ramsey 实验流程为：量子比特先初始化到激发态，
然后绕 $x$ 轴做一次 $\pi/2$ 旋转，空闲演化时间为 $t$，
最后再做一次 $\pi/2$ 旋转后测量：

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy
from qutip import about, basis, sigmax, sigmaz
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import LinearSpinChain

pi = np.pi
num_samples = 500
amp = 0.1
f = 0.5
t2 = 10 / f

# Define a processor.
proc = LinearSpinChain(num_qubits=1, sx=amp / 2, t2=t2)
ham_idle = 2 * pi * sigmaz() / 2 * f
resonant_sx = 2 * pi * sigmax() - ham_idle / (amp / 2)
proc.add_drift(ham_idle, targets=0)
proc.add_control(resonant_sx, targets=0, label="sx0")


# Define a Ramsey experiment.
def ramsey(t, proc):
    qc = QubitCircuit(1)
    qc.add_gate("RX", 0, arg_value=pi / 2)
    qc.add_gate("IDLE", 0, arg_value=t)
    qc.add_gate("RX", 0, arg_value=pi / 2)
    proc.load_circuit(qc)
    result = proc.run_state(init_state=basis(2, 0), e_ops=sigmaz())
    return result.expect[0][-1]


idle_tlist = np.linspace(0.0, 30.0, num_samples)
measurements = np.asarray([ramsey(t, proc) for t in idle_tlist])

rx_gate_time = 1 / 4 / amp  # pi/2
total_time = 2 * rx_gate_time + idle_tlist[-1]
tlist = np.linspace(0.0, total_time, num_samples)

peak_ind = scipy.signal.find_peaks(measurements)[0]


def decay_func(t, t2, f0):
    return f0 * np.exp(-1.0 / t2 * t)


(t2_fit, f0_fit), _ = scipy.optimize.curve_fit(
    decay_func, idle_tlist[peak_ind], measurements[peak_ind]
)
print("T2:", t2)
print("Fitted T2:", t2_fit)

fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
ax.plot(idle_tlist[:], measurements[:], "-", label="Simulation",
        color="slategray")
ax.plot(
    idle_tlist,
    decay_func(idle_tlist, t2_fit, f0_fit),
    "--",
    label="Theory",
    color="slategray",
)
ax.set_xlabel(r"Idling time $t$ [$\mu$s]")
ax.set_ylabel("Ramsey signal", labelpad=2)
ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1]))
ax.set_position([0.18, 0.2, 0.75, 0.75])
ax.grid()
```

在上面的代码块中，我们使用线性自旋链处理器主要是为了调用其编译器，
并未使用其默认哈密顿量。
我们改为手动定义：一个始终开启的漂移哈密顿量 $\sigma^z$（频率 $f=0.5$ MHz）、
一个共振的 $\sigma^x$ 驱动（幅度 $0.1/2$ MHz），以及相干时间 $T_2=10/f$。
对不同空闲时间 $t$，记录可观测量 $\sigma^z$ 的期望值，得到实线曲线。
如预期，其包络服从由 $T_2$ 表征的指数衰减（虚线）。
注意，由于 $\pi/2$ 脉冲是作为实际物理过程仿真的，拟合衰减并非从 1 开始。
这展示了如何在仿真中纳入态制备误差。

```python
about()
```
