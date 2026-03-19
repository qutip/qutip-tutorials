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

# 在脉冲层面模拟 Deutsch-Jozsa 算法


Author: Boxi Li (etamin1201@gmail.com)


本示例演示如何用 QuTiP 在量子硬件模型上模拟简单量子算法。
模拟器由 `Processor`（及其子类）定义。
`Processor` 可表示一般量子设备，
其量子系统（如量子比特）间相互作用由控制哈密顿量定义。
脉冲层仿真的总体介绍请见
[用户指南](https://qutip-qip.readthedocs.io/en/stable/qip-processor.html)。

下面我们将一个简单三比特量子电路编译为不同哈密顿量模型上的控制脉冲。

## Deutsch-Jozsa 算法
Deutsch-Jozsa 算法是最简单且相较经典算法具指数加速的量子算法之一。
其假设存在函数 $f:\{0,1\}^n \rightarrow \{0,1\}$，
并且该函数要么是 balanced，要么是 constant。
constant 指对所有输入都有相同输出（全 0 或全 1）；
balanced 指输入域中一半输出为 1、另一半为 0。
更严格定义见 https://en.wikipedia.org/wiki/Deutsch-Jozsa_algorithm。

Deutsch-Jozsa 电路包含 $n$ 个输入比特和 1 个初始为态 1 的辅助比特。
算法末尾在计算基上测量前 $n$ 个比特。
若函数是 constant，$n$ 个比特测量结果全为 0；
若是 balanced，则永远不会测得 $\left|00...0\right\rangle$。

以下示例实现 balanced 函数
$f:\{00,01,10,11\} \rightarrow \{0,1\}$，
其中 $f(00)=f(11)=0$，$f(01)=f(10)=1$。
因此应有测得 $\left|00\right\rangle$ 的概率为 0。

```python
import numpy as np
from qutip import about, basis, ptrace
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import (LinearSpinChain, OptPulseProcessor, SCQubits,
                              SpinChainModel)
```

```python
qc = QubitCircuit(num_qubits=3)
qc.add_gate("X", targets=2)
qc.add_gate("SNOT", targets=0)
qc.add_gate("SNOT", targets=1)
qc.add_gate("SNOT", targets=2)

# function f(x)
qc.add_gate("CNOT", controls=0, targets=2)
qc.add_gate("CNOT", controls=1, targets=2)

qc.add_gate("SNOT", targets=0)
qc.add_gate("SNOT", targets=1)
```

```python
qc
```

## 使用自旋链模型
首先使用 `LinearSpinChain` 哈密顿量模型仿真该电路。
控制哈密顿量定义见
[`SpinChainModel`](https://qutip-qip.readthedocs.io/en/stable/apidoc/qutip_qip.device.html#qutip_qip.device.SpinChainModel)。

```python
processor = LinearSpinChain(3)
processor.load_circuit(qc);
```

为快速可视化脉冲，`Processor` 提供 `plot_pulses` 方法。
下图中每种颜色代表系统某个控制哈密顿量随时间的脉冲序列。
每个时间区间内脉冲保持常值。

```python
processor.plot_pulses(title="Control pulse of Spin chain",
                      figsize=(8, 4), dpi=100);
```

由于自旋链模型只允许相邻比特相互作用，
会在第一个 CNOT 前后加入 SWAP 门来交换前两个比特。
SWAP 门分解为三个 iSWAP，
CNOT 分解为两个 iSWAP 加额外单比特修正。
Hadamard 与双比特门都需要分解到本征门集
（iSWAP 与绕 $x,z$ 轴旋转）。
编译得到的是方波脉冲，且 $\sigma_z$ 与 $\sigma_x$ 控制系数不同，
因此门时长也不同。


### 无退相干

```python
basis00 = basis([2, 2], [0, 0])
psi0 = basis([2, 2, 2], [0, 0, 0])
result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)))
```

### 有退相干

```python
processor.t1 = 100
processor.t2 = 30
psi0 = basis([2, 2, 2], [0, 0, 0])
result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)))
```

## 使用最优控制模块
该能力集成在子类 `OptPulseProcessor` 中，
它调用最优控制模块方法为目标门寻找最优脉冲序列。
可对整体幺正演化优化，也可逐门优化。
这里选择逐门优化。

```python
setting_args = {
    "SNOT": {"num_tslots": 6, "evo_time": 2},
    "X": {"num_tslots": 1, "evo_time": 0.5},
    "CNOT": {"num_tslots": 12, "evo_time": 5},
}
# Use the control Hamiltonians of the spin chain model.
processor = OptPulseProcessor(
    num_qubits=3, model=SpinChainModel(3, setup="linear")
)
processor.load_circuit(  # Provide parameters for the algorithm
    qc,
    setting_args=setting_args,
    merge_gates=False,
    verbose=True,
    amp_ubound=5,
    amp_lbound=0,
);
```

```python
processor.plot_pulses(
    title="Control pulse of OptPulseProcessor", figsize=(8, 4), dpi=100
);
```

在该最优控制模型中，我们使用 GRAPE 算法，
其中控制脉冲为分段常数函数。
我们给 GRAPE 提供与自旋链模型相同的控制哈密顿量。
在编译得到的最优脉冲中，大多数执行时间里各控制通道都处于激活状态
（脉冲幅度非零）。
还可观察到：即便是不同比特上的同类门（如 Hadamard），
其优化脉冲也不相同，说明最优解并不唯一，
并且仍可加入额外硬件约束进一步筛选解。


### 无退相干

```python
basis00 = basis([2, 2], [0, 0])
psi0 = basis([2, 2, 2], [0, 0, 0])
result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)))
```

### 有退相干

```python
processor.t1 = 100
processor.t2 = 30
psi0 = basis([2, 2, 2], [0, 0, 0])
result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)))
```

可见在含噪演化下，测得态 00 的概率不再严格为 0。


## 使用超导量子比特模型
下面使用子类 `SCQubits` 仿真同一电路。
它会根据超导量子比特系统可用哈密顿量寻找脉冲。
关于线性自旋链模型更多细节可参考
[spin chain notebook](../time-evolution/005_spin-chain.md)。

```python
processor = SCQubits(num_qubits=3)
processor.load_circuit(qc);
```

```python
processor.plot_pulses(title="Control pulse of SCQubits",
                      figsize=(8, 4), dpi=100);
```

对超导量子比特处理器，编译出的脉冲呈高斯形状。
这对超导量子比特很关键：其第二激发能级与计算子空间能级间隔较近，
平滑脉冲通常可降低泄漏到非计算子空间的概率。
和自旋链模型类似，会加入 SWAP 来交换第 0 与第 1 比特，
且一个 SWAP 会编译为三个 CNOT。
控制项 $ZX^{21}$ 不会被使用，
因为本电路中不存在“由第二比特控制、作用于第一比特”的 CNOT。


### 无退相干

```python
basis00 = basis([3, 3], [0, 0])
psi0 = basis([3, 3, 3], [0, 0, 0])
result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)))
```

### 有退相干

```python
processor.t1 = 50.0e3
processor.t2 = 20.0e3
psi0 = basis([3, 3, 3], [0, 0, 0])
result = processor.run_state(init_state=psi0)
print("Probability of measuring state 00:")
print(np.real((basis00.dag() * ptrace(result.states[-1], [0, 1]) * basis00)))
```

```python
about()
```

