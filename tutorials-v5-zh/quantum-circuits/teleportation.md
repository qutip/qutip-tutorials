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

# 量子隐形传态电路

```python
from math import sqrt

from qutip import about, basis, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Measurement
```

## 简介

本 notebook 介绍基础量子隐形传态电路（https://en.wikipedia.org/wiki/Quantum_teleportation），
包含测量与经典控制。
它同时演示如何在电路中添加测量门与经典控制逻辑。

我们将描述实现量子隐形传态的电路。
电路使用两条经典线和三条量子比特线。
第一条量子线表示需要从 Alice 传给 Bob 的量子态 $|q0\rangle = |\psi\rangle$
（因此第一个量子比特在 Alice 手中）。

```python
teleportation = QubitCircuit(
    3, num_cbits=2, input_states=[r"\psi", "0", "0", "c0", "c1"]
)
```

首先 Alice 与 Bob 需要在第二、第三个量子比特之间建立共享 EPR 对
（$\frac{|00\rangle+|11\rangle}{\sqrt{2}}$），
方法是在 Alice 的量子比特上施加 Hadamard 门，然后施加纠缠 CNOT 门。

```python
teleportation.add_gate("SNOT", targets=[1])
teleportation.add_gate("CNOT", targets=[2], controls=[1])
```

接着，Alice 让量子比特 $|q0\rangle$ 与其 EPR 量子比特相互作用，
随后对 Alice 所持有的两个量子比特测量。
第一个量子比特测量结果存入经典寄存器 $c1$，
第二个量子比特测量结果存入经典寄存器 $c0$。

```python
teleportation.add_gate("CNOT", targets=[1], controls=[0])
teleportation.add_gate("SNOT", targets=[0])

teleportation.add_measurement("M0", targets=[0], classical_store=1)
teleportation.add_measurement("M1", targets=[1], classical_store=0)
```

现在根据经典控制位对 Bob 的量子比特施加操作：
由 $c0$ 控制施加 $X$ 门，由 $c1$ 控制施加 $Z$ 门。
其对应关系如下：

$|00\rangle\rightarrow $ 无操作 \
$|01\rangle\rightarrow Z$ \
$|10\rangle\rightarrow X$ \
$|11\rangle\rightarrow ZX$ 

最终电路应使第三个量子比特处于态 $|\psi\rangle$。

```python
teleportation.add_gate("X", targets=[2], classical_controls=[0])
teleportation.add_gate("Z", targets=[2], classical_controls=[1])
```

至此隐形传态电路已构建完成。可用以下命令查看门结构。

```python
teleportation.gates
```

也可直接可视化电路：

```python
teleportation
```

其中第一个量子比特是用户指定的 $|\psi\rangle$，其余两个应为 $|0\rangle$。

### 示例 1
#### $|\psi\rangle= |+\rangle$

```python
a = 1 / sqrt(2) * basis(2, 0) + 1 / sqrt(2) * basis(2, 1)
state = tensor(a, basis(2, 0), basis(2, 0))
```

我们先查看第一个量子比特的测量统计以确认初态设置正确，再运行电路。

```python
initial_measurement = Measurement("start", targets=[0])
initial_measurement.measurement_comp_basis(state)
```

可使用 `QubitCircuit.run()` 运行电路：
给定初始态向量（或密度矩阵）后，函数会执行一次完整电路（包含中间测量采样），
并返回最终结果（也可通过参数 `cbits` 显式设置经典位）。
返回值为 `Result` 对象；
可通过 `get_states()` 访问结果态，其中 `index=0` 表示取第一个（本例也是唯一一个）结果。

```python
state_final = teleportation.run(state)
print(state_final)
```

运行后，查看最后一个量子比特测量统计，确认态已正确传送。

```python
final_measurement = Measurement("start", targets=[2])
final_measurement.measurement_comp_basis(state_final)
```

### 示例 2
#### $|\psi\rangle= |1\rangle$

```python
state = tensor(basis(2, 1), basis(2, 0), basis(2, 0))
initial_measurement = Measurement("start", targets=[0])
initial_measurement.measurement_comp_basis(state)
```

```python
state_final = teleportation.run(state)
print(state_final)
```

```python
final_measurement = Measurement("start", targets=[2])
final_measurement.measurement_comp_basis(state_final)
```

电路模块另一项实用功能是 **QubitCircuit.run_statistics()**，
可返回电路所有可能输出态及其概率。
同样返回 `Result` 对象。
可通过 `get_results()` 获取结果态与对应概率。

```python
results = teleportation.run_statistics(state)
results.probabilities
```

```python
results.final_states
```

```python
about()
```
