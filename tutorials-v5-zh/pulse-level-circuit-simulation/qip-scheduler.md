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

# 量子门与指令调度器

Author: Boxi Li (etamin1201@gmail.com)

物理量子比特有限的相干时间是限制量子计算性能的主要因素之一。
原则上，电路执行时间应远短于相干时间。
缩短执行时间的一种方式是让电路中的多个门并行执行。
例如在 [Grover 算法](https://en.wikipedia.org/wiki/Grover%27s_algorithm) 中，
若同一个单比特门作用于所有量子比特，并行执行可节省大量时间。

量子计算机中的调度器与经典计算机类似：
其目标是安排量子电路执行顺序以最小化总执行时间。
一个简单规则是：若两个门不作用于同一量子比特，则可并行执行。
当然也可加入更复杂硬件约束，本例仅考虑这个基本准则。
调度问题的非平凡之处在于，它必须考虑可交换量子门的可能置换。
因此，在满足硬件约束下探索门置换空间，是调度器的核心难点。

我们先展示如何使用 `qutip_qip` 内置工具调度量子电路门操作，
随后展示编译后控制脉冲的调度。
最后给出一个可交换门置换会影响调度结果的简单示例及处理方法。

```python
# imports
from qutip import about
from qutip_qip.circuit import QubitCircuit
from qutip_qip.compiler import Instruction, Scheduler
from qutip_qip.device import LinearSpinChain
```

## 门级调度
先定义一个量子电路。

```python
circuit = QubitCircuit(3)
circuit.add_gate("X", 0)
circuit.add_gate("ISWAP", targets=[1, 2])
circuit.add_gate("X", 2)
circuit.add_gate("Y", 0)
circuit.add_gate("X", 0)
circuit
```

这个电路本身并不复杂，但适合用于演示调度器。
现在定义调度器并对电路门执行进行调度。

```python
scheduler = Scheduler("ASAP")  # schedule as soon as possible
scheduled_time = scheduler.schedule(circuit)
scheduled_time
```

结果给出了每个门的调度起始时间。
第一轮执行在量子比特 0 和 1 上的 iSWAP，以及量子比特 0 上的 X 门；
第二轮执行量子比特 2 上的 X 门和量子比特 0 上的 Y 门；
最后一轮执行量子比特 0 上的 X 门。如下打印：

```python
cycle_list = [[] for i in range(max(scheduled_time) + 1)]

for i, time in enumerate(scheduled_time):
    gate = circuit.gates[i]
    cycle_list[time].append(gate.name + str(gate.targets))
for cycle in cycle_list:
    print(cycle)
```

我们也可以按“尽量晚执行”（ALAP）规则调度。

```python
scheduler = Scheduler("ALAP")  # schedule as late as possible
scheduled_time = scheduler.schedule(circuit)
cycle_list = [[] for i in range(max(scheduled_time) + 1)]
for i, time in enumerate(scheduled_time):
    gate = circuit.gates[i]
    cycle_list[time].append(gate.name + str(gate.targets))
for cycle in cycle_list:
    print(cycle)
```

区别在于 `iSWAP` 门和量子比特 2 上的 X 门整体后移了一个周期。


## 指令/脉冲调度
实际中，不同量子门执行时长往往不同。
为考虑这一点，我们定义量子指令列表：其中 X 门时长设为 1，
iSWAP 门时长设为 3.5。

```python
scheduler = Scheduler("ASAP")
instructions = []

for gate in circuit.gates:
    if gate.name in ("X"):
        duration = 1
    elif gate.name == "ISWAP":
        duration = 3.5
    instruction = Instruction(gate, duration=duration)
    instructions.append(instruction)
scheduler.schedule(instructions)
```

此时每个门的执行时间不再能简单映射为离散门周期。
但可通过 qutip 的[含噪电路模拟器](https://qutip-qip.readthedocs.io/en/stable/qip-processor.html)
观察编译后的控制信号。
（注意执行时间遵循自旋链硬件参数，且 Y 门会分解为 Z-X-Z 旋转。）

```python
device = LinearSpinChain(3)
device.load_circuit(
    circuit, "ASAP"
)  # The circuit are compiled to instructions and scheduled.
device.plot_pulses();
```

绿色与橙色脉冲分别代表绕 X 轴与 Z 轴旋转。
绿色脉冲对应 iSWAP 门，它与量子比特 0 上若干单比特旋转同时执行。


## 可交换门的处理
考虑下面电路：

```python
circuit = QubitCircuit(3)
circuit.add_gate("SNOT", 0)
circuit.add_gate("CNOT", 1, 0)
circuit.add_gate("CNOT", 2, 0)
circuit.add_gate("SNOT", 2)
circuit
```

乍看之下似乎无法并行。
但两个 CNOT 实际可交换；若置换顺序，就可让其中一个 CNOT 与最后一个 Hadamard 门并行。

```python
scheduler = Scheduler("ALAP")
scheduled_time = scheduler.schedule(circuit)

cycle_list = [[] for i in range(max(scheduled_time) + 1)]
for i, time in enumerate(scheduled_time):
    gate = circuit.gates[i]
    cycle_list[time].append(gate.name + str(gate.targets))
for cycle in cycle_list:
    print(cycle)
```

## 随机打乱
调度算法属于启发式方法，因而不一定总能找到最优结果。
因此可通过参数 `random_shuffle` 与 `repeat_num` 在调度过程中引入随机性。

```python
about()
```
