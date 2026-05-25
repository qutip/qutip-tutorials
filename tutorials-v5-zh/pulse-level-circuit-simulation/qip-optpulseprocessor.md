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

# OptPulseProcessor 示例


Author: Boxi Li (etamin1201@gmail.com)

```python
from numpy import pi
from qutip import about, basis, fidelity, identity, sigmax, sigmaz, tensor
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import OptPulseProcessor
from qutip_qip.operations import expand_operator, toffoli
```

`qutip.OptPulseProcessor` 是一个带噪量子设备模拟器，
并集成了 `qutip.control` 模块中的最优脉冲算法。
它是 `qutip.Processor` 的子类，
可为 `qutip.QubitCircuit` 或 `qutip.Qobj` 列表寻找最优脉冲序列
（这也是其名为 `OptPulseProcessor` 的原因）。
关于 `qutip.Processor` 的使用说明，请参考
[入门指南](https://qutip-qip.readthedocs.io/en/stable/qip-processor.html)。

## 单量子比特门
与父类 `Processor` 一样，先定义系统可用哈密顿量。
`OptPulseProcessor` 额外有一个漂移哈密顿量参数：
其系数不随时间变化，因此不参与优化。

```python
num_qubits = 1
# Drift Hamiltonian
H_d = sigmaz()
# The (single) control Hamiltonian
H_c = sigmax()
processor = OptPulseProcessor(num_qubits, drift=H_d)
processor.add_control(H_c, 0)
```

`load_circuit` 方法会调用 `qutip.control.optimize_pulse_unitary`，
并返回脉冲系数。

```python
qc = QubitCircuit(num_qubits)
qc.add_gate("SNOT", 0)

# This method calls optimize_pulse_unitary
tlist, coeffs = processor.load_circuit(
    qc, min_grad=1e-20, init_pulse_type="RND", num_tslots=6,
    evo_time=1, verbose=True
)
processor.plot_pulses(
    title="Control pulse for the Hadamard gate", use_control_latex=False
);
```

与 `Processor` 一样，仿真由 QuTiP 求解器完成。
`run_state` 会调用 `mesolve` 并返回结果。
也可加入噪声观察保真度变化，例如加入 `t1` 退相干时间。

```python
rho0 = basis(2, 1)
plus = (basis(2, 0) + basis(2, 1)).unit()
minus = (basis(2, 0) - basis(2, 1)).unit()
result = processor.run_state(init_state=rho0)
print("Fidelity:", fidelity(result.states[-1], minus))

# add noise
processor.t1 = 40.0
result = processor.run_state(init_state=rho0)
print("Fidelity with qubit relaxation:", fidelity(result.states[-1], minus))
```

## 多量子比特门


下面示例中，我们用 `OptPulseProcessor` 为一个多比特电路寻找最优控制脉冲。
为简化起见，电路只包含一个 Toffoli 门。

```python
toffoli()
```

我们使用单比特控制 $\sigma_x$ 与 $\sigma_z$。
参数 `cyclic_permutation=True` 会创建 3 个算符，分别作用于 3 个比特。

```python
N = 3
H_d = tensor([identity(2)] * 3)
test_processor = OptPulseProcessor(N, H_d)
test_processor.add_control(sigmaz(), cyclic_permutation=True)
test_processor.add_control(sigmax(), cyclic_permutation=True)
```

比特间相互作用由量子比特 0&1 以及 1&2 之间的 $\sigma_x\sigma_x$ 给出。
`expand_operator` 可将算符按指定目标比特扩展到更高维空间。

```python
sxsx = tensor([sigmax(), sigmax()])
sxsx01 = expand_operator(sxsx, 3, targets=[0, 1])
sxsx12 = expand_operator(sxsx, 3, targets=[1, 2])
test_processor.add_control(sxsx01)
test_processor.add_control(sxsx12)
```

使用上述控制哈密顿量后，我们在 6 个时间片上为 Toffoli 门寻找最优脉冲。
输入除 `QubitCircuit` 外，也可直接给算符列表。

```python
def get_control_latex():
    """
    Get the labels for each Hamiltonian.
    It is used in the method``plot_pulses``.
    It is a 2-d nested list, in the plot,
    a different color will be used for each sublist.
    """
    return [
        [r"$\sigma_z^%d$" % n for n in range(test_processor.num_qubits)],
        [r"$\sigma_x^%d$" % n for n in range(test_processor.num_qubits)],
        [r"$g_01$", r"$g_12$"],
    ]


test_processor.model.get_control_latex = get_control_latex
```

```python
test_processor.dims = [2, 2, 2]
```

```python
test_processor.load_circuit([toffoli()], num_tslots=6,
                            evo_time=1, verbose=True)

test_processor.plot_pulses(title="Contorl pulse for toffoli gate");
```

## 合并量子电路
若电路包含多个门，可选择先合并为一个总体幺正，再寻找合并后的最优脉冲。

```python
qc = QubitCircuit(3)
qc.add_gate("CNOT", controls=0, targets=2)
qc.add_gate("RX", targets=2, arg_value=pi / 4)
qc.add_gate("RY", targets=1, arg_value=pi / 8)
```

```python
setting_args = {
    "CNOT": {"num_tslots": 20, "evo_time": 3},
    "RX": {"num_tslots": 2, "evo_time": 1},
    "RY": {"num_tslots": 2, "evo_time": 1},
}

test_processor.load_circuit(
    qc, merge_gates=False, setting_args=setting_args, verbose=True
)
fig, axes = test_processor.plot_pulses(
    title="Control pulse for a each gate in the circuit", show_axis=True
)
axes[-1].set_xlabel("time");
```

上图中，$t=0$ 到 $t=3$ 的脉冲对应 CNOT 门，
后续对应两个单比特门。
脉冲变化频率差异仅来自我们设置的 `evo_time`。
可见三个门是按顺序执行的。

```python
qc = QubitCircuit(3)
qc.add_gate("CNOT", controls=0, targets=2)
qc.add_gate("RX", targets=2, arg_value=pi / 4)
qc.add_gate("RY", targets=1, arg_value=pi / 8)
test_processor.load_circuit(
    qc, merge_gates=True, verbose=True, num_tslots=20, evo_time=5
)
test_processor.plot_pulses(title="Control pulse for a \
                           merged unitary evolution");
```

在这张图中不再有分阶段结构：
三个门先合并，再由算法为合并后的幺正演化寻找最优脉冲。

```python
about()
```
