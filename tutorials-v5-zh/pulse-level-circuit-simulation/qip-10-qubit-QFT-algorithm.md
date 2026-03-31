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

# 编译并仿真 10 量子比特量子傅里叶变换（QFT）算法

在本 notebook 中，我们仿真一个 10 量子比特的量子傅里叶变换（QFT）算法。
QFT 是量子计算中最重要的算法之一。
例如，它是用于整数分解的 Shor 算法的一部分。
下面的代码使用 CNOT 和单量子比特旋转定义一个 10 量子比特 QFT 电路，
并分别在门级和脉冲级进行仿真。

```python
from qutip import about, basis, fidelity
from qutip_qip.algorithms import qft_gate_sequence
from qutip_qip.device import LinearSpinChain

num_qubits = 10
# The QFT circuit
qc = qft_gate_sequence(num_qubits, swapping=False, to_cnot=True)
# Gate-level simulation
state1 = qc.run(basis([2] * num_qubits, [0] * num_qubits))
# Pulse-level simulation
processor = LinearSpinChain(num_qubits)
processor.load_circuit(qc)
state2 = processor.run_state(basis([2] * num_qubits,
                                   [0] * num_qubits)).states[-1]
fidelity(state1, state2)
```

我们在下面的单元中绘制编译后的脉冲。
图中的脉冲对应于自旋链模型本征门集合下实现的 QFT 算法：
单量子比特门标记为绕 $x$、$z$ 轴旋转，iSWAP 门通过自旋-自旋交换相互作用实现，记为 $g_i$。
单量子比特驱动项的正负号表示控制脉冲的相位；
而耦合强度 $g_i$ 上的负号仅是由相互作用定义约定（见 \cref{eq:ham spin chain}）导致的。

```python
def get_control_latex(model):
    """
    Get the labels for each Hamiltonian.
    It is used in the method method :meth:`.Processor.plot_pulses`.
    It is a 2-d nested list, in the plot,
    a different color will be used for each sublist.
    """
    num_qubits = model.num_qubits
    num_coupling = model._get_num_coupling()
    return [
        {f"sx{m}": r"$\sigma_x^{}$".format(m) for m in range(num_qubits)},
        {f"sz{m}": r"$\sigma_z^{}$".format(m) for m in range(num_qubits)},
        {f"g{m}": r"$g_{}$".format(m) for m in range(num_coupling)},
    ]


fig, axes = processor.plot_pulses(
    figsize=(5, 7), dpi=150, pulse_labels=get_control_latex(processor.model)
)
axes[-1].set_xlabel("$t$");
```

```python
about()
```
