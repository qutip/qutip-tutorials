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

# QASM 电路的导入与导出

Notebook Author: Sidhant Saraogi(sid1397@gmail.com)


本 notebook 介绍 [OpenQASM](https://github.com/Qiskit/openqasm) 的导入与导出功能，
也可作为 QASM 格式的简短入门。
量子汇编语言（QASM）可作为量子电路的中间表示。
这是 QuTiP 进行电路导入/导出的一种方式，
从而使 QuTiP 的 QIP 模块与 Qiskit、Cirq 更容易兼容。

```python
import numpy as np
from qutip import about, basis, rand_ket, tensor
from qutip_qip.operations import Measurement
from qutip_qip.operations.gates import gate_sequence_product
from qutip_qip.qasm import read_qasm, print_qasm
```

流程很简单：用户只需将 `.qasm` 文件保存在合适位置，
并维护该文件的绝对路径，即可方便读取。
本示例中，`qasm_files` 目录已预置一些 qasm 电路示例。
更多示例可见 [OpenQASM 仓库](https://github.com/Qiskit/openqasm)。
下面先读取其中一个示例：

```python
path = "qasm_files/swap.qasm"
qasm_file = open(path, "r")
print(qasm_file.read())
```

## QASM 导入


该 QASM 文件以 QASM 格式描述了 QuTiP 原生 SWAP 门。
导入时使用 `read_qasm`，参数包括文件路径、`mode`（默认 `"qiskit"`）与
`version`（默认 `"2.0"`）。

我们可以通过检查该电路对应的幺正矩阵来验证其确实实现 SWAP 门。
这可以借助 `QubitCircuit` 的 `propagators` 方法配合
`gate_sequence_product` 完成。

```python
qc = read_qasm(path, mode="qiskit", version="2.0")
gate_sequence_product(qc.propagators())
```

`mode` 表示 QuTiP 内部处理 QASM 文件的方式。
在 `"qiskit"` 模式下，QASM 解析会跳过 `qelib1.inc` 的 include 命令，
并将其中定义的自定义门直接映射为 QuTiP 门，而不解析其 gate 定义。

**注意**：`qelib1.inc` 是包含若干 QASM 门定义的“头文件”。
它是 OpenQASM 仓库中的标准文件，
并会由 QuTiP（以及 Qiskit/Cirq）在导出 QASM 时包含。

`version` 表示当前处理的 OpenQASM 标准版本。
对应文档见 [OpenQASM](https://github.com/Qiskit/openqasm) 仓库。
目前仅支持 OpenQASM 2.0（也是最常用版本）。


### QASM 导出

我们也可以将 `QubitCircuit` 转换为 QASM 格式。
这在将量子电路导出到 Qiskit、Cirq 等其他量子软件包时非常有用。
目前可用三种输出方式：`print_qasm`、`str_qasm`、`write_qasm`。

```python
print_qasm(qc)
```

### 自定义门

QASM 支持使用 `gate` 关键字将已定义门组合成自定义门。
在 `"qiskit"` 模式下，可默认认为解释器支持
OpenQASM 仓库中的 [stdgates.inc](https://github.com/openqasm/openqasm/blob/main/examples/stdgates.inc)
所定义的所有门。

在 `swap_custom.qasm` 中，我们用预定义的 `cx` 门定义了 `swap` 门。

```python
path = "qasm_files/swap_custom.qasm"
qasm_file = open(path, "r")
print(qasm_file.read())
```

此外，该电路还测量了两个量子比特 `q[0]` 和 `q[1]`，
并将结果分别存入经典寄存器 `c[0]` 和 `c[1]`。

```python
qc = read_qasm(path)
```

现在可以运行电路，确认其被正确加载并执行了正确操作。
可使用 `QubitCircuit.run` 并提供合适输入态。
这里取输入态 `|01\rangle`。

```python
qc.run(tensor(basis(2, 0), basis(2, 1)))
```

如预期，输出为交换后的态 `|10\rangle`。


### 测量与经典控制

QASM 也支持测量及由经典比特控制量子门等特性。
QuTiP 同样支持。
示例可参考量子隐形传态电路。
更完整解释见[量子隐形传态 notebook](teleportation.md)。

```python
path = "qasm_files/teleportation.qasm"
qasm_file = open(path, "r")
qasm_str = qasm_file.read()
print(qasm_str)
```

 还可以通过 `read_qasm(..., strmode=True)` 直接从字符串读取 QASM。

```python
teleportation = read_qasm(qasm_str, strmode=True)
```

**注意**：
上面的 warning 是预期行为，用于提醒：
从 QASM 导入到 QuTiP 时，不会保留不同量子/经典寄存器的原始命名信息。
若用户希望导出后保持命名一致，这可能带来影响。


我们可快速验证隐形传态电路是否正确：将第一个量子比特态传送到第三个量子比特。

```python
state = tensor(rand_ket(2), basis(2, 0), basis(2, 0))

initial_measurement = Measurement("start", targets=[0])
_, initial_probabilities = initial_measurement.measurement_comp_basis(state)

state_final = teleportation.run(state)

final_measurement = Measurement("start", targets=[2])
_, final_probabilities = final_measurement.measurement_comp_basis(state_final)

np.testing.assert_allclose(initial_probabilities, final_probabilities)
```

**注意**：QASM 导入的自定义门通常无法直接再导出。
目前仅支持导出 QuTiP 原生定义的门。
对于 `qelib1.inc` 里未提供但 QuTiP 原生支持的门，
QuTiP 会把对应门定义直接写入导出的 `.qasm` 文件。
导出支持门与测量，但暂不支持受控门的导出。

```python
about()
```
