---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: qutip-dev
    language: python
    name: python3
---

# QuTiP 示例：量子门及其使用方法


Author: Anubhav Vardhan (anubhavvardhan@gmail.com)

User-defined gate added by: Boxi Li (etamin1201@gmail.com)

QuTiP 更多信息见 [http://qutip.org](http://qutip.org)

#### 安装说明
电路图可视化需要 LaTeX 和 [ImageMagick](https://imagemagick.org/index.php)。模块会自动处理 LaTeX 代码、生成 pdf 并转成 png。
在 Mac 和 Linux 上，如果已安装 conda，可用 `conda install imagemagick` 快速安装 ImageMagick。
否则请参考 ImageMagick 官方文档安装说明。

Windows 上需要下载并安装 ImageMagick 安装包。
此外还需要 [perl](https://www.perl.org/get.html)（用于 pdfcrop）和 [Ghostscript](https://ghostscript.com/releases/index.html)（ImageMagick 转 png 的额外依赖）。

验证安装是否完成：在命令提示符中尝试以下三个命令是否可用：`pdflatex`、`pdfcrop` 和 `magick anypdf.pdf antpdf.png`（其中 `anypdf.pdf` 是任意 pdf 文件）。

```python
import numpy as np
from numpy import pi
from qutip import Qobj, about
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import (Gate, berkeley, cnot, cphase, csign, fredkin,
                                  gate_sequence_product, globalphase, iswap,
                                  molmer_sorensen, phasegate, qrot, rx, ry, rz,
                                  snot, sqrtiswap, sqrtnot, sqrtswap, swap,
                                  swapalpha, toffoli)

%matplotlib inline
```

## 引言


http://en.wikipedia.org/wiki/Quantum_gate



## QuTiP 中的量子门及其表示


### 受控相位门（Controlled-PHASE）

```python
cphase(pi / 2)
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("CSIGN", controls=[0], targets=[1])
q.draw()
```

### 绕 X 轴旋转

```python
rx(pi / 2)
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate("RX", targets=[0], arg_value=pi / 2, style={"showarg": True})
q.draw()
```

### 绕 Y 轴旋转

```python
ry(pi / 2)
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate("RY", targets=[0], arg_value=pi / 2, style={"showarg": True})
q.draw()
```

### 绕 Z 轴旋转

```python
rz(pi / 2)
```

```python
q = QubitCircuit(1, reverse_states=False)
q.add_gate("RZ", targets=[0], arg_value=pi / 2, style={"showarg": True})
q.draw()
```

### CNOT

```python
cnot()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("CNOT", controls=[0], targets=[1])
q.draw()
```

### CSIGN

```python
csign()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("CSIGN", controls=[0], targets=[1])
q.draw()
```

### Berkeley 门

```python
berkeley()
```

```python
q = QubitCircuit(2, reverse_states=False)
q.add_gate("BERKELEY", targets=[0, 1])
q.draw()
```

### SWAPalpha

```python
swapalpha(pi / 2)
```

### FREDKIN

```python
fredkin()
```

### TOFFOLI

```python
toffoli()
```

### SWAP

```python
swap()
q = QubitCircuit(2, reverse_states=False)
q.add_gate("SWAP", targets=[0, 1])
q.draw()
```

### ISWAP

```python
iswap()
q = QubitCircuit(2, reverse_states=False)
q.add_gate("ISWAP", targets=[0, 1])
q.draw()
```

### SQRTiSWAP

```python
sqrtiswap()
```

### SQRTSWAP

```python
sqrtswap()
```

### SQRTNOT

```python
sqrtnot()
```

### HADAMARD

```python
snot()
```

### PHASEGATE

```python
phasegate(pi / 2)
```

### GLOBALPHASE

```python
globalphase(pi / 2)
```

### Mölmer-Sörensen 门

```python
molmer_sorensen(pi / 2)
```

### 量子比特旋转门

```python
qrot(pi / 2, pi / 4)
```

### 将量子门扩展到更大寄存器


上面的示例展示了 QuTiP 中各量子门在最小比特数下的矩阵表示。
如果要在大小为 $N$ 的比特寄存器中表示同一量子门，可在调用门函数时传入可选关键字参数 `N`。
例如，生成 $N=3$ 比特寄存器上的 CNOT 矩阵：

```python
cnot(N=3)
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate("CNOT", controls=[1], targets=[2])
q.draw()
```

此外，控制位和目标位（适用时）也可通过关键字 `control` 和 `target`（有时是 `controls` 或 `targets`）指定：

```python
cnot(N=3, control=2, target=0)
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate("CNOT", controls=[0], targets=[2])
q.draw()
```

## 构建量子线路


QuTiP 实现的量子门可通过 `QubitCircuit` 类构造任意量子线路。输出可为幺正矩阵或 LaTeX 表示。


下面示例以 SWAP 门为例。已知 SWAP 可等价分解为三次 CNOT：

```python
N = 2
qc0 = QubitCircuit(N)
qc0.add_gate("ISWAP", [0, 1], None)
qc0.draw()
```

```python
U_list0 = qc0.propagators()
U0 = gate_sequence_product(U_list0)
U0
```

```python
qc1 = QubitCircuit(N)
qc1.add_gate("CNOT", 0, 1)
qc1.add_gate("CNOT", 1, 0)
qc1.add_gate("CNOT", 0, 1)
qc1.draw()
```

```python
U_list1 = qc1.propagators()
U1 = gate_sequence_product(U_list1)
U1
```

除了手动把 SWAP 转成 CNOT，也可用 `QubitCircuit` 内置函数自动转换。

```python
qc2 = qc0.resolve_gates("CNOT")
qc2.draw()
```

```python
U_list2 = qc2.propagators()
U2 = gate_sequence_product(U_list2)
U2
```

从 QuTiP 4.4 起，还支持在电路任意位置插入门。

```python
qc1.add_gate("CSIGN", index=[1], targets=[0], controls=[1])
qc1.draw()
```

## 基变换示例

```python
qc3 = QubitCircuit(3)
qc3.add_gate("CNOT", 1, 0)
qc3.add_gate("RX", 0, None, pi / 2, r"\pi/2")
qc3.add_gate("RY", 1, None, pi / 2, r"\pi/2")
qc3.add_gate("RZ", 2, None, pi / 2, r"\pi/2")
qc3.add_gate("ISWAP", [1, 2])
qc3.draw()
```

```python
U3 = gate_sequence_product(qc3.propagators())
U3
```

### 变换可仅用双比特门表示：

```python
qc4 = qc3.resolve_gates("CNOT")
qc4.draw()
```

```python
U4 = gate_sequence_product(qc4.propagators())
U4
```

```python
qc5 = qc3.resolve_gates("ISWAP")
qc5.draw()
```

```python
U5 = gate_sequence_product(qc5.propagators())
U5
```

### 也可用任意两种单比特旋转门 + 双比特门来表示。

```python
qc6 = qc3.resolve_gates(["ISWAP", "RX", "RY"])
qc6.draw()
```

```python
U6 = gate_sequence_product(qc6.propagators())
U6
```

```python
qc7 = qc3.resolve_gates(["CNOT", "RZ", "RX"])
qc7.draw()
```

```python
U7 = gate_sequence_product(qc7.propagators())
U7
```

## 处理非相邻相互作用


`QubitCircuit` 可以把非相邻比特间相互作用分解成一串相邻相互作用，这对自旋链模型等系统很有用。

```python
qc8 = QubitCircuit(3)
qc8.add_gate("CNOT", 2, 0)
qc8.draw()
```

```python
U8 = gate_sequence_product(qc8.propagators())
U8
```

```python
qc9 = qc8.adjacent_gates()
qc9.gates
```

```python
U9 = gate_sequence_product(qc9.propagators())
U9
```

```python
qc10 = qc9.resolve_gates("CNOT")
qc10.draw()
```

```python
U10 = gate_sequence_product(qc10.propagators())
U10
```

## 在电路中间插入门
从 QuTiP 4.4 开始，可在线路任意位置添加门。只需指定 `index` 参数。
据此也能一次把同一门插入多个位置。

```python
qc = QubitCircuit(1)
qc.add_gate("RX", targets=1, arg_value=np.pi / 2)
qc.add_gate("RX", targets=1, arg_value=np.pi / 2)
qc.add_gate("RY", targets=1, arg_value=np.pi / 2, index=[0])
qc.gates
```

## 用户自定义门
从 QuTiP 4.4 起，用户可用 Python 函数定义门：函数最多接收一个参数并返回 `Qobj`，其维度需与量子比特系统匹配。

```python
def user_gate1(arg_value):
    # controlled rotation X
    mat = np.zeros((4, 4), dtype=complex)
    mat[0, 0] = mat[1, 1] = 1.0
    mat[2:4, 2:4] = rx(arg_value).full()
    return Qobj(mat, dims=[[2, 2], [2, 2]])


def user_gate2():
    # S gate
    mat = np.array([[1.0, 0], [0.0, 1.0j]])
    return Qobj(mat, dims=[[2], [2]])
```

若要让 `QubitCircuit` 识别这些门，需要修改其属性 `QubitCircuit.user_gates`，格式为 `{name: gate_function}`。

```python
qc = QubitCircuit(2)
qc.user_gates = {"CTRLRX": user_gate1, "S": user_gate2}
```

调用 `add_gate` 时给出目标比特与参数：

```python
# qubit 0 controls qubit 1
qc.add_gate("CTRLRX", targets=[0, 1], arg_value=pi / 2)
# qubit 1 controls qubit 0
qc.add_gate("CTRLRX", targets=[1, 0], arg_value=pi / 2)
# a gate can also be added using the Gate class
g_T = Gate("S", targets=[1])
qc.add_gate("S", targets=[1])
props = qc.propagators()
```

```python
props[0]  # qubit 0 controls qubit 1
```

```python
props[1]  # qubit 1 controls qubit 0
```

```python
props[2]  # S  gate acts on qubit 1
```

## 软件版本

```python
about()
```
