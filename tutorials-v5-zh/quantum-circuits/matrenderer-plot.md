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

# 使用 MatRenderer 绘制与自定义量子电路

Author: Rushiraj Gadhvi (gadhvirushiraj@gmail.com)

本 notebook 是使用 QuTiP-QIP `MatRenderer`（基于 Matplotlib 的渲染工具）
绘制量子电路的完整指南。
它展示了创建与修改电路图时可用的多种自定义选项，
内容分为两大部分：

- **电路级自定义**
- **门级自定义**

```python
import qutip
import numpy as np
from qutip import Qobj
from qutip_qip.operations import rx
from qutip_qip.circuit import QubitCircuit
```

```python
qc = QubitCircuit(2, num_cbits=1)
qc.add_gate("CNOT", controls=0, targets=1)
qc.add_gate("SNOT", targets=1)
qc.add_gate("ISWAP", targets=[0, 1])
qc.add_measurement("M0", targets=1, classical_store=0)
```

```python
qc.draw()
```

---

### 电路级自定义选项

---

##### 修改字体大小

```python
qc.draw(fontsize=13)
```

##### 背景颜色

```python
qc.draw(bgcolor="gray")
```

##### 门凸起样式控制

```python
qc.draw(bulge=False)
```

##### 自定义线路标签

```python
qc.draw(wire_label=["some_name", "some_long_name", "long_long_name"])
```

##### 控制电路末端额外线路延伸

```python
qc.draw(end_wire_ext=0)
```

```python
qc.draw(end_wire_ext=5)
```

##### 压缩电路布局

```python
qc.draw(gate_margin=0.1)
```

##### 添加标题

```python
qc.draw(title="QuTiP")
```

##### 切换主题

```python
qc.draw(theme="light")
```

```python
qc.draw(theme="dark", title="QuTiP")
```

```python
qc.draw(theme="modern")
```

##### 层对齐

```python
qc = QubitCircuit(3)
qc.add_gate("H", targets=[1])
qc.add_gate("RZ", targets=[2], arg_value=0.5, style={"showarg": True})
qc.add_gate("RZ", targets=[2], arg_value=0.5, style={"showarg": True})
qc.add_gate("CNOT", controls=[0], targets=[1], style={"showarg": True})
qc.add_gate("CNOT", controls=[0], targets=[1], style={"showarg": True})
qc.add_gate("SWAP", targets=[0, 2])
```

```python
# not-aligned gates
qc.draw(theme="modern")
```

```python
# aligned gates
qc.draw(align_layer=True, theme="modern")
```

---

### 门级自定义选项

---

```python
qc = QubitCircuit(7)
qc.add_gate("H", targets=[0], style={"fontcolor": "red"})
qc.add_gate("H", targets=[1], style={"fontstyle": "italic"})
qc.add_gate("H", targets=[2], style={"fontweight": "bold"})
qc.add_gate("H", targets=[3], style={"color": "green"})
qc.add_gate("H", targets=[4], style={"fontsize": 12})
qc.add_gate("H", targets=[5], arg_label="hadamard gate")
qc.add_gate("H", targets=[6], style={"fontfamily": "cursive"})
```

```python
qc.draw()
```

##### 显示参数值

```python
qc = QubitCircuit(3)
qc.add_gate("RX", targets=[0], arg_value=np.pi / 12, style={"showarg": True})
qc.add_gate(
    "RY", targets=[1], arg_value=2 * np.pi / 3, style={"showarg": True}
)
qc.add_gate("RY", targets=[2], arg_value=0.3, style={"showarg": True})
qc.draw()
```

#### 使用用户自定义门

```python
def user_gate1(arg_value):
    # controlled rotation X
    mat = np.zeros((4, 4), dtype=np.complex)
    mat[0, 0] = mat[1, 1] = 1.0
    mat[2:4, 2:4] = rx(arg_value).full()
    return Qobj(mat, dims=[[2, 2], [2, 2]])


def user_gate2():
    # S gate
    mat = np.array([[1.0, 0], [0.0, 1.0j]])
    return Qobj(mat, dims=[[2], [2]])
```

```python
qc = QubitCircuit(3)
qc.user_gates = {"CTRLRX": user_gate1, "S": user_gate2}

# qubit 1 controls qubit 0
qc.add_gate("CTRLRX", targets=[1, 0], arg_value=np.pi / 2)
# qubit 2 is target of S gate
qc.add_gate("S", targets=[2])
```

```python
qc.draw()
```

---

### 更多电路示例

---

```python
trotter_simulation_noisey = QubitCircuit(4)

trotter_simulation_noisey.add_gate("RZ", targets=[0])
trotter_simulation_noisey.add_gate("RZ", targets=[1])

trotter_simulation_noisey.add_gate("CNOT", controls=0, targets=1)
trotter_simulation_noisey.add_gate("RX", targets=[0])
trotter_simulation_noisey.add_gate("CNOT", controls=0, targets=1)

trotter_simulation_noisey.add_gate("CNOT", controls=0, targets=2)
trotter_simulation_noisey.add_gate("RX", targets=[0])
trotter_simulation_noisey.add_gate("CNOT", controls=0, targets=2)

trotter_simulation_noisey.add_gate("RZ", targets=[2], arg_value=-np.pi / 2)
trotter_simulation_noisey.add_gate("CNOT", controls=0, targets=2)
trotter_simulation_noisey.add_gate("RY", targets=[0])
trotter_simulation_noisey.add_gate("CNOT", controls=0, targets=2)
trotter_simulation_noisey.add_gate("RZ", targets=[2], arg_value=np.pi / 2)

trotter_simulation_noisey.add_gate("CNOT", controls=1, targets=3)
trotter_simulation_noisey.add_gate("RX", targets=[1])
trotter_simulation_noisey.add_gate("CNOT", controls=1, targets=3)

trotter_simulation_noisey.add_gate("RZ", targets=[3], arg_value=-np.pi / 2)
trotter_simulation_noisey.add_gate("CNOT", controls=1, targets=3)
trotter_simulation_noisey.add_gate("RY", targets=[1])
trotter_simulation_noisey.add_gate("CNOT", controls=1, targets=3)
trotter_simulation_noisey.add_gate("RZ", targets=[3], arg_value=np.pi / 2)

trotter_simulation_noisey.draw(theme="dark", title="Trotter Simulation")
```

---

```python
qutip.about()
```
