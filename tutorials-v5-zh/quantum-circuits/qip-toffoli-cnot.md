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

# Toffoli 门用 CNOT 与单量子比特旋转的分解


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

<!-- #region -->
本 notebook 演示如何将 Toffoli 门重写为 CNOT 门和单量子比特门的组合，
并通过比较两组门序列的矩阵表示来验证其等价性。
关于 Toffoli 分解的更多信息，可参考 Nielsen & Chuang 第 4.3 节（p178）。


**注意：电路图可视化显示需要 [ImageMagick](https://imagemagick.org/index.php)。**

如果已安装 conda，可通过 `conda install imagemagick` 快速安装 ImageMagick。
否则请参考 ImageMagick 官方文档中的安装说明。
<!-- #endregion -->

```python
from qutip import about
from qutip_qip.operations import gate_sequence_product
from qutip_qip.circuit import QubitCircuit
```

```python
q = QubitCircuit(3, reverse_states=False)
q.add_gate("TOFFOLI", controls=[0, 2], targets=[1])
```

```python
q.draw()
```

```python
U = gate_sequence_product(q.propagators())

U.tidyup()
```

```python
q2 = q.resolve_gates()
```

```python
q2.draw()
```

```python
U2 = gate_sequence_product(q2.propagators())

U2.tidyup()
```

```python
U == U2
```

## 版本信息

```python
about()
```
