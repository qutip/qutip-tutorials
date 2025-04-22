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

# Decomposition of the Toffoli gate in terms of CNOT and single-qubit rotations


Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson

<!-- #region -->
This notebooks demonstrates how a toffoli gate can be rewritten in terms of CNOT gates and single qubit gates, and verifies the equivalence of the two gate sequences by comparing their matrix representations. For more information about the toffoli decomposition, see Nielsen & Chuang, Sec. 4.3, p178.


**Note: The circuit image visualizations require [ImageMagick](https://imagemagick.org/index.php) for display.**

ImageMagick can be easily installed with the command `conda install imagemagick` if you have conda installed.
Otherwise, please follow the installation instructions on the ImageMagick documentation.
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
q.png
```

```python
U = gate_sequence_product(q.propagators())

U.tidyup()
```

```python
q2 = q.resolve_gates()
```

```python
q2.png
```

```python
U2 = gate_sequence_product(q2.propagators())

U2.tidyup()
```

```python
U == U2
```

## Versions

```python
about()
```
