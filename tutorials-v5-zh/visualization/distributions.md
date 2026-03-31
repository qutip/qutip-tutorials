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

# 使用 qutip.distributions

Author: Mathis Beaudoin (2025)

### 简介

本 notebook 演示如何在 QuTiP 中使用概率分布。首先导入所需包。

```python
from qutip import fock, about
from qutip.distributions import HarmonicOscillatorWaveFunction
from qutip.distributions import HarmonicOscillatorProbabilityFunction
import matplotlib.pyplot as plt
```

### 谐振子波函数

这里我们用 `HarmonicOscillatorWaveFunction()` 类展示谐振子波函数在空间中的分布（n=0 到 n=7）。

可选地，使用 `extent` 参数为各坐标设置取值范围；还可用 `steps` 参数设置该范围内的数据点数量。
给定这些信息后，分布会被生成，并可通过 `.visualize()` 方法可视化。

```python
M = 8
N = 20

fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

for n in range(M):
    psi = fock(N, n)
    wf = HarmonicOscillatorWaveFunction(psi, 1.0, extent=[-10, 10])
    wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))
```

### 谐振子概率分布函数

`HarmonicOscillatorProbabilityFunction()` 类对应于 `HarmonicOscillatorWaveFunction()` 数据的模方。
这里沿用前面的示例。

```python
M = 8
N = 20

fig, ax = plt.subplots(M, 1, figsize=(10, 12), sharex=True)

for n in range(M):
    psi = fock(N, n)
    wf = HarmonicOscillatorProbabilityFunction(psi, 1.0, extent=[-10, 10])
    wf.visualize(fig=fig, ax=ax[M-n-1], show_ylabel=False, show_xlabel=(n == 0))
```

### 环境信息

```python
about()
```
