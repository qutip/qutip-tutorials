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

# Notebook 标题

Author: C. Staufenbiel, 2022

### 简介

这个 notebook 是 QuTiP 用户指南类 Jupyter Notebook 的模板。通过这个模板，
你可以了解用户指南通常的组织方式。我们也希望借此保证所有 notebook 风格一致，
便于新用户阅读和理解。创建自己的 notebook 时，只需复制这个模板并填入你的内容。
模板中的描述可以帮助你把握整体写作风格。

在引言部分，你应该像这里一样说明本 notebook 的目标。接下来会进入具体步骤。
一个好的实践是：每个代码单元前都配一个 markdown 说明单元，这通常更受新用户欢迎。
另外，请记得更新 notebook 标题以及各章节标题。

### 第一部分

在这个 notebook（以及大多数 notebook）中，第一步通常是导入需要的包。

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip import Bloch, basis, sesolve, sigmay, sigmaz

%matplotlib inline
```

下一步我们设置一个量子比特态，并将其画在 Bloch 球上。直观图示通常很有帮助。

```python
psi = (2.0 * basis(2, 0) + basis(2, 1)).unit()
b = Bloch()
b.add_states(psi)
b.show()
```

### 仿真

下面定义一个简单哈密顿量，使用 `qutip.sesolve` 求解薛定谔方程，并观察
$\sigma_y$ 的期望值。你也可以在代码单元中加入注释，区分不同操作步骤。

```python
# simulate the unitary dynamics
H = sigmaz()
times = np.linspace(0, 10, 100)
result = sesolve(H, psi, times, [sigmay()])

# plot the expectation value
plt.plot(times, result.expect[0])
plt.xlabel("Time"), plt.ylabel("<sigma_y>")
plt.show()
```

我们绘制了拉莫尔进动的结果图。每个 notebook 都应在末尾包含 `qutip.about()`，
用于展示运行环境信息，方便他人复现。

### 环境信息

```python
qutip.about()
```

### 测试

这一部分可以加入测试，用于验证 notebook 是否产生了预期输出。我们将该部分放在
末尾，以免影响阅读体验。请使用 `assert` 定义测试，这样出现错误输出时单元会报错。

```python
assert np.allclose(result.expect[0][0], 0), \
    "期望值初始点不符合预期"
assert 1 == 1
```

```python

```
