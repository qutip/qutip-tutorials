---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Bloch-Redfield 求解器：时间依赖算符

Authors: C. Staufenbiel, 2022

参照 [Bloch-Redfield 文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html?#time-dependent-bloch-redfield-dynamics) 中的说明。

### 简介
本 notebook 介绍如何在 Bloch-Redfield 求解器中使用时间依赖算符，
相关内容也可见[对应文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html?#time-dependent-bloch-redfield-dynamics)。

我们将讨论时间依赖哈密顿量和时间依赖耗散。

### 导入

```python
import numpy as np
from qutip import about, basis, brmesolve, destroy, plot_expectation_values

%matplotlib inline
```

在这个小示例中，我们构建一个具有 `N` 个态的系统，并以数算符作为哈密顿量。
对于恒定哈密顿量且不给出 `a_ops` 的情况，可以观察到期望值 $\langle n \rangle $ 为常数。

```python
# num modes
N = 2
# Hamiltonian
a = destroy(N)
H = a.dag() * a

# initial state
psi0 = basis(N, N - 1)

# times for simulation
times = np.linspace(0, 10, 100)

# solve using brmesolve
result_const = brmesolve(H, psi0, times, e_ops=[a.dag() * a])
```

```python
plot_expectation_values(result_const, ylabels=["<n>"]);
```

接下来我们定义一个字符串来描述时间依赖。
可以使用 Cython 实现支持的函数。完整支持函数列表见
[文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-time.html#time)。
例如 `sin` 或 `exp`。时间变量记为 `t`。

```python
time_dependence = "sin(t)"
```

### 时间依赖哈密顿量

作为第一个例子，我们定义一个时间依赖哈密顿量（见[这里](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-time.html)）。

$$ H = \hat{n} + sin(t) \hat{x} $$

同样，我们可以用 `brmesolve()` 求解其动力学。

```python
H_t = [H, [a + a.dag(), time_dependence]]
result_brme = brmesolve(H_t, psi0, times, e_ops=[a.dag() * a])
plot_expectation_values(result_brme, ylabels=["<n>"]);
```

### 时间依赖耗散

上面我们没有使用噪声功率谱，而 Bloch-Redfield 求解器正是主要用于这类问题。
该谱通过参数 `a_ops` 传入。
我们也可以在 `a_ops` 中加入基于字符串的时间依赖，从而让耗散本身随时间变化。

这里定义如下形式的噪声功率谱：

$$ J(\omega, t) = \kappa * e^{-t} \quad \text{for} \; \omega \geq 0$$

```python
# setup dissipation
kappa = 0.2
a_ops = [
    ([a+a.dag(), f'sqrt({kappa}*exp(-t))'], '(w>=0)')
]
# solve
result_brme_aops = brmesolve(H, psi0, times, a_ops, e_ops=[a.dag() * a])

plot_expectation_values([result_brme_aops], ylabels=["<n>"]);
```

与热浴的耦合有时写成如下算符形式：

$$ A = f(t)a + f(t)^* a^\dagger $$

若要在 `brmesolve` 中加入该类耦合，可在 `a_ops` 里传入元组。
例如当 $f(t) = e^{i * t}$ 时，可按如下方式定义强度为 $\kappa$ 的耦合算符 $A$。
注意第二个函数必须是第一个函数的复共轭，
且第二个算符是第一个算符的厄米共轭。

```python
a_ops = [([[a, 'exp(1j*t)'], [a.dag(), 'exp(-1j*t)']],
          f'{kappa} * (w >= 0)')]

# solve using brmesolve and plot expecation
result_brme_aops_sum = brmesolve(H, psi0, times, a_ops, e_ops=[a.dag() * a])
plot_expectation_values([result_brme_aops_sum], ylabels=["<n>"]);
```

### 环境信息

```python
about()
```

### 测试

```python
assert np.allclose(result_const.expect[0], 1.0)

# compare result from brme with a_ops to analytic solution
analytic_aops = (N - 1) * np.exp(-kappa * (1.0 - np.exp(-times)))
assert np.allclose(result_brme_aops.expect[0], analytic_aops)

assert np.all(np.diff(result_brme_aops_sum.expect[0]) <= 0.0)
```
