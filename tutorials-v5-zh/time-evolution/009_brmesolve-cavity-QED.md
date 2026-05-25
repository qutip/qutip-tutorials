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

<!-- #region -->
# Bloch-Redfield 求解器：含耗散原子-腔系统

Author: C. Staufenbiel (2022)

灵感来自 P.D. Nation 的
[`brmesolve notebook`](https://github.com/qutip/qutip-notebooks/blob/master/examples/brmesolve.ipynb)。

### 简介

本 notebook 不会系统介绍 `qutip.brmesolve()` 的全部细节。
更完整的入门请参考
[*Bloch-Redfield Solver: Two Level System*](007_brmesolve_tls.md)
以及[官方文档](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-bloch-redfield.html)。

Lindblad 主方程求解器 `qutip.mesolve()` 通过塌缩算符处理耗散，
这些算符可分别作用在系统各子系统上。
例如在原子-腔系统中，可分别为腔和原子定义耗散（对应湮灭算符）。
本例将展示：当原子与腔强耦合时，这种处理方式的局限性。

本例采用如下 Rabi 哈密顿量：

$$H =  \omega_0 a^\dagger a + \omega_0 \sigma_+ \sigma_- + g(a^\dagger + a)(\sigma_- + \sigma_+)$$

我们将改变耦合强度 $g$，并比较 `qutip.mesolve()` 与 `qutip.brmesolve()` 的结果。


### 导入

<!-- #endregion -->

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, brmesolve, destroy, identity, ket2dm, mesolve,
                   plot_energy_levels, plot_expectation_values, tensor)

%matplotlib inline
```

<!-- #region -->


这里构建一个与热浴耦合、含耗散的原子-腔系统。
<!-- #endregion -->

```python
N = 10  # num. cavity modes

# operators
a = tensor(destroy(N), identity(2))
sm = tensor(identity(N), destroy(2))

# expectation operators
e_ops = [a.dag() * a, sm.dag() * sm]

# initial state
psi0 = ket2dm(tensor(basis(N, 1), basis(2, 0)))
```

下面设置原子-腔系统频率与耦合强度，并定义与环境的耗散。
这里仅考虑“漏腔”情形：只允许腔向环境损耗，不考虑原子耗散。

```python
w0 = 1.0 * 2 * np.pi
g_weak = 0.1 * 2 * np.pi
g_strong = 0.75 * 2 * np.pi
kappa = 0.05

# collapse operators (for mesolve)
c_ops = [np.sqrt(kappa) * a]
# noise power spectrum (for brmesolve)
a_ops = [[(a + a.dag()), lambda w: kappa * (w > 0)]]

# Hamiltonians
H_no = w0 * a.dag() * a + w0 * sm.dag() * sm
H_weak = w0 * a.dag() * a + w0 * sm.dag() * sm + \
         g_weak * (a + a.dag()) * (sm + sm.dag())
H_strong = w0 * a.dag() * a + w0 * sm.dag() * sm + \
           g_strong * (a + a.dag()) * (sm + sm.dag())
```

现在分别在弱耦合与强耦合下，使用 `qutip.mesolve` 与 `qutip.brmesolve` 求解动力学。
### 弱耦合

```python
# times for simulation
times = np.linspace(0, 10 * 2 * np.pi / g_weak, 1000)
# simulation
result_me_weak = mesolve(H_weak, psi0, times, c_ops, e_ops)
result_brme_weak = brmesolve(H_weak, psi0, times, a_ops, e_ops)
fig, axes = plot_expectation_values(
    [result_me_weak, result_brme_weak], ylabels=["<n_cav>", "<n_atom>"]
)
for ax in axes:
    ax.legend(['mesolove', 'brmesolve'], loc='upper right')
```

在弱耦合下，Lindblad 主方程求解器 `qutip.mesolve` 与
Bloch-Redfield 求解器 `qutip.brmesolve` 给出的结果较为一致。

### 强耦合

```python
# times for simulation
times = np.linspace(0, 10 * 2 * np.pi / g_strong, 1000)
# simulation
result_me_strong = mesolve(H_strong, psi0, times, c_ops, e_ops)
result_brme_strong = brmesolve(H_strong, psi0, times, a_ops, e_ops)
fig, axes = plot_expectation_values(
    [result_me_strong, result_brme_strong], ylabels=["<n_cav>", "<n_atom>"]
)
for ax in axes:
    ax.legend(['mesolove', 'brmesolve'], loc='upper right')
```

在强耦合区间，两种求解器结果出现差异。
原因在于强耦合下系统本征态由“原子+腔”混合组成（杂化本征态）。
Lindblad 方式默认“子系统耗散不影响其他子系统状态”，
即此例中腔损耗不影响原子。
但在强耦合下该假设不再成立：耗散会在耦合系统本征态间引发跃迁，
从而也影响原子态。
Bloch-Redfield 求解器天然考虑了这种杂化效应，
因此在该场景下通常更准确。

从能级图也可看出杂化：
无耦合时原子与腔能级按 `w_0` 各自分布；
弱耦合下能级仅小幅劈裂；
强耦合下原本属于不同子系统的能级明显聚集并混合。

```python
plot_energy_levels([H_no, H_weak, H_strong],
                   h_labels=["no coupling", "weak", "strong"]);
```

### 非 secular 解
`qutip.brmesolve()` 默认启用 secular 近似，
即忽略哈密顿量中快速振荡项。
Bloch-Redfield 方法并非必须使用该近似，
可通过设置参数 `sec_cutoff=-1` 关闭。
这在某些仿真中有用。
对本例强耦合原子-腔系统而言，关闭近似后变化不大。

```python
result_brme_nonsec = brmesolve(H_strong, psi0, times, a_ops,
                               sec_cutoff=-1, e_ops=e_ops)
fig, axes = plot_expectation_values(
    [result_brme_strong, result_brme_nonsec], ylabels=["<n_cav>", "<n_atom>"]
)
for ax in axes:
    ax.legend(['brme_strong', 'brme_nonsec'], loc='upper right')
```

### 态的迹
Lindblad 主方程方法保证密度矩阵演化保持物理性：
即在数值误差范围内保持迹与正定性。

Bloch-Redfield 的一个缺点是无法严格保证这一点，
因此密度矩阵迹可能偏离 1。
下面绘制弱耦合情形下两种方法演化状态的迹。
可见 Bloch-Redfield 的迹对 1 有轻微偏离
（本例约 $10^{-12}$ 量级）。
这个偏差在本例中不构成问题，但在其他系统中可能影响结果。

注意：图中 y 轴自动做了“+1 平移并按 $10^{-12}$ 缩放”，
因此理论迹 1 对应图中的 $y=0$。

```python
# calculate states for weak coupling
me_states = mesolve(H_weak, psi0, times, c_ops, e_ops=[])
brme_states = brmesolve(H_weak, psi0, times, a_ops, e_ops=[])
# plot the traces and expected trace
plt.axhline(1.0, label="expected trace", c="red", linestyle="--")
plt.plot(times, [state.tr() for state in brme_states.states], label="brme")
plt.plot(times, [state.tr() for state in me_states.states], label="me")
plt.legend(), plt.xlabel("Time"), plt.ylabel("Trace of states");
```

## 环境信息

```python
about()
```

## 测试

```python
# Weak coupling should be close
assert np.allclose(result_me_weak.expect[0],
                   result_brme_weak.expect[0], atol=0.05)
assert np.allclose(result_me_weak.expect[1],
                   result_brme_weak.expect[1], atol=0.05)

# Strong coupling should not be close
assert not np.allclose(result_me_strong.expect[0],
                       result_brme_strong.expect[0], atol=0.1)
assert not np.allclose(result_me_strong.expect[1],
                       result_brme_strong.expect[1], atol=0.1)

# Trace of states should be approx. 1
assert np.allclose([s.tr() for s in me_states.states],
                   np.ones(times.shape[0]))
assert np.allclose([s.tr() for s in brme_states.states],
                   np.ones(times.shape[0]))
```
