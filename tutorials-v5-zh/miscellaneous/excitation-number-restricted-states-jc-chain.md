---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: qutip-tutorials-v5
    language: python
    name: python3
---

<!-- #region -->
# 激发数受限态：Jaynes-Cummings 链

Authors: Robert Johansson (jrjohansson@gmail.com), Neill Lambert (nwlambert@gmail.com), Maximilian Meyer-Moelleringhof (m.meyermoelleringhof@gmail.com)

## 引言

ENR（excitation-number-restricted）函数会为多体系统构造一个只包含“总激发数受限”状态的基底。
当模型守恒总激发数时（如下方 JC 链示例），这种做法尤其有用。

可先看一个 4 个模、每个模 5 个态的系统。
总希尔伯特空间维数为 $5^4 = 625$。
若我们只关心总激发数不超过 2 的状态，则只需保留如下状态，例如


    (0, 0, 0, 0)
    (0, 0, 0, 1)
    (0, 0, 0, 2)
    (0, 0, 1, 0)
    (0, 0, 1, 1)
    (0, 0, 2, 0)
    ...

ENR 函数会在这个受限空间内创建 4 个模对应的算符和态。
例如

```python
a1, a2, a3, a4 = enr_destroy([5, 5, 5, 5], excitations=2)
```

会为每个模创建湮灭算符。
之后就可以像常规流程一样，用 a1, ..., a4 构建哈密顿量、塌缩算符、期望值算符等。

本示例通过与常规 QuTiP 实现对比，展示 ENR 态的优势。
我们分别计算时间演化与偏迹，结果一致，同时性能明显提升。

#### 注意

QuTiP 中许多默认函数对这种方法构造的态和算符会失效。
另外，在该形式下，不同子系统的湮灭/产生算符不再对易。
因此构造哈密顿量时，必须把湮灭算符写在右侧、产生算符写在左侧（更多细节见 QuTiP v5 官方论文）。
可用的 ENR 相关函数请参考官方文档：[Energy Restricted Operators in the official documentation](https://qutip.readthedocs.io/en/qutip-5.0.x/apidoc/functions.html#module-qutip.core.energy_restricted)。
<!-- #endregion -->

```python
import numpy as np
from qutip import (Qobj, about, basis, destroy, enr_destroy, enr_fock,
                   enr_state_dictionaries, identity, liouvillian, mesolve,
                   plot_expectation_values, tensor)
from qutip.core.energy_restricted import EnrSpace

%matplotlib inline
```

## Jaynes-Cummings 链

标准 Jaynes-Cummings 模型描述单个二能级原子与单个腔模的相互作用。
在本示例中，我们将多个这类系统排成链，并令相邻系统通过腔模相互耦合。
记 $a_i$（$a_i^\dagger$）为第 $i$ 个腔的湮灭（产生）算符，$s_i$（$s_i^\dagger$）为第 $i$ 个原子的湮灭（产生）算符。
完整哈密顿量分为三部分：

各子系统本征项：

$H_0 = \sum_{i=0}^{N} a_i^\dag a_i + s_i^\dag s_i$,

原子-腔耦合项：

$H_{int,AC} = \sum_{i=0}^{N} = \frac{1}{2} (a_i^\dag s_i + s_i^\dag a_i)$,

腔-腔耦合项：

$H_{int,CC} = \sum_{i=0}^{N-1} 0.9 \cdot (a_i^\dag a_{i+1} + a_{i+1}^\dag a_{i})$,

其中耦合强度 $0.9$ 仅为示例取值。


### 问题参数

```python
N = 4  # number of systems
M = 2  # number of cavity states
dims = [M, 2] * N  # dimensions of JC spin chain
excite = 1  # total number of excitations
init_excite = 1  # initial number of excitations
```

### 构建时间演化求解流程

```python
def solve(d, psi0):
    # 腔模湮灭算符
    a = d[::2]
    # 原子湮灭算符
    sm = d[1::2]

    # 注意湮灭与产生算符的顺序
    H0 = sum([aa.dag() * aa for aa in a]) + sum([s.dag() * s for s in sm])

    # 原子-腔耦合
    Hint_ac = 0
    for n in range(N):
        Hint_ac += 0.5 * (a[n].dag() * sm[n] + sm[n].dag() * a[n])

    # 腔-腔耦合
    Hint_cc = 0
    for n in range(N - 1):
        Hint_cc += 0.9 * (a[n].dag() * a[n + 1] + a[n + 1].dag() * a[n])

    H = H0 + Hint_ac + Hint_cc

    e_ops = [x.dag() * x for x in d]
    c_ops = [0.01 * x for x in a]

    times = np.linspace(0, 250, 1000)
    L = liouvillian(H, c_ops)
    opt = {"nsteps": 5000, "store_states": True}
    result = mesolve(H, psi0, times, c_ops, e_ops, options=opt)
    return result, H, L
```

### 常规 QuTiP 态与算符

```python
d = [
    tensor(
        [
            destroy(dim1) if idx1 == idx2 else identity(dim1)
            for idx1, dim1 in enumerate(dims)
        ]
    )
    for idx2, _ in enumerate(dims)
]
psi0 = tensor(
    [
        basis(dim, init_excite) if idx == 1 else basis(dim, 0)
        for idx, dim in enumerate(dims)
    ]
)
```

常规张量积空间中，不同子系统的算符彼此对易。
示例：

```python
d[0].dag() * d[1] == d[1] * d[0].dag()
```

求解时间演化：

```python
res1, H1, L1 = solve(d, psi0)
print(f"Run time: {res1.stats['run time']}s")
```

### 使用 ENR 态与算符

```python
d_enr = enr_destroy(dims, excite)
init_enr = [init_excite if n == 1 else 0 for n in range(2 * N)]
psi0_enr = enr_fock(dims, excite, init_enr)
```

使用 ENR 后，必须放弃多希尔伯特空间的标准张量结构。
因此不同子系统算符一般不再对易：

```python
d_enr[0].dag() * d_enr[1] == d_enr[1] * d_enr[0].dag()
```

求解时间演化：

```python
res2, H2, L2 = solve(d_enr, psi0_enr)
print(f"Run time: {res2.stats['run time']}s")
```

### 期望值比较

```python
fig, axes = plot_expectation_values([res1, res2])
fig.set_figwidth(10)
fig.set_figheight(8)
for idx, ax in enumerate(axes):
    if idx % 2:
        ax.set_ylabel(f"Atom {idx//2}")
    else:
        ax.set_ylabel(f"Cavity {idx//2}")
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
fig.tight_layout()
```

### 偏迹计算

ENR 态会使许多 QuTiP 标准特性失效，
`ptrace` 就是其中之一。
下面演示如何对 ENR 态计算偏迹，并与标准 QuTiP 方式对比结果。

```python
def ENR_ptrace(rho, sel, excitations):
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)

    if (sel < 0).any() or (sel >= len(rho.dims[0])).any():
        raise TypeError("Invalid selection index in ptrace.")

    drho = rho.dims[0]
    _, state2idx, _ = enr_state_dictionaries(drho, excitations)

    dims_short = np.asarray(drho).take(sel).tolist()
    nstates2, state2idx2, _ = enr_state_dictionaries(dims_short, excitations)

    # 构造新的密度矩阵
    rhout = np.zeros((nstates2, nstates2), dtype=np.complex64)
    # 被迹掉子系统的维度索引
    rest = np.setdiff1d(np.arange(len(drho)), sel)
    for state in state2idx:
        for state2 in state2idx:
            # 将对应元素加到新密度矩阵
            state_red = np.asarray(state).take(rest)
            state2_red = np.asarray(state2).take(rest)
            if np.all(state_red == state2_red):
                rhout[
                    state2idx2[tuple(np.asarray(state).take(sel))],
                    state2idx2[tuple(np.asarray(state2).take(sel))],
                ] += rho[state2idx[state], state2idx[state2]]

    new_dims = np.asarray(drho).take(sel).tolist()
    return Qobj(rhout, dims=[EnrSpace(new_dims, excite)] * 2)
```

```python
res1.states[10].ptrace(1)
```

```python
ENR_ptrace(res2.states[10], 1, excite)
```

```python
res1.states[10].ptrace([0, 1, 4])
```

```python
ENR_ptrace(res2.states[10], [0, 1, 4], excite)
```

## 关于

```python
about()
```

## 测试

```python
assert np.allclose(
    res1.states[10].ptrace([1]).full(),
    ENR_ptrace(res2.states[10], [1], excite).full(),
), "The approaches do not yield the same result."
```
