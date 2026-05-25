---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: qiskit-stable8
    language: python
    name: python3
---

# 使用 GRAPE 计算 CNOT 实现所需控制场

[这是基于 Robert Johansson 已弃用 CNOT GRAPE notebook 的更新实现](https://nbviewer.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-cnot.ipynb)

```python
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
# the library for quantum control
import qutip_qtrl.pulseoptim as cpo
```

```python
# total duration
T = 2 * np.pi
# number of time steps
times = np.linspace(0, T, 500)
```

```python
U_0 = qt.operators.identity(4)
U_target = qt.core.gates.cnot()
```

### 初始算符

```python
U_0
```

### 目标算符

```python
U_target
```

```python
# Drift Hamiltonian
g = 0
H_drift = g * (
    qt.tensor(qt.sigmax(), qt.sigmax()) + qt.tensor(qt.sigmay(), qt.sigmay())
)
```

```python
H_ctrl = [
    qt.tensor(qt.sigmax(), qt.identity(2)),
    qt.tensor(qt.sigmay(), qt.identity(2)),
    qt.tensor(qt.sigmaz(), qt.identity(2)),
    qt.tensor(qt.identity(2), qt.sigmax()),
    qt.tensor(qt.identity(2), qt.sigmay()),
    qt.tensor(qt.identity(2), qt.sigmaz()),
    qt.tensor(qt.sigmax(), qt.sigmax()),
    qt.tensor(qt.sigmay(), qt.sigmay()),
    qt.tensor(qt.sigmaz(), qt.sigmaz()),
]
```

```python
H_labels = [
    r"$u_{1x}$",
    r"$u_{1y}$",
    r"$u_{1z}$",
    r"$u_{2x}$",
    r"$u_{2y}$",
    r"$u_{2z}$",
    r"$u_{xx}$",
    r"$u_{yy}$",
    r"$u_{zz}$",
]
```

## GRAPE

```python
result = cpo.optimize_pulse_unitary(
    H_drift,
    H_ctrl,
    U_0,
    U_target,
    num_tslots=500,
    evo_time=(2 * np.pi),
    # this attribute is crucial for convergence!!
    amp_lbound=-(2 * np.pi * 0.05),
    amp_ubound=(2 * np.pi * 0.05),
    fid_err_targ=1e-9,
    max_iter=500,
    max_wall_time=60,
    alg="GRAPE",
    optim_method="FMIN_L_BFGS_B",
    method_params={
        "disp": True,
        "maxiter": 1000,
    },
)
```

```python
for attr in dir(result):
    if not attr.startswith("_"):
        print(f"{attr}: {getattr(result, attr)}")

print(np.shape(result.final_amps))
```

## 绘制存在单比特隧穿时 CNOT 门的控制场

```python
def plot_control_amplitudes(times, final_amps, labels):
    num_controls = final_amps.shape[1]

    y_max = 0.1  # Fixed y-axis scale
    y_min = -0.1

    for i in range(num_controls):
        fig, ax = plt.subplots(figsize=(8, 3))

        for j in range(num_controls):
            # Highlight the current control
            color = "black" if i == j else "gray"
            alpha = 1.0 if i == j else 0.1
            ax.plot(
                times,
                final_amps[:, j],
                label=labels[j],
                color=color,
                alpha=alpha
                )
        ax.set_title(f"Control Fields Highlighting: {labels[i]}")
        ax.set_xlabel("Time")
        ax.set_ylabel(labels[i])
        ax.set_ylim(y_min, y_max)  # Set fixed y-axis limits
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


plot_control_amplitudes(times, result.final_amps / (2 * np.pi), H_labels)
```

## 保真度/重叠

```python
U_target
```

```python
U_f = result.evo_full_final
U_f.dims = [[2, 2], [2, 2]]
```

```python
U_f
```

```python
print(f"Fidelity: {qt.process_fidelity(U_f, U_target)}")
```

## 过程层析


量子过程层析（QPT）是一种通过重建过程矩阵（也称 chi 矩阵）
来表征未知量子操作的技术。
该矩阵描述了输入量子态如何被该操作变换。

下面定义双量子比特系统在泡利基下的基算符
`{I, X, Y, Z}`，
这些算符构成了描述任意量子操作的完备基。


### 理想 CNOT 门

```python
op_basis = [[qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]] * 2
op_label = [["i", "x", "y", "z"]] * 2
```

`U_target` 是理想 CNOT 门。

`qt.to_super(U_target)` 将其转换为超算符形式，这是 QPT 所需表示。

`qt.qpt(U_i_s, op_basis)` 计算理想门的 chi 矩阵。

```python
fig = plt.figure(figsize=(12, 6))

U_i_s = qt.to_super(U_target)

chi = qt.qpt(U_i_s, op_basis)

fig = qt.qpt_plot_combined(chi, op_label, fig=fig, threshold=0.001)
```

```python
op_basis = [[qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]] * 2
op_label = [["i", "x", "y", "z"]] * 2
```

```python
fig = plt.figure(figsize=(12, 6))

U_f_s = qt.to_super(U_f)

chi = qt.qpt(U_f_s, op_basis)

fig = qt.qpt_plot_combined(chi, op_label, fig=fig, threshold=0.01)
```

## 版本信息


```python
qt.about()
```
