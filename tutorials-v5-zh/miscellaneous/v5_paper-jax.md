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

# QuTiPv5 论文示例：QuTiP-JAX 与 mesolve 及自动微分

Authors: Maximilian Meyer-Moelleringhof (m.meyermoelleringhof@gmail.com), Rochisha Agarwal (rochisha.agarwal2302@gmail.com), Neill Lambert (nwlambert@gmail.com)

多年来，GPU 一直是加速数值任务的核心工具。
如今很多库都能开箱即用地利用 GPU 来加速高代价计算。
QuTiP 的灵活数据层可直接与这类库配合，从而显著缩短计算时间。
在多种框架中，与 QuTiP 结合最活跃的是 QuTiP-JAX 集成[
1](#References)，主要因为 JAX 拥有稳健的自动微分能力，并在机器学习领域广泛应用。

本示例展示 JAX 如何通过 QuTiP-JAX 包自然集成到 QuTiP v5[
2](#References)。
第一部分给出一个一维自旋链示例，演示如何配合 `mesolve()` 与 JAX 求解主方程。
第二部分聚焦自动微分：先看开放量子系统计数统计，再看受驱动量子比特系统，利用 `mcsolve()` 计算激发态布居对驱动频率的梯度。

## 引言

除标准 QuTiP 外，使用 QuTiP-JAX 还需安装 JAX。
该包自带 `jax.numpy`，在接口上与 `numpy` 基本对应，可无缝集成。

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip_jax as qj
from diffrax import PIDController, Tsit5
from jax import default_device, devices, grad, jacfwd, jacrev, jit
from qutip import (CoreOptions, about, basis, destroy, lindblad_dissipator,
                   liouvillian, mcsolve, mesolve, projection, qeye, settings,
                   sigmam, sigmax, sigmay, sigmaz, spost, spre, sprepost,
                   steadystate, tensor)

%matplotlib inline
```

导入 `qutip_jax` 后，数据层会新增 `jax` 与 `jax_dia` 两种格式。
它们分别对应稠密（`jax`）与自定义稀疏（`jax_dia`）表示。

```python
print(qeye(3, dtype="jax").dtype.__name__)
print(qeye(3, dtype="jaxdia").dtype.__name__)
```

若希望主方程求解器也使用 JAX 数据层，有两种方式。
第一，在求解器 options 中设置 `method: diffrax`。
第二，调用 `qutip_jax.set_as_default()`。
它会自动把默认数据类型切到 JAX 兼容版本，并把默认求解方法设为 `diffrax`。

```python
qj.set_as_default()
```

若要恢复原设置，可传入 `revert=True`。

```python
# qj.set_as_default(revert = True)
```

## 使用 JAX + Diffrax + GPU：一维 Ising 自旋链

在示例前先说明：GPU 加速效果高度依赖问题类型。
GPU 擅长并行大量小矩阵-向量运算，例如跨参数批量积分小系统，或重复小矩阵运算的量子线路仿真。
对“单个大矩阵 ODE”而言，收益并不总是明显，因为 ODE 求解本质上具有串行性。
不过在 QuTiP v5 论文[
2](#References) 中展示了存在一个规模交叉点，超过后使用 JAX 会更有优势。

### 一维 Ising 自旋链

为说明 QuTiP-JAX 用法，我们考虑一维自旋链哈密顿量

$H = \sum_{i=1}^N g_0 \sigma_z^{(n)} - \sum_{n=1}^{N-1} J_0 \sigma_x^{(n)} \sigma_x^{(n+1)}$.

这里有 $N$ 个自旋，能级劈裂为 $g_0$，耦合强度为 $J_0$。
链末端与环境耦合，其 Lindblad 耗散用塌缩算符 $\sigma_x^{(N-1)}$ 与耦合率 $\gamma$ 建模。

在 [QuTiPv5 论文](#References) 中，作者系统比较了随维度 $N$ 变化的计算时间。
这里不复现超算性能，而是聚焦于正确实现：用 JAX 与 `mesolve()` 求解该系统的 Lindblad 方程。

```python
# system parameters
N = 4  # number of spins
g0 = 1  # energy splitting
J0 = 1.4  # coupling strength
gamma = 0.1  # dissipation rate

# simulation parameters
tlist = jnp.linspace(0, 5, 100)
opt = {
    "normalize_output": False,
    "store_states": True,
    "method": "diffrax",
    "stepsize_controller": PIDController(
        rtol=settings.core["rtol"], atol=settings.core["atol"]
    ),
    "solver": Tsit5(),
}
```

```python
with CoreOptions(default_dtype="jaxdia"):
    # Operators for individual qubits
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    # Hamiltonian - Energy splitting terms
    H = 0.0
    for i in range(N):
        H += g0 * sz_list[i]

    # Interaction terms
    for n in range(N - 1):
        H += -J0 * sx_list[n] * sx_list[n + 1]

    # Collapse operator acting locally on single spin
    c_ops = [gamma * sx_list[N - 1]]

    # Initial state
    state_list = [basis(2, 1)] * (N - 1)
    state_list.append(basis(2, 0))
    psi0 = tensor(state_list)

    result = mesolve(H, psi0, tlist, c_ops, e_ops=sz_list, options=opt)
```

```python
for i, s in enumerate(result.expect):
    plt.plot(tlist, s, label=rf"$n = {i+1}$")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma^{(n)}_z \rangle$")
plt.legend()
plt.show()
```

## 自动微分

前一节展示了 QuTiP 中 JAX 数据层的用法。
除此之外，JAX 还提供自动微分能力。
在传统方案中，导数常用数值近似（如有限差分）计算，尤其高阶导数时往往代价高且精度受限。

自动微分则利用链式法则计算导数。
核心思想是：任意数值函数可分解为基础解析函数与运算组合。
因此借助链式法则，几乎所有高层函数的导数都可系统获得。

该技术应用很多，这里只展示两个与量子系统相关的例子。

### 量子系统与环境间激发交换统计

考虑一个通过单一跃迁算符与环境接触的开放量子系统。
再引入一个测量装置，记录系统与环境之间的激发交换流。
在给定时间 $t$ 内交换激发数 $n$ 的概率分布称为全计数统计，记作 $P_n(t)$。
该统计可导出许多实验观测量，如散粒噪声与电流。

本例通过修正后的密度算符与 Lindblad 主方程来计算此统计。
引入*倾斜*密度算符 $G(z,t) = \sum_n e^{zn} \rho^n (t)$，其中 $\rho^n(t)$ 表示在时刻 $t$ 已发生 $n$ 次交换时的条件密度算符，故 $\text{Tr}[\rho^n(t)] = P_n(t)$。
带跃迁算符 $C$ 时，其主方程为

$\dot{G}(z,t) = -\dfrac{i}{\hbar} [H(t), G(z,t)] + \dfrac{1}{2} [2 e^z C \rho(t)C^\dagger - \rho C^\dagger C - C^\dagger C \rho(t)]$.

可见在 $z = 0$ 时，该方程退化为普通 Lindblad 方程，并有 $G(0,t) = \rho(t)$。
更重要的是，它可通过对 $z$ 求导给出计数统计：

$\langle n^m \rangle (t) = \sum_n n^m \text{Tr} [\rho^n (t)] = \dfrac{d^m}{dz^m} \text{Tr} [G(z,t)]|_{z=0}$.

这些导数正是 JAX 自动微分发挥作用的地方。

```python
# system parameters
ed = 1
GammaL = 1
GammaR = 1

# simulation parameters
options = {
    "method": "diffrax",
    "normalize_output": False,
    "stepsize_controller": PIDController(rtol=1e-7, atol=1e-7),
    "solver": Tsit5(scan_kind="bounded"),
    "progress_bar": False,
}
```

在 JAX 中可指定使用的设备/处理器。
本例 notebook 里使用 CPU；若在你自己的机器上运行，可改成 GPU。

```python
with default_device(devices("cpu")[0]):
    with CoreOptions(default_dtype="jaxdia"):
        d = destroy(2)
        H = ed * d.dag() * d
        c_op_L = jnp.sqrt(GammaL) * d.dag()
        c_op_R = jnp.sqrt(GammaR) * d

        L0 = (
            liouvillian(H)
            + lindblad_dissipator(c_op_L)
            - 0.5 * spre(c_op_R.dag() * c_op_R)
            - 0.5 * spost(c_op_R.dag() * c_op_R)
        )
        L1 = sprepost(c_op_R, c_op_R.dag())

        rho0 = steadystate(L0 + L1)

        def rhoz(t, z):
            L = L0 + jnp.exp(z) * L1  # jump term
            tlist = jnp.linspace(0, t, 50)
            result = mesolve(L, rho0, tlist, options=options)
            return result.final_state.tr()

        # first derivative
        drhozdz = jacrev(rhoz, argnums=1)
        # second derivative
        d2rhozdz = jacfwd(drhozdz, argnums=1)
```

```python
tf = 100
Itest = GammaL * GammaR / (GammaL + GammaR)
shottest = Itest * (1 - 2 * GammaL * GammaR / (GammaL + GammaR) ** 2)
ncurr = drhozdz(tf, 0.0) / tf
nshot = (d2rhozdz(tf, 0.0) - drhozdz(tf, 0.0) ** 2) / tf

print("===== RESULTS =====")
print("Analytical current", Itest)
print("Numerical current", ncurr)
print("Analytical shot noise (2nd cumulant)", shottest)
print("Numerical shot noise (2nd cumulant)", nshot)
```

### 受驱动单量子比特系统与频率优化

第二个自动微分示例考虑受驱 Rabi 模型，其含时哈密顿量为

$H(t) = \dfrac{\hbar \omega_0}{2} \sigma_z + \dfrac{\hbar \Omega}{2} \cos (\omega t) \sigma_x$

其中 $\omega_0$ 是能级劈裂，$\Omega$ 是 Rabi 频率，$\omega$ 是驱动频率，$\sigma_{x/z}$ 为 Pauli 矩阵。
加入耗散后，系统动力学由 Lindblad 主方程描述，可用塌缩算符 $C = \sqrt{\gamma} \sigma_-$ 表征能量弛豫。

本例关注量子比特激发态布居

$P_e(t) = \bra{e} \rho(t) \ket{e}$

及其对频率 $\omega$ 的梯度。

我们通过调节驱动频率 $\omega$ 来优化该量，并借助 JAX 自动微分工具与 QuTiP 的 `mcsolve()` 计算 $P_e(t)$ 关于 $\omega$ 的梯度。

```python
# system parameters
gamma = 0.1  # dissipation rate
```

```python
# time dependent drive
@jit
def driving_coeff(t, omega):
    return jnp.cos(omega * t)


# system Hamiltonian
def setup_system():
    H_0 = sigmaz()
    H_1 = sigmax()
    H = [H_0, [H_1, driving_coeff]]
    return H
```

```python
# simulation parameters
psi0 = basis(2, 0)
tlist = jnp.linspace(0.0, 10.0, 100)
c_ops = [jnp.sqrt(gamma) * sigmam()]
e_ops = [projection(2, 1, 1)]
```

```python
# Objective function: returns final exc. state population
def f(omega):
    H = setup_system()
    arg = {"omega": omega}
    result = mcsolve(H, psi0, tlist, c_ops, e_ops=e_ops, ntraj=100, args=arg)
    return result.expect[0][-1]
```

```python
# Gradient of the excited state population with respect to omega
grad_f = grad(f)(2.0)
```

```python
print(grad_f)
```

## 参考文献




[1] [QuTiP-JAX](https://github.com/qutip/qutip-jax)

[2] [QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)


## 关于

```python
about()
```

## 测试

```python
assert jnp.isclose(Itest, ncurr, rtol=1e-5), "Current calc. deviates"
assert jnp.isclose(shottest, nshot, rtol=1e-1), "Shot noise calc. deviates."
```
