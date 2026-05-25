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

# QuTiPv5 论文示例 - 使用 QIP 的量子线路

Authors: Maximilian Meyer-Moelleringhof (m.meyermoelleringhof@gmail.com), Boxi Li (etamin1201@gmail.com), Neill Lambert (nwlambert@gmail.com)

量子线路是表示与操纵量子算法的标准框架（无论视觉上还是概念上）。
作为 QuTiP 生态成员，QuTiP-QIP[
1](#References) 提供了该框架并带来若干关键能力。
它可将表示线路的幺正算符无缝接入 QuTiP 的 `Qobj` 体系。
此外，它还连接了 QuTiP-QOC 与开放系统求解器，支持带真实噪声效应的脉冲级线路仿真。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    about,
    basis,
    destroy,
    expect,
    ket2dm,
    mesolve,
    qeye,
    sesolve,
    sigmax,
    sigmay,
    sigmaz,
    tensor,
)
from qutip_qip.circuit import QubitCircuit
from qutip_qip.device import SCQubits

%matplotlib inline
```

## 引言

本示例展示如何

- 构建一个简单量子线路来模拟双量子比特哈密顿量动力学
- 使用辅助比特引入噪声，模拟开放系统动力学
- 把两类仿真都放到硬件后端（处理器）上运行，后端会自行模拟其内禀噪声动力学

考虑两比特相互作用哈密顿量：

$H = \dfrac{\epsilon_1}{2} \sigma_z^{(1)} + \dfrac{\epsilon_2}{2} \sigma_z^{(2)} + g \sigma_{x}^{(1)} \sigma_{x}^{(2)}$.

其中 Pauli 矩阵 $\sigma_z^{(1/2)}$ 描述各比特二能级系统，
两比特通过 $\sigma_x^{(1/2)}$ 耦合，强度为 $g$。

```python
# Parameters
epsilon1 = 0.7
epsilon2 = 1.0
g = 0.3

sx1 = sigmax() & qeye(2)
sx2 = qeye(2) & sigmax()

sy1 = sigmay() & qeye(2)
sy2 = qeye(2) & sigmay()

sz1 = sigmaz() & qeye(2)
sz2 = qeye(2) & sigmaz()

H = 0.5 * epsilon1 * sz1 + 0.5 * epsilon2 * sz2 + g * sx1 * sx2

init_state = basis(2, 0) & basis(2, 1)
```

## 线路可视化

在进入仿真前，先看如何可视化线路。
在 QuTiP-QIP 中，线路可用三种方式绘制：文本、matplotlib、latex。

```python
qc = QubitCircuit(2, num_cbits=1)
qc.add_gate("H", 0)
qc.add_gate("H", 1)
qc.add_gate("CNOT", 1, 0)
qc.add_measurement("M", targets=[0], classical_store=0)
```

然后调用不同渲染器绘图：

```python
qc.draw("text")
qc.draw("matplotlib")
qc.draw("latex")
```

## 模拟哈密顿量动力学

量子仿真中的常见策略是把薛定谔方程传播离散为多个短时间步，逐步逼近目标解。
单步传播子可用 *Trotter 分解* 近似：

$\psi(t_f) = e^{-i (H_A + H_B) t_f} \psi(0) \approx [e^{-i H_A dt} e^{-i H_B dt}]^d \psi(0)$,

其中时间步长 $dt = t_f / d$ 足够小时近似有效。
关键是把 $H_A,H_B$ 选成可映射到量子门的形式。

本例取

$H_A = \dfrac{\epsilon_1}{2} \sigma_z^{(1)} + \dfrac{\epsilon_2}{2} \sigma_z^{(2)}$，以及

$H_B = g \sigma_x^{(1)} \sigma_x^{(2)}$。

可构造两比特线路并使用门：

$A_1 = e^{-i \epsilon_1 \sigma_z^{(1)} dt / 2}$,

$A_2 = e^{-i \epsilon_2 \sigma_z^{(2)} dt / 2}$，以及

$B = e^{-i g \sigma_x^{(1)} \sigma_x^{(2)} dt}$。

将它们作用到初态并重复 $d$ 次。

不同硬件原生门集不同。
由于我们将使用超导量子比特后端，这里把上述门用 RZ（绕 Z 旋转）、RZX（XZ 组合旋转）和 Hadamard 门表示。
QuTiP-QIP 支持大量门型，也支持自定义门；更多信息见原始论文[
1](#References)。

当然，仿真精度很依赖 Trotter 步长。
为控制教程运行时间，这里设 $dt=4.0$，误差可能较明显。
建议尝试更小步长观察误差降低。

```python
# simulation parameters
tf = 20.0  # total time
dt = 4.0  # Trotter step size
num_steps = int(tf / dt)
times_circ = np.arange(0, tf + dt, dt)
```

```python
# initialization of two qubit circuit
trotter_simulation = QubitCircuit(2)

# gates for trotterization with small timesteps dt
trotter_simulation.add_gate("RZ", targets=[0], arg_value=(epsilon1 * dt))
trotter_simulation.add_gate("RZ", targets=[1], arg_value=(epsilon2 * dt))

trotter_simulation.add_gate("H", targets=[0])
trotter_simulation.add_gate("RZX", targets=[0, 1], arg_value=g * dt * 2)
trotter_simulation.add_gate("H", targets=[0])
trotter_simulation.compute_unitary()
```

```python
trotter_simulation.draw("matplotlib")
```

```python
# Evaluate multiple iteration of a circuit
result_circ = init_state
state_trotter_circ = [init_state]

for dd in range(num_steps):
    result_circ = trotter_simulation.run(state=result_circ)
    state_trotter_circ.append(result_circ)
```

### 含噪硬件

可把量子线路加载到硬件后端，模拟在不同硬件上执行。
本例选择超导线路：

```python
processor = SCQubits(num_qubits=2, t1=2e5, t2=2e5)
processor.load_circuit(trotter_simulation)
# Since SCQubit is modelled as a qutrit, we need three-level systems here
init_state_trit = tensor(basis(3, 0), basis(3, 1))
```

随后按与前面类似方式运行仿真。
区别是这里返回的是所用 QuTiP 求解器的 `results` 对象。
由于初始化时给定了有限 $T_1,T_2$，本例使用 `mesolve()`。
处理器内部会定义其哈密顿量、可用控制、脉冲形状、$T_1$/$T_2$ 等。

```python
state_proc = init_state_trit
state_list_proc = [init_state_trit]

for dd in range(num_steps):
    result = processor.run_state(state_proc)
    state_proc = result.final_state
    state_list_proc.append(result.final_state)
```

可查看求解器使用的脉冲形状：

```python
processor.plot_pulses()
```

### 结果对比

为评估量子线路仿真，与标准 `sesolve` 结果进行比较。

```python
# Exact Schrodinger equation
tlist = np.linspace(0, tf, 200)
states_sesolve = sesolve(H, init_state, tlist).states
```

```python
sz_qutrit = basis(3, 0) * basis(3, 0).dag() - basis(3, 1) * basis(3, 1).dag()

expec_sesolve = expect(sz1, states_sesolve)
expec_trotter = expect(sz1, state_trotter_circ)
expec_supcond = expect(sz_qutrit & qeye(3), state_list_proc)

plt.plot(tlist, expec_sesolve, "-", label="Ideal")
plt.plot(times_circ, expec_trotter, "--d", label="Trotter circuit")
plt.plot(times_circ, expec_supcond, "-.o", label="noisy hardware")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

## Lindblad 仿真

为模拟 Lindblad 主方程，这里参考两个近期方案[
2, 3](#References)：
利用一系列幺正算符近似 Lindblad 动力学到任意阶。
其实现依赖辅助比特，以及对 Lindblad 塌缩算符执行测量/重置。
初态取扩展态 $\ket{\psi_D(t=0)} = \ket{\psi(t = 0)} \otimes \ket{0}^{\otimes K}$，其中 $K$ 是辅助比特数。
每个时间步中，系统与辅助比特相互作用时长为 $\sqrt{dt}$，随后辅助比特被重置到基态。

系统比特 $i$ 与其对应辅助比特（塌缩通道 $k$）的相互作用幺正可 Trotter 近似为

$U(\sqrt{dt}) \approx e^{-\frac{1}{2} i \sigma_{x}^{(i)}\sigma_{x}^{(k)} \sqrt{\gamma_k dt}} \cdot e^{\frac{1}{2} i \sigma_{y}^{(i)}\sigma_{y}^{(k)}\sqrt{\gamma_k dt}}$,

其中 $\gamma_k$ 为耗散率。
与上例一样，这些幺正可分解到超导后端原生门 Hadamard、`RZ`、`RX`、`RZX` 上。

```python
gam = 0.03  # dissipation rate
```

```python
trotter_simulation_noisy = QubitCircuit(4)

sqrt_term = np.sqrt(gam * dt)

# Coherent dynamics
trotter_simulation_noisy.add_gate("RZ", targets=[1], arg_value=epsilon1 * dt)
trotter_simulation_noisy.add_gate("RZ", targets=[2], arg_value=epsilon2 * dt)

trotter_simulation_noisy.add_gate("H", targets=[1])
trotter_simulation_noisy.add_gate("RZX", targets=[1, 2], arg_value=g * dt * 2)
trotter_simulation_noisy.add_gate("H", targets=[1])

# Decoherence
# exp(-i XX t)
trotter_simulation_noisy.add_gate("H", targets=[0])
trotter_simulation_noisy.add_gate("RZX", targets=[0, 1], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("H", targets=[0])

# exp(-i YY t)
trotter_simulation_noisy.add_gate("RZ", 1, arg_value=np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 0, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RZX", [0, 1], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("RZ", 1, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 0, arg_value=np.pi / 2)

# exp(-i XX t)
trotter_simulation_noisy.add_gate("H", targets=[2])
trotter_simulation_noisy.add_gate("RZX", targets=[2, 3], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("H", targets=[2])

# exp(-i YY t)
trotter_simulation_noisy.add_gate("RZ", 3, arg_value=np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 2, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RZX", [2, 3], arg_value=sqrt_term)
trotter_simulation_noisy.add_gate("RZ", 3, arg_value=-np.pi / 2)
trotter_simulation_noisy.add_gate("RX", 2, arg_value=np.pi / 2)

trotter_simulation_noisy.draw("matplotlib")
```

```python
state_system = ket2dm(init_state)
state_trotter_circ = [init_state]
ancilla = basis(2, 1) * basis(2, 1).dag()
for dd in range(num_steps):
    state_full = tensor(ancilla, state_system, ancilla)
    state_full = trotter_simulation_noisy.run(state=state_full)
    state_system = state_full.ptrace([1, 2])
    state_trotter_circ.append(state_system)
```

同样，我们希望在超导硬件后端上运行这段 Trotter 演化。注意：由于复杂度上升，计算可能需要数分钟（依赖本机性能）。

```python
processor = SCQubits(num_qubits=4, t1=3.0e4, t2=3.0e4)
processor.load_circuit(trotter_simulation_noisy)
```

```python
state_system = ket2dm(init_state_trit)
state_list_proc = [state_system]
for dd in range(num_steps):
    state_full = tensor(
        basis(3, 1) * basis(3, 1).dag(),
        state_system,
        basis(3, 1) * basis(3, 1).dag(),
    )
    result_noisey = processor.run_state(
        state_full,
        solver="mesolve",
        options={
            "store_states": False,
            "store_final_state": True,
        },
    )
    state_full = result_noisey.final_state
    state_system = state_full.ptrace([1, 2])
    state_list_proc.append(state_system)
    print(f"Step {dd+1}/{num_steps} finished.")
```

```python
processor.plot_pulses()
```

### 结果对比

```python
# Standard mesolve solution
sm1 = tensor(destroy(2).dag(), qeye(2))
sm2 = tensor(qeye(2), destroy(2).dag())
c_ops = [np.sqrt(gam) * sm1, np.sqrt(gam) * sm2]

result_me = mesolve(H, init_state, tlist, c_ops, e_ops=[sz1, sz2])
```

```python
expec_mesolve = result_me.expect[0]
expec_trotter = expect(sz1, state_trotter_circ)
expec_supcond = expect(sz_qutrit & qeye(3), state_list_proc)

plt.plot(tlist, expec_mesolve, "-", label=r"Ideal")
plt.plot(times_circ, expec_trotter, "--d", label="trotter")
plt.plot(times_circ, expec_supcond, "-.o", label=r"noisy hardware")
plt.xlabel("Time")
plt.ylabel("Expectation values")
plt.legend()
plt.show()
```

## 参考文献

\[1\] [Li, et. al, Quantum (2022)](http://dx.doi.org/10.22331/q-2022-01-24-630)

\[2\] [Ding, et. al, PRX Quantum (2024)](https://doi.org/10.1103/PRXQuantum.5.020332)

\[3\] [Cleve and Lang, ICALP (2017)](https://doi.org/10.48550/arXiv.1612.09512)

\[4\] [QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)


## 关于

```python
about()
```

### 测试

```python
np.testing.assert_allclose(expec_trotter, expec_supcond, atol=0.22)

tc1 = np.abs(tlist - times_circ[1]).argmin()
tc2 = np.abs(tlist - times_circ[2]).argmin()
np.testing.assert_allclose([expec_mesolve[tc1]], [expec_trotter[1]], atol=0.2, rtol=0.3)
np.testing.assert_allclose([expec_mesolve[tc1]], [expec_supcond[1]], atol=0.2, rtol=0.3)
np.testing.assert_allclose([expec_mesolve[tc2]], [expec_trotter[2]], atol=0.2, rtol=0.3)
np.testing.assert_allclose([expec_mesolve[tc2]], [expec_supcond[2]], atol=0.2, rtol=0.3)
```
