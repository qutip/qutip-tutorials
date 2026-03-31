---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Jaynes-Cummings-Hubbard 模型导论：三站点系统

Authors: Vanshaj Bindal (Cardiff University)

## 引言

Jaynes-Cummings-Hubbard 模型（JCHM）是量子光学与凝聚态物理交叉领域中的一个基础量子系统。它描述了一组光学腔阵列：每个腔内包含一个两能级原子（或其他两能级系统），并且光子可以在相邻腔之间隧穿。

### 理论背景

JCHM 将两个重要量子模型结合在一起：

1. **Jaynes-Cummings 模型**：描述单个两能级原子与腔内量子化电磁场之间的相互作用，是腔量子电动力学（QED）中光-物质耦合的基本模型。

2. **Bose-Hubbard 模型**：描述玻色粒子（此处即光子）在晶格站点间跃迁并存在站内相互作用，是多体系统量子相变研究的核心模型之一。

两者结合后，JCHM 呈现出丰富物理现象，包括类似 Mott 绝缘态（光子局域在腔内）与类似超流态（光子在晶格中离域）之间的量子相变。

### 数学描述

对于三站点系统，哈密顿量可写为：

$$H = \omega_c \sum_{i=1}^3 a_i^\dagger a_i + \frac{\omega_a}{2} \sum_{i=1}^3 \sigma_i^z + g \sum_{i=1}^3 (a_i^\dagger \sigma_i^- + a_i \sigma_i^+) - J \sum_{i=1}^2 (a_i^\dagger a_{i+1} + a_{i+1}^\dagger a_i)$$

其中：
- $\omega_c$ 为腔频率
- $\omega_a$ 为原子跃迁频率
- $g$ 为原子-腔耦合强度
- $J$ 为相邻腔之间的跃迁强度
- $a_i^\dagger$ 与 $a_i$ 分别是第 $i$ 个腔中的光子产生与湮灭算符
- $\sigma_i^z$、$\sigma_i^+$ 与 $\sigma_i^-$ 分别是第 $i$ 个原子的 Pauli 与升降算符

该哈密顿量包含四部分：
1. 腔能量项：$\omega_c \sum_{i=1}^3 a_i^\dagger a_i$
2. 原子能量项：$\frac{\omega_a}{2} \sum_{i=1}^3 \sigma_i^z$
3. 原子-腔相互作用项：$g \sum_{i=1}^3 (a_i^\dagger \sigma_i^- + a_i \sigma_i^+)$
4. 光子跃迁项：$-J \sum_{i=1}^2 (a_i^\dagger a_{i+1} + a_{i+1}^\dagger a_i)$

### 导入包

首先导入所需库。QuTiP（Quantum Toolbox in Python）将提供本教程所需的量子算符与求解器。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import about, basis, destroy, expect, mesolve, qeye, sigmaz, tensor

%matplotlib inline
```

## 构建三站点 JCHM 哈密顿量

先定义系统参数，它们决定了量子系统的物理性质：

- `N`：每个腔保留的 Fock 态数目（限制最大光子数）
- `omega_c` 与 `omega_a`：腔与原子的频率（相等时处于共振）
- `g`：每个原子与其腔之间的耦合强度
- `J`：相邻腔间的跃迁强度，决定光子隧穿难易程度

```python
# System parameters
N = 5  # Number of Fock states per cavity(reduced for computational efficiency)
omega_c = 1.0  # Cavity frequency (sets the energy scale)
omega_a = 1.0  # Atomic transition frequency (in resonance with cavities)
g = 0.3  # Atom-cavity coupling strength
J = 0.2  # Hopping strength between cavities
```

### 创建哈密顿量

接下来我们编写函数来构建三站点 JCHM 的完整哈密顿量。这需要通过算符张量积精确表示三个腔与三个原子的联合希尔伯特空间。

关键挑战在于如何在总系统中正确表示“只作用于某个子系统”的算符。做法是与单位算符做张量积，把局域算符扩展到全空间。

```python
def create_three_site_jchm(N, omega_c, omega_a, g, J):
    """
    Create the Hamiltonian for a three-site Jaynes-Cummings-Hubbard model.
    In QuTiP, we construct operators for composite systems using tensor products.
    For each operator, we create a tensor product of the desired operator at the
    specific site and identity operators at all other sites.

    Parameters:
    -----------
    N : int
        Number of Fock states in each cavity.
    omega_c : float
        Cavity frequency.
    omega_a : float
        Atomic transition frequency.
    g : float
        Atom-cavity coupling strength.
    J : float
        Hopping strength between cavities.

    Returns:
    --------
    H : Qobj
        Hamiltonian for the three-site JCHM.
    ops : dict
        Dictionary of operators for measurements.
    """
    # Define operators for each cavity-atom system using tensor products
    # The full Hilbert space is: (cavity1 鈯?atom1 鈯?cavity2 鈯?atom2 鈯?cavity3 鈯?atom3)

    # Cavity annihilation operators
    c1 = tensor(destroy(N), qeye(2), qeye(N), qeye(2), qeye(N), qeye(2))  # Cavity 1
    c2 = tensor(qeye(N), qeye(2), destroy(N), qeye(2), qeye(N), qeye(2))  # Cavity 2
    c3 = tensor(qeye(N), qeye(2), qeye(N), qeye(2), destroy(N), qeye(2))  # Cavity 3

    # Atomic Pauli operators
    sz1 = tensor(qeye(N), sigmaz(), qeye(N), qeye(2), qeye(N), qeye(2))  # Atom 1
    sz2 = tensor(qeye(N), qeye(2), qeye(N), sigmaz(), qeye(N), qeye(2))  # Atom 2
    sz3 = tensor(qeye(N), qeye(2), qeye(N), qeye(2), qeye(N), sigmaz())  # Atom 3

    # Atomic lowering operators
    sm1 = tensor(
        qeye(N), destroy(2), qeye(N), qeye(2), qeye(N), qeye(2)
    )  # Atom 1 lowering
    sm2 = tensor(
        qeye(N), qeye(2), qeye(N), destroy(2), qeye(N), qeye(2)
    )  # Atom 2 lowering
    sm3 = tensor(
        qeye(N), qeye(2), qeye(N), qeye(2), qeye(N), destroy(2)
    )  # Atom 3 lowering

    # Atomic raising operators (adjoint of lowering)
    sp1 = sm1.dag()
    sp2 = sm2.dag()
    sp3 = sm3.dag()

    # Construct the Hamiltonian

    # 1. Cavity energy terms:
    # These terms represent the energy of photons in each cavity
    H_cavity = omega_c * (c1.dag() * c1 + c2.dag() * c2 + c3.dag() * c3)

    # 2. Atom energy terms:
    # These terms represent the energy of each two-level atom
    H_atom = 0.5 * omega_a * (sz1 + sz2 + sz3)

    # 3. Cavity-atom interaction terms:
    # These terms represent the interaction between cavities and atom
    H_interaction = g * (
        (c1.dag() * sm1 + c1 * sp1)
        + (c2.dag() * sm2 + c2 * sp2)
        + (c3.dag() * sm3 + c3 * sp3)
    )

    # 4. Photon hopping terms:
    # These terms allow photons to tunnel between adjacent cavities
    H_hopping = -J * ((c1.dag() * c2 + c2.dag() * c1) + (c2.dag() * c3 + c3.dag() * c2))

    # Total Hamiltonian
    H = H_cavity + H_atom + H_interaction + H_hopping

    # Create dictionary of measurement operators for later use
    ops = {
        "cavity_n": [
            c1.dag() * c1,
            c2.dag() * c2,
            c3.dag() * c3,
        ],  # Photon number operators
        "atom_e": [
            sm1.dag() * sm1,
            sm2.dag() * sm2,
            sm3.dag() * sm3,
        ],  # Atomic excitation operators
        "cavity_a": [c1, c2, c3],  # Cavity field operators
    }

    return H, ops
```

现在构造系统哈密顿量。即便只有三个站点，得到的希尔伯特空间维度也会很大，因为总维度按 $(N \times 2)^3$ 增长。

```python
H, ops = create_three_site_jchm(N, omega_c, omega_a, g, J)
print(f"Dimension of the Hilbert space: {H.shape}")
```

## 探索基态

量子系统的基态是其最低能量态。对 JCHM 而言，基态性质会随参数变化而改变，尤其依赖跃迁强度 $J$ 与耦合强度 $g$ 的比值。

在强耦合区（$g \gg J$）中，基态更接近 Mott 绝缘态，光子与原子强烈绑定形成极化激元；在强跃迁区（$J \gg g$）中，基态更接近超流态，光子在各腔之间离域。

下面通过计算基态来观察其基本性质：

```python
# Calculate ground state (lowest energy eigenstate)
evals, evecs = H.eigenstates(eigvals=1, sort="low")
ground_state = evecs[0]

# Calculate expectation values of photon numbers and atomic excitations
photon_numbers = [expect(ops["cavity_n"][i], ground_state) for i in range(3)]
atom_excitations = [expect(ops["atom_e"][i], ground_state) for i in range(3)]

# Print the values for detailed inspection
print("Ground state energy:", evals[0])
print("Photon numbers in ground state:", photon_numbers)
print("Atomic excitations in ground state:", atom_excitations)

# Plot the results to visualize the distribution
plt.figure(figsize=(10, 6))
plt.bar([0, 1, 2], photon_numbers, width=0.3, label="Photon number")
plt.bar([0.4, 1.4, 2.4], atom_excitations, width=0.3, label="Atomic excitation")
plt.xlabel("Site")
plt.ylabel("Expectation value")
plt.title("Ground state properties")
plt.xticks([0.2, 1.2, 2.2], ["Site 1", "Site 2", "Site 3"])
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 时间演化与动力学

JCHM 最有趣的方面之一是光子与原子激发的动力学。通过求解含时薛定谔方程，可以观察光子如何在腔链中传播并与原子相互作用。

我们将构造一个初态：第一个腔有一个光子，所有原子都在基态。该初态对应一个局域激发，随后随时间演化。

### 动力学理论背景

量子态的时间演化由薛定谔方程支配：

$$i\hbar\frac{d|\psi(t)\rangle}{dt} = H|\psi(t)\rangle$$

在 QuTiP 中，我们使用 `mesolve` 对该方程做数值求解。对于无耗散的封闭系统，动力学是纯相干的，会出现由光子跃迁与原子耦合导致的量子振荡。封闭系统通常可用 `sesolve`，但这里 `mesolve` 会自动委托给 `sesolve`。

```python
# Create initial state: first cavity has one photon, all atoms in ground state
# The state is a tensor product: |1鉄┾倎 鈯?|g鉄┾倎 鈯?|0鉄┾倐 鈯?|g鉄┾倐 鈯?|0鉄┾們 鈯?|g鉄┾們
psi0 = tensor(
    basis(N, 1), basis(2, 0), basis(N, 0), basis(2, 0), basis(N, 0), basis(2, 0)
)

# Define time points for the evolution
tlist = np.linspace(0, 40, 200)

# Calculate time evolution using the master equation solver
# For a closed system without dissipation, this solves the Schr枚dinger equation
result = mesolve(H, psi0, tlist, [], e_ops=(ops["cavity_n"] + ops["atom_e"]))

# Plot the results to visualize the dynamics
plt.figure(figsize=(12, 10))

# Plot photon numbers - shows how photons move through the chain
plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(tlist, result.expect[i], label=f"Cavity {i+1}")
plt.xlabel("Time")
plt.ylabel("Photon number")
plt.title("Photon dynamics in coupled cavities")
plt.legend()
plt.grid(True)

# Plot atomic excitations - shows how atoms interact with photons
plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(tlist, result.expect[i + 3], label=f"Atom {i+1}")
plt.xlabel("Time")
plt.ylabel("Excitation probability")
plt.title("Atomic excitation dynamics")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 相变特征

JCHM 的另一重要现象是：随着跃迁强度增大，系统会从类 Mott 绝缘相向类超流相转变。

### 相变理论背景

在热力学极限（无限大系统）下，JCHM 在某个临界 $J/g$ 比值处发生量子相变：

1. **Mott 绝缘相（$J < J_critical$）**：光子局域在各腔内，并与对应原子强耦合形成极化激元。

2. **超流相（$J > J_critical$）**：光子在晶格中离域，并具有长程相干。

虽然三站点系统太小，无法呈现严格意义上的相变，但仍可通过序参量观测到相变前驱特征。

### 序参量

为刻画不同相，我们使用两个序参量：

1. **光子数涨落（$\Delta n$）**：衡量光子数方差。超流相中由于位置不确定性更大，该量通常上升。

2. **腔场振幅（$\langle a \rangle$）**：可作为“序参量”，在超流相中趋于非零。

```python
def compute_order_parameters(H, ops):
    """
    Compute order parameters for the ground state of the Hamiltonian.

    Order parameters help us identify different quantum phases:
    - Photon number fluctuations (delta_n): Larger in the superfluid phase
    - Cavity field amplitude (alpha): Non-zero in the superfluid phase

    Parameters:
    -----------
    H : Qobj
        Hamiltonian of the system.
    ops : dict
        Dictionary of operators.

    Returns:
    --------
    delta_n : float
        Average photon number fluctuations.
    alpha : float
        Average cavity field amplitude.
    """
    # Find the ground state
    evals, evecs = H.eigenstates(eigvals=1, sort="low")
    ground_state = evecs[0]

    # Calculate photon numbers and fluctuations
    delta_n_values = []
    alpha_values = []

    for i in range(3):  # For all three cavities
        # Photon number
        n_op = ops["cavity_n"][i]
        n = expect(n_op, ground_state)

        # Photon number squared - needed for variance calculation
        n_sq = expect(n_op * n_op, ground_state)

        # Photon number fluctuation (standard deviation)
        delta_n = np.sqrt(n_sq - n**2)
        delta_n_values.append(delta_n)

        # Field amplitude - a measure of coherence
        a_op = ops["cavity_a"][i]
        alpha = abs(expect(a_op, ground_state))
        alpha_values.append(alpha)

    # Return average values across all cavities
    return np.mean(delta_n_values), np.mean(alpha_values)
```

接下来在一段跃迁强度范围内计算这些序参量，观察其在接近并跨越相变区域时的变化：

```python
# Calculate order parameters for different hopping strengths
J_values = np.linspace(0.01, 0.5, 20)
delta_n_values = []
alpha_values = []

for J_val in J_values:
    # Create Hamiltonian with current J value
    H_J, ops_J = create_three_site_jchm(N, omega_c, omega_a, g, J_val)

    # Calculate order parameters
    delta_n, alpha = compute_order_parameters(H_J, ops_J)

    # Store values
    delta_n_values.append(delta_n)
    alpha_values.append(alpha)

    print(f"J = {J_val:.3f}: calculated", end="\r")

print("\nCalculations complete!")

# Plot order parameters
plt.figure(figsize=(10, 6))
plt.plot(J_values, delta_n_values, "o-", label="Photon number fluctuations")
plt.plot(J_values, alpha_values, "s-", label="Cavity field amplitude")
plt.axvline(x=0.2, color="r", linestyle="--", label="Approximate phase boundary")
plt.xlabel("Hopping strength (J)")
plt.ylabel("Order parameter")
plt.title("Order Parameters vs. Hopping Strength")
plt.legend()
plt.grid(True)
plt.show()
```
对于有限尺寸系统（尤其三站点小系统），基态会保留某些对称性，使场算符期望值（$\langle a \rangle$）即便跨越相变区也可能保持为零。光子数涨落（蓝色圆点）仍按预期增大，体现了相变前驱行为。

在更大系统中，$\langle a \rangle$ 会在超流相中变为非零，可作为真正的序参量；但在本小系统里，需要借助光子数涨落等其他量来识别转变迹象。

## 链中的光子传播

为更直观展示光子在三站点链中的传播，我们绘制随时间变化的各腔光子数热图。这能清晰呈现 JCHM 中光子传播的波动特性。

### 光子传播理论背景

在 JCHM 中，光子并非经典地“从一个腔跳到下一个腔”，而是表现出量子波动行为，包含干涉与概率振幅在晶格中的扩展。传播图样同时取决于跃迁强度 $J$ 和原子-腔耦合强度 $g$。

```python
# Calculate time evolution with finer time resolution
tlist1 = np.linspace(0, 30, 1000)
result = mesolve(H, psi0, tlist1, [], e_ops=ops["cavity_n"])

photon_data = np.array([result.expect[0], result.expect[1], result.expect[2]])

# Create a color plot showing photon propagation
plt.figure(figsize=(10, 6))
plt.imshow(
    photon_data,
    aspect="auto",
    extent=[0, tlist1[-1], 0.5, 3.5],
    origin="lower",
    interpolation="bilinear",
    cmap="viridis",
)
plt.colorbar(label="Photon number")
plt.xlabel("Time")
plt.ylabel("Cavity site")
plt.yticks([1, 2, 3])
plt.title("Photon propagation through the three-site chain")
plt.tight_layout()
plt.show()
```

## 失谐的影响

到目前为止我们令腔与原子频率共振（$\omega_c = \omega_a$）。现在考察二者存在失谐时的情形。失谐定义为 $\Delta = \omega_a - \omega_c$，它会显著影响系统动力学。

### 失谐理论背景

当腔与原子失谐时，二者能级不再精确匹配，从而改变能量交换效率。在 JCHM 中，失谐会：

1. 改变腔与原子之间的有效耦合
2. 改变光子在链中的传输速度
3. 改变原子被激发的概率

较大失谐通常会减弱腔-原子有效耦合，使系统更接近“几乎自由光子”的弱耦合腔行为。

```python
# Set up parameters for detuning study
delta_values = [-0.5, 0, 0.5]  # Detuning values
J_fixed = 0.2  # Fixed hopping strength

# Time parameters
tlist = np.linspace(0, 40, 200)

plt.figure(figsize=(10, 6))

for delta in delta_values:
    # Calculate new atomic frequency with detuning (螖 = 蠅a - 蠅c)
    omega_a_detuned = omega_c + delta

    # Create Hamiltonian with detuning
    H_detuned, ops_detuned = create_three_site_jchm(
        N, omega_c, omega_a_detuned, g, J_fixed
    )

    # Create initial state
    psi0 = tensor(
        basis(N, 1), basis(2, 0), basis(N, 0), basis(2, 0), basis(N, 0), basis(2, 0)
    )

    # Calculate time evolution
    result = mesolve(H_detuned, psi0, tlist, [], e_ops=[ops_detuned["cavity_n"][0]])

    # Plot photon number in first cavity
    plt.plot(tlist, result.expect[0], label=f"Detuning 螖 = {delta}")

plt.xlabel("Time")
plt.ylabel("Photon number in first cavity")
plt.title("Effect of Detuning on Photon Dynamics")
plt.legend()
plt.grid(True)
plt.show()
```

## 改变耦合强度

原子与腔之间的耦合强度 $g$ 也是决定 JCHM 动力学的关键参数。下面考察不同 $g$ 如何改变光子传输行为。

### 耦合强度理论背景

原子-腔耦合强度 $g$ 决定光子与原子相互作用的强弱：

1. **弱耦合（$g$ 小）**：光子与原子作用较弱，可更自由地在腔间跃迁
2. **强耦合（$g$ 大）**：光子与原子强作用形成极化激元（光子与原子激发的混合态）
3. **超强耦合（$g \sim \omega_c$）**：会导致旋波近似失效（本模拟未覆盖）

当 $g/J$ 增大时，系统会更接近 Mott 绝缘相，光子更倾向被束缚在各自腔中。

```python
# Set up parameters for coupling strength study
g_values = [0.1, 0.3, 0.5]  # Different coupling strengths
J_fixed = 0.2  # Fixed hopping strength

# Time parameters
tlist = np.linspace(0, 40, 200)

plt.figure(figsize=(10, 6))

for g_val in g_values:
    # Create Hamiltonian with specific coupling strength
    H_g, ops_g = create_three_site_jchm(N, omega_c, omega_a, g_val, J_fixed)

    # Create initial state
    psi0 = tensor(
        basis(N, 1), basis(2, 0), basis(N, 0), basis(2, 0), basis(N, 0), basis(2, 0)
    )

    # Calculate time evolution
    result = mesolve(
        H_g, psi0, tlist, [], e_ops=[ops_g["cavity_n"][1]]
    )  # Measure photon number in second cavity

    # Plot photon number in second cavity
    plt.plot(tlist, result.expect[0], label=f"Coupling g = {g_val}")

plt.xlabel("Time")
plt.ylabel("Photon number in second cavity")
plt.title("Effect of Coupling Strength on Photon Transfer")
plt.legend()
plt.grid(True)
plt.show()
```

## 结论

本教程基于 QuTiP 中的三站点系统，探索了 Jaynes-Cummings-Hubbard 模型。该模型连接了量子光学与凝聚态物理，为耦合腔系统中的光-物质相互作用提供了直观认识。

### 关键物理结论：

1. **基态性质**：JCHM 基态体现极化激元形成，表现为较高原子激发与较低光子数。

2. **量子动力学**：光子在腔链中以波动方式传播，呈现量子干涉，而非经典粒子式运动。

3. **相变特征**：即便在小系统中，随着跃迁强度提高，仍可观察到从 Mott 绝缘相到超流相的前驱迹象。

4. **参数依赖性**：系统行为对失谐与耦合强度高度敏感，这些参数决定激发局域化与离域化之间的平衡。

JCHM 在量子模拟、量子光学和量子信息处理中都有重要应用。它是理解更复杂量子光学晶格系统的基础模块，并可望在超导电路、光子晶体、光晶格冷原子等平台上实现。

## 软件版本

```python
about()
```
