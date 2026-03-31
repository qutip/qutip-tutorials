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

# QuTiPv5 论文示例：`sesolve`、`mesolve` 与新的求解器类

Authors: Maximilian Meyer-Moelleringhof (m.meyermoelleringhof@gmail.com), Neill Lambert (nwlambert@gmail.com), Paul Menczel (paul@menczel.net)

在 QuTiP 中，`Qobj` 和 `QobjEvo` 是几乎所有计算的核心。
借助它们，配合丰富的求解器，可以模拟具有多种相互作用与结构的开放量子系统。
通常求解器接收初态、哈密顿量以及环境（常用速率或耦合强度描述），然后通过数值积分得到时间演化。

QuTiP v5[
1](#references) 引入了统一的求解器交互接口。
新接口基于类：用户可先为特定问题实例化求解器对象。
当同一哈密顿量被不同初态、时间步或其他选项反复使用时，这种方式尤其方便。
求解器复用时通常可获得明显加速。

在实例化时，先只给出哈密顿量和塌缩算符（例如 Lindblad 主方程的塌缩算符）。
初态、时间步等在 `Solver.run()` 中提供并执行仿真。

本 notebook 通过多个例子说明新求解器类的用法。

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    QobjEvo,
    SESolver,
    UnderDampedEnvironment,
    about,
    basis,
    brmesolve,
    fidelity,
    mesolve,
    qeye,
    sigmam,
    sigmax,
    sigmaz,
    spost,
    spre,
    sprepost,
)
from qutip.solver.heom import HEOMSolver

%matplotlib inline
```

## Part 0：新求解器类入门

第一个例子考虑两个相互作用的量子比特（暂时不与环境耦合）。
其哈密顿量为

$$H = \dfrac{\epsilon_1}{2} \sigma_z^{(1)} + \dfrac{\epsilon_2}{2} \sigma_z^{(2)} + g \sigma_{x}^{(1)} \sigma_{x}^{(2)}.$$ 

Pauli 矩阵 $\sigma_z^{(1/2)}$ 描述各自二能级系统，
两比特通过 $\sigma_x^{(1/2)}$ 耦合，耦合强度为 $g$。

```python
epsilon1 = 1.0
epsilon2 = 1.0
g = 0.1

sx1 = sigmax() & qeye(2)
sx2 = qeye(2) & sigmax()
sz1 = sigmaz() & qeye(2)
sz2 = qeye(2) & sigmaz()

H = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2

print(H)
```

其动力学满足薛定谔方程

$$i \hbar \dfrac{d}{dt} \ket{\psi} = H \ket{\psi}.$$ 

因此可用 `SESolver` 求解。

```python
se_solver = SESolver(H)
psi0 = basis(2, 0) & basis(2, 1)
tlist = np.linspace(0, 40, 100)
```

```python
se_res = se_solver.run(psi0, tlist, e_ops=[sz1, sz2])
```

```python
plt.plot(tlist, se_res.expect[0], label="i=1")
plt.plot(tlist, se_res.expect[1], label="i=2")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(i)} \rangle$")
plt.legend()
plt.show()
```

### 求解器与积分器选项

QuTiP v5 的另一个变化是：`options` 使用标准 Python 字典。
这样更灵活，也便于不同求解器暴露各自选项。
完整选项列表见各求解器在线文档。

常见选项包括 `store_states`（是否保存每个时刻状态）、`store_final_state`（是否保存末态）。
`method` 也很常用，用于指定 ODE 积分方法。
与数值积分相关的具体选项依赖所选方法；这里示例：
`atol`（绝对误差容限）、`nsteps`（相邻输出时刻间最大步数）、`max_step`（默认 Adams 方法允许的最大积分步长）。

```python
options = {"store_states": True, "atol": 1e-12, "nsteps": 1e3, "max_step": 0.1}
se_solver.options = options
```

```python
se_res = se_solver.run(psi0, tlist)
```

```python
print(se_res)
```

## Part 1：Lindblad 动力学及更一般情形

一般来说，薛定谔方程描述任意量子系统动力学。
但系统变大或连续自由度出现时，直接求解往往不可行。
因此各种主方程成为描述有限维（开放）量子系统动力学的主流方法。
主方程通常指关于约化密度算符 $\rho(t)$ 的一阶线性微分方程。
在 QuTiP 中，`mesolve` 是通用主方程求解器。
它可处理多种形式，默认实现的是 Lindblad 类型：

$$ \dot{\rho}(t) = - \dfrac{i}{\hbar} [H(t), \rho(t)] + \sum_n \dfrac{1}{2}[ 2 C_n \rho(t) C_n^\dagger - \rho(t) C_n^\dagger C_n - C^\dagger_n C_n \rho(t) ] . $$

除 $\rho(t)$ 与 $H(t)$ 外，方程还包含塌缩（跃迁）算符 $C_n = \sqrt{\gamma_n} A_n$。
它们定义系统与环境接触导致的耗散。
速率 $\gamma_n$ 则描述由算符 $A_n$ 连接态之间跃迁发生的频率。

继续双比特例子，这次通过塌缩算符
$C_1 = \sqrt{\gamma} \sigma_{-}^{(1)}$ 与 $C_2 = \sqrt{\gamma} \sigma_{-}^{(2)}$
把系统接到环境上，其中 $\sigma_-^{(i)}$ 将第 $i$ 个比特从激发态降到基态。
这里使用 `mesolve`。

```python
sm1 = sigmam() & qeye(2)
sm2 = qeye(2) & sigmam()
gam = 0.1  # dissipation rate
c_ops = [np.sqrt(gam) * sm1, np.sqrt(gam) * sm2]
```

```python
me_local_res = mesolve(H, psi0, tlist, c_ops, e_ops=[sz1, sz2])
```

```python
plt.plot(tlist, me_local_res.expect[0], label="i=1")
plt.plot(tlist, me_local_res.expect[1], label="i=2")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(i)} \rangle$")
plt.legend()
plt.show()
```

### 全局主方程 - Born-Markov-secular 近似

前例里塌缩算符是对每个比特局域作用的。
但在不同近似下，会得到不同形式的塌缩算符。
例如当比特间相互作用强于与热浴耦合时，在标准 Born-Markov 近似下会得到“全局”主方程。
此时虽仍可视作升降过程，但作用对象是耦合后双比特总体系本征态：

$$A_{ij} = \ket{\psi_i}\bra{\psi_i}$$

其速率为

$$\gamma_{ij} = | \bra{\psi_i} d \ket{\psi_j} |^2 S(\Delta_{ij}).$$

其中 $\ket{\psi_i}$ 是 $H$ 的本征态，$\Delta_{ij} = E_j - E_i$ 为能隙差。
$d$ 是系统-环境耦合算符。
功率谱取

$$S(\omega) = 2 J(\omega) [n_{th} (\omega) + 1] \theta(\omega) + 2J(-\omega)[n_{th}(-\omega)]\theta(-\omega)$$

它依赖环境细节（如谱密度 $J(\omega)$）及温度（通过 Bose-Einstein 分布 $n_{th}(\omega)$）。
$\theta$ 为 Heaviside 函数。

若假设平坦谱密度 $J(\omega)=\gamma/2$ 且零温，可写为

$$S(\omega) = \gamma \theta(\omega).$$

本例用 `mesolve()` 手工实现该零温环境。

```python
def power_spectrum(w):
    if w >= 0:
        return gam
    else:
        return 0


def make_co_list(energies, eigenstates):
    Nmax = len(eigenstates)
    collapse_list = []
    for i in range(Nmax):
        for j in range(Nmax):
            delE = energies[j] - energies[i]
            m1 = sx1.matrix_element(eigenstates[i].dag(), eigenstates[j])
            m2 = sx2.matrix_element(eigenstates[i].dag(), eigenstates[j])
            absolute = np.abs(m1) ** 2 + np.abs(m2) ** 2
            rate = power_spectrum(delE) * absolute
            if rate > 0:
                outer = eigenstates[i] * eigenstates[j].dag()
                collapse_list.append(np.sqrt(rate) * outer)
    return collapse_list
```

```python
all_energy, all_state = H.eigenstates()
collapse_list = make_co_list(all_energy, all_state)
tlist_long = np.linspace(0, 1000, 100)
```

```python
opt = {"store_states": True}
me_global_res = mesolve(
    H, psi0, tlist_long, collapse_list, e_ops=[sz1, sz2], options=opt
)
```

有趣的是，长时间演化后末态会接近双比特耦合系统的基态：

```python
grnd_state = all_state[0] @ all_state[0].dag()
fidelity = fidelity(me_global_res.states[-1], grnd_state)
print(f"Fidelity with ground-state: {fidelity:.6f}")
```

### 求解器对比

下面将上面的局域 Lindblad 与缀饰（全局）Lindblad 结果，与 Bloch-Redfield 求解器做比较。
Bloch-Redfield 的细节在其他教程中有更完整介绍；这里用它在给定热浴功率谱下求解弱耦合主方程。
在小耦合下，局域与全局主方程都与 Bloch-Redfield 一致；
但在强耦合下，局域主方程会偏离全局与 Bloch-Redfield 结果。

```python
# weak coupling
g = 0.1 * epsilon1
H_weak = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2
```

```python
# generate new collapse operators for weak coupling Hamiltonian
all_energy, all_state = H_weak.eigenstates()
co_list = make_co_list(all_energy, all_state)
```

```python
me_local_res = mesolve(H_weak, psi0, tlist, c_ops, e_ops=[sz1, sz2])
me_global_res = mesolve(H_weak, psi0, tlist, co_list, e_ops=[sz1, sz2])
br_res = brmesolve(
    H_weak,
    psi0,
    tlist,
    e_ops=[sz1, sz2],
    a_ops=[[sx1, power_spectrum], [sx2, power_spectrum]],
)
```

```python
plt.plot(tlist, me_local_res.expect[0], label=r"Local Lindblad")
plt.plot(tlist, me_global_res.expect[0], "--", label=r"Dressed Lindblad")
plt.plot(tlist, br_res.expect[0], ":", label=r"Bloch-Redfield")
plt.title("Weak Coupling")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

```python
# strong coupling
g = 2 * epsilon1
H_strong = (epsilon1 / 2) * sz1 + (epsilon2 / 2) * sz2 + g * sx1 * sx2
```

```python
# generate new collapse operators for weak coupling Hamiltonian
all_energy, all_state = H_strong.eigenstates()
co_list = make_co_list(all_energy, all_state)

# time list with smaller steps
tlist_fine = np.linspace(0, 40, 1000)
```

```python
me_local_res = mesolve(H_strong, psi0, tlist_fine, c_ops, e_ops=[sz1, sz2])
me_global_res = mesolve(H_strong, psi0, tlist_fine, co_list, e_ops=[sz1, sz2])
br_res = brmesolve(
    H_strong,
    psi0,
    tlist_fine,
    e_ops=[sz1, sz2],
    a_ops=[[sx1, power_spectrum], [sx2, power_spectrum]],
)
```

```python
plt.plot(tlist_fine, me_local_res.expect[0], label=r"Local Lindblad")
plt.plot(tlist_fine, me_global_res.expect[0], "--", label=r"Dressed Lindblad")
plt.plot(tlist_fine, br_res.expect[0], ":", label=r"Bloch-Redfield")
plt.title("Strong Coupling")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(1)} \rangle$")
plt.legend()
plt.show()
```

### 手工构造 Liouvillian 超算符

如前所述，QuTiP 主方程求解器也能处理其他形式主方程。
通过 `spre()`、`spost()` 和 `sprepost()` 可手工搭建相应方程。
这些函数把原始希尔伯特空间算符映射到“双空间”算符，QuTiP 内部在该空间中优化计算（细节见 [QuTiPv5 论文](#References)）。

例如，上例主方程的 Lindbladian 可手工写为：


```python
lindbladian = -1.0j * (spre(H) - spost(H))
for c in c_ops:
    lindbladian += sprepost(c, c.dag())
    lindbladian -= 0.5 * (spre(c.dag() * c) + spost(c.dag() * c))
```

```python
manual_res = mesolve(lindbladian, psi0, tlist_fine, [], e_ops=[sz1, sz2])
```

```python
plt.plot(tlist_fine, me_local_res.expect[0], label="i=1")
plt.plot(tlist_fine, me_local_res.expect[1], label="i=2")
plt.xlabel("Time")
plt.ylabel(r"$\langle \sigma_z^{(i)} \rangle$")
plt.legend()
plt.show()
```

## Part 2：含时系统

最后比较另一类示例：受驱系统。
先用 `mesolve()` 在“有/无旋波近似（RWA）”下求解，再在“无 RWA”下与 Bloch-Redfield 和 HEOM 对比。
考虑哈密顿量

$$ H = \frac{\Delta}{2} \sigma_z + \frac{A}{2} \sin (\omega_d t) \sigma_x , $$

其中 $\Delta$ 为能级劈裂，$A$ 为驱动幅度，$\omega_d$ 为驱动频率。

```python
# Hamiltonian parameters
Delta = 2 * np.pi  # qubit splitting
omega_d = Delta  # drive frequency
A = 0.01 * Delta  # drive amplitude

# Bath parameters
gamma = 0.005 * Delta / (2 * np.pi)  # dissipation strength
temp = 0  # temperature

# Simulation parameters
psi0 = basis(2, 0)  # initial state
e_ops = [sigmaz()]
T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 1000 * T, 500)
```

### `mesolve`（无 RWA）

```python
# driving field
def f(t):
    return np.sin(omega_d * t)
```

```python
H0 = Delta / 2.0 * sigmaz()
H1 = [A / 2.0 * sigmax(), f]
H = [H0, H1]
```

```python
c_ops_me = [np.sqrt(gamma) * sigmam()]
```

```python
driv_res = mesolve(H, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)
```

### 旋波近似下的 `mesolve`

```python
H_RWA = (Delta - omega_d) * 0.5 * sigmaz() + A / 4 * sigmax()
c_ops_me_RWA = [np.sqrt(gamma) * sigmam()]
```

```python
driv_RWA_res = mesolve(H_RWA, psi0, tlist, c_ops=c_ops_me_RWA, e_ops=e_ops)
```

### `brmesolve`

```python
# Bose einstein distribution
def nth(w):
    if temp > 0:
        return 1 / (np.exp(w / temp) - 1)
    else:
        return 0


# Power spectrum
def power_spectrum(w):
    if w > 0:
        return gamma * (nth(w) + 1)
    elif w == 0:
        return 0
    else:
        return gamma * nth(-w)
```

```python
a_ops = [[sigmax(), power_spectrum]]
```

```python
driv_br_res = brmesolve(H, psi0, tlist, a_ops, sec_cutoff=-1, e_ops=e_ops)
```

### `HEOMSolver`

```python
max_depth = 4  # number of hierarchy levels

wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi
gamma_heom = 1.9 * w0

lambd = np.sqrt(
    0.5
    * gamma
    * ((w0**2 - wsamp**2) ** 2 + (gamma_heom**2) * ((wsamp) ** 2))
    / (gamma_heom * wsamp)
)
```

```python
# Create Environment
bath = UnderDampedEnvironment(lam=lambd, w0=w0, gamma=gamma_heom, T=0)
fit_times = np.linspace(0, 5, 1000)  # range for correlation function fit

# Fit correlation function with exponentials
exp_bath, fit_info = bath.approx_by_cf_fit(
    fit_times, Ni_max=1, Nr_max=2, target_rmse=None
)
print(fit_info["summary"])
```

```python
HEOM_corr_fit = HEOMSolver(
    QobjEvo(H),
    (exp_bath, sigmax()),
    max_depth=max_depth,
    options={"nsteps": 15000, "rtol": 1e-12, "atol": 1e-12},
)
results_corr_fit = HEOM_corr_fit.run(psi0 * psi0.dag(), tlist, e_ops=e_ops)
```

### 求解器结果对比

```python
plt.figure()

plt.plot(tlist, driv_res.expect[0], "-", label="mesolve (time-dep)")
plt.plot(tlist, driv_RWA_res.expect[0], "-.", label="mesolve (rwa)")
plt.plot(tlist, np.real(results_corr_fit.expect[0]), "--", label=r"heomsolve")
plt.plot(tlist, driv_br_res.expect[0], ":", linewidth=3, label="brmesolve")

plt.xlabel(r"$t\, /\, \Delta^{-1}$")
plt.ylabel(r"$\langle \sigma_z \rangle$")
plt.legend()
plt.show()
```

### 绝热能级切换

为展示局域基塌缩算符何时会导致误差，我们看单比特系统。
这次量子比特能级会在正负之间绝热切换：

$$H = \dfrac{\Delta}{2} \sin{(\omega_d t)} \sigma_z.$$ 

若驱动足够慢，热浴应能响应这一变化，并诱导高能级向低能级跃迁。
因此在 `mesolve()` 中也应考虑系统能量变化，才能得到正确结果。

```python
# Hamiltonian
omega_d = 0.05 * Delta  # drive frequency
A = Delta  # drive amplitude
H_adi = [[A / 2.0 * sigmaz(), f]]

# Bath parameters
gamma = 0.05 * Delta / (2 * np.pi)

# Simulation parameters
T = 2 * np.pi / omega_d  # period length
tlist = np.linspace(0, 2 * T, 400)
```

```python
# Simple mesolve
c_ops_me = [np.sqrt(gamma) * sigmam()]
adi_me_res = mesolve(H_adi, psi0, tlist, c_ops=c_ops_me, e_ops=e_ops)
```

```python
# HEOM
max_depth = 4  # number of hierarchy levels

wsamp = 2 * np.pi
w0 = 5 * 2 * np.pi
gamma_heom = 1.9 * w0

lambd = np.sqrt(
    0.5
    * gamma
    / (gamma_heom * wsamp)
    * ((w0**2 - wsamp**2) ** 2 + (gamma_heom**2) * ((wsamp) ** 2))
)
```

```python
# Create Environment
bath = UnderDampedEnvironment(lam=lambd, w0=w0, gamma=gamma_heom, T=0)
fit_times = np.linspace(0, 5, 1000)  # range for correlation function fit

# Fit correlation function with exponentials
exp_bath, fit_info = bath.approx_by_cf_fit(
    fit_times, Ni_max=1, Nr_max=2, target_rmse=None
)
print(fit_info["summary"])
```

```python
HEOM_corr_fit = HEOMSolver(
    QobjEvo(H_adi),
    (exp_bath, sigmax()),
    max_depth=max_depth,
    options={"nsteps": 15000, "rtol": 1e-12, "atol": 1e-12},
)
adi_corr_fit_res = HEOM_corr_fit.run(psi0 * psi0.dag(), tlist, e_ops=e_ops)
```

```python
# BRSolve non-flat power spectrum
a_ops_non_flat = [[sigmax(), exp_bath]]
brme_result = brmesolve(H_adi, psi0, tlist, a_ops=a_ops_non_flat, e_ops=e_ops)
```

```python
# BRSolve
a_ops = [[sigmax(), power_spectrum]]
brme_result2 = brmesolve(H_adi, psi0, tlist, a_ops=a_ops, e_ops=e_ops)
```

```python
plt.plot(tlist, adi_me_res.expect[0], "-", label="mesolve")
plt.plot(tlist, np.real(adi_corr_fit_res.expect[0]), "--", label=r"heom")
plt.plot(tlist, brme_result.expect[0], ":", linewidth=6, label="br non-flat")
plt.plot(tlist, brme_result2.expect[0], ":", linewidth=6, label="br")

plt.xlabel(r"$t\, /\, \Delta^{-1}$", fontsize=18)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=18)
plt.legend()
plt.show()
```

## 参考文献

\[1\] [QuTiP 5: The Quantum Toolbox in Python](https://arxiv.org/abs/2412.04705)


## 关于

```python
about()
```

## 测试

```python
br_exp = br_res.expect[0]
mg_exp = me_global_res.expect[0]
np.testing.assert_allclose(br_exp, mg_exp, atol=0.02, rtol=0)

d_exp = driv_res.expect[0]
dr_exp = driv_RWA_res.expect[0]
rc_exp = np.real(results_corr_fit.expect[0])
db_exp = driv_br_res.expect[0]
np.testing.assert_allclose(d_exp, dr_exp, atol=0.01, rtol=0)
np.testing.assert_allclose(dr_exp, rc_exp, atol=0.02, rtol=0)
np.testing.assert_allclose(rc_exp, db_exp, atol=0.02, rtol=0)
np.testing.assert_allclose(db_exp, d_exp, atol=0.01, rtol=0)
```
