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

# 稳态：同相干涉后的 Jaynes-Cummings 发射

K.A. Fischer, Stanford University

updated by A.V. Domingues, University of Sao Paulo

本 Jupyter 笔记展示如何模拟失谐 Jaynes-Cummings 系统在同相干涉（homodyne）后的量子统计性质。
目标是评估：耗散 Jaynes-Cummings 系统的第一极化激元是否能充当理想二能级系统。
本笔记紧随论文示例：[An architecture for self-homodyned nonclassical light](https://arxiv.org/abs/1611.01566), Phys. Rev. Applied 7, 044002 (2017)。

QuTiP 更多信息见：<http://qutip.org/>

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import about, destroy, expect, mesolve, qeye, steadystate, tensor

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

## 二能级系统（TLS）简介

量子二能级系统（TLS）是量子光-物质相互作用最基础的模型。
本例中，系统由连续模相干态驱动，其偶极相互作用哈密顿量为

$$ H_\mathrm{TLS} =\hbar \omega_0 \sigma^\dagger \sigma + \frac{\hbar\Omega_\mathrm{TLS}(t)}{2}\left( \sigma\textrm{e}^{-i\omega_dt} + \sigma^\dagger \textrm{e}^{i\omega_dt}\right),$$

其中 $\omega_0$ 为跃迁频率，$\sigma$ 为降算符，$\omega_d$ 为驱动中心频率，$\Omega_\mathrm{TLS}(t)$ 为驱动强度。

通过旋转坐标变换可去掉显式含时项。尤其在共振驱动（$\omega_d=\omega_0$）时最简单：

$$ \tilde{H}_\mathrm{TLS} =\frac{\hbar\Omega(t)}{2}\left( \sigma+ \sigma^\dagger \right).$$

### 设置 TLS 参数

```python
# define system operators
gamma = 1  # decay rate
sm_TLS = destroy(2)  # dipole operator
c_op_TLS = [np.sqrt(gamma) * sm_TLS]  # represents spontaneous emission

# choose range of driving strengths to simulate
Om_list_TLS = gamma * np.logspace(-2, 1, 300)

# calculate steady-state density matricies for the driving strengths
rho_ss_TLS = []
for Om in Om_list_TLS:
    H_TLS = Om * (sm_TLS + sm_TLS.dag())
    rho_ss_TLS.append(steadystate(H_TLS, c_op_TLS))
```

发射可分解为“相干部分”和“非相干部分”。
相干部分来自偶极矩经典均值：

$$I_\mathrm{c}=\lim_{t\rightarrow\infty}\Gamma\langle\sigma^\dagger(t)\rangle\langle\sigma(t)\rangle,$$

非相干部分来自偶极矩标准差（即量子涨落）：

$$I_\mathrm{inc}=\lim_{t\rightarrow\infty}\Gamma\langle\sigma^\dagger(t)\sigma(t)\rangle-I_\mathrm{c}.$$ 

二者共同作用，使理想 TLS 满足二阶相干函数 $g^{(2)}(0)=0$。

```python
# decompose the emitted light into the coherent and incoherent
# portions
I_c_TLS = expect(sm_TLS.dag(), rho_ss_TLS) * expect(sm_TLS, rho_ss_TLS)
I_inc_TLS = expect(sm_TLS.dag() * sm_TLS, rho_ss_TLS) - I_c_TLS
```

### 可视化相干/非相干发射

```python
plt.semilogx(Om_list_TLS, abs(I_c_TLS), label=r"TLS $I_\mathrm{c}$")
plt.semilogx(Om_list_TLS, abs(I_inc_TLS), "r", label=r"TLS $I_\mathrm{inc}$")
plt.xlabel("Driving strength [$\\Gamma$]")
plt.ylabel("Normalized flux [$\\Gamma$]")
plt.legend(loc=2);
```

## Jaynes-Cummings 系统简介

Jaynes-Cummings（JC）系统是量子光-物质相互作用最基本模型之一：一个二能级系统（如原子跃迁）与单光学模耦合。
在强耦合下，光与物质形成极化激元，能级呈非谐梯形结构。
“光子阻塞”利用其中最非谐的极化激元，期望产生 $g^{(2)}(0)<1$ 的发射。
我们将通过比较相干/非相干分量和 $g^{(2)}(0)$，评估其与理想 TLS 的接近程度。

本例中，JC 系统由连续模相干态驱动，哈密顿量为

$$ H =\hbar \omega_a a^\dagger a + \hbar \left(\omega_a+\Delta\right) \sigma^\dagger \sigma+ \hbar g\left(a^\dagger\sigma +a\sigma^\dagger\right) + \frac{\hbar\Omega(t)}{2}\left( a\textrm{e}^{-i\omega_dt} + a^\dagger \textrm{e}^{i\omega_dt}\right),$$

其中 $\omega_a$ 为腔共振频率，$\Delta$ 为腔-原子失谐。我们选择有限 $\Delta$，因为这会增强 JC 梯形的非谐性。
通过旋转坐标变换也可像前面一样去除含时项。

### 设置 JC 参数

```python
# truncate size of cavity's Fock space
N = 15

# setup system operators
sm = tensor(destroy(2), qeye(N))
a = tensor(qeye(2), destroy(N))

# define system parameters, barely into strong coupling regime
kappa = 1
g = 0.6 * kappa
detuning = 3 * g  # cavity-atom detuning
delta_s = detuning / 2 + np.sqrt(detuning**2 / 4 + g**2)

# we only consider cavities in the good-emitter limit, where
# the atomic decay is irrelevant
c_op = [np.sqrt(kappa) * a]
```

### 有效极化激元二能级系统

理想情况下，最非谐极化激元与基态形成一个理想二能级系统，其有效发射率为

$$\Gamma_\mathrm{eff}= \frac{\kappa}{2}+2\,\textrm{Im} \left\{\sqrt{ g^2-\left( \frac{\kappa}{4}+\frac{\textbf{i}\Delta}{2} \right)^2 }\right\}.$$ 

```python
effective_gamma = kappa / 2 + 2 * np.imag(
    np.sqrt(g**2 - (kappa / 4 + 1j * detuning / 2) ** 2)
)

# set driving strength based on the effective polariton's
# emission rate (driving strength goes as sqrt{gamma})
Om = 0.4 * np.sqrt(effective_gamma)
```

### 定义同相干涉参考系统

为实现 JC 输出的最优同相干涉，希望让光通过一个“裸腔”（无原子）并计算其相干振幅。
（该量当然可解析算出，但 QuTiP 同样可以轻松数值得到。）

```python
# reference cavity operator
a_r = destroy(N)
c_op_r = [np.sqrt(kappa) * a_r]

# reference cavity Hamiltonian, no atom coupling
H_c = Om * (a_r + a_r.dag()) + delta_s * a_r.dag() * a_r

# solve for coherent state amplitude at driving strength Om
rho_ss_c = steadystate(H_c, c_op_r)
alpha = -expect(rho_ss_c, a_r)
alpha_c = alpha.conjugate()
```

### 计算 JC 发射

JC 系统稳态发射通量为 $T=\kappa\langle a^\dagger a \rangle$。
加入同相干涉后，变为 $T=\langle b^\dagger b \rangle$，其中
$b=\sqrt{\kappa}/2\, a + \beta$，表示 JC 发射与振幅为 $\beta$ 的相干场干涉后的算符。

算符 $b$ 允许调节被测得的相干散射部分，而不会改变非相干部分（因为入射通量仅有相干成分）。
我们关心最优同相干涉，使 JC 发射尽可能逼近 TLS 发射。
该最优值由参考腔给出：$\beta=-\sqrt{\kappa}/2\langle a_\textrm{ref} \rangle$。

```python
def calculate_rho_ss(delta_scan):
    H = (
        Om * (a + a.dag())
        + g * (sm.dag() * a + sm * a.dag())
        + delta_scan * (sm.dag() * sm + a.dag() * a)
        - detuning * sm.dag() * sm
    )
    return steadystate(H, c_op)


delta_list = np.linspace(-6 * g, 9 * g, 200)
rho_ss = [calculate_rho_ss(delta) for delta in delta_list]

# calculate JC emission
I_jc = expect(a.dag() * a, rho_ss)

# calculate JC emission homodyned with optimal state beta
I_int = expect((a.dag() + alpha_c) * (a + alpha), rho_ss)
```

### 可视化有/无干涉时的发射通量

黑色虚线表示无干涉强度，紫色实线表示加入干涉后的强度。
灰色竖线标出非谐极化激元谱位置。
其线宽更窄，源于更慢的有效衰减率（在 good-emitter 极限下更“原子化”）。

```python
plt.figure(figsize=(8, 5))

plt.plot(delta_list / g, I_jc / effective_gamma, "k", linestyle="dashed", label="JC")
plt.plot(
    delta_list / g, I_int / effective_gamma, "blueviolet", label="JC w/ interference"
)
plt.vlines(delta_s / g, 0, 0.7, "gray")
plt.xlim(-6, 9)
plt.ylim(0, 0.7)
plt.xlabel("Detuning [g]")
plt.ylabel(r"Noramlized flux [$\Gamma_\mathrm{eff}$]")
plt.legend(loc=1);
```

### 计算 JC 发射的相干/非相干分量及 $g^{(2)}(0)$

注意

$$g^{(2)}(0)=\frac{\langle a^\dagger a^\dagger a a \rangle}{\langle a^\dagger a \rangle^2}.$$ 

```python
Om_list = kappa * np.logspace(-2, 1, 300) * np.sqrt(effective_gamma)
```

```python
def calculate_rho_ss(Om):
    H = (
        Om * (a + a.dag())
        + g * (sm.dag() * a + sm * a.dag())
        + delta_s * (sm.dag() * sm + a.dag() * a)
        - detuning * sm.dag() * sm
    )
    return steadystate(H, c_op)


rho_ss = [calculate_rho_ss(Om) for Om in Om_list]

# decompose emission again into incoherent and coherent portions
I_c = expect(a.dag(), rho_ss) * expect(a, rho_ss)
I_inc = expect(a.dag() * a, rho_ss) - I_c

# additionally calculate g^(2)(0)
g20 = expect(a.dag() * a.dag() * a * a, rho_ss) / expect(a.dag() * a, rho_ss) ** 2
```

### 可视化结果

上图黑色虚线是相干分量，可见在大驱动下占主导；此时与会饱和的 TLS 发射差异显著。
JC 不饱和源于非谐极化激元之上的近谐梯形。
此外 $g^{(2)}(0)$（下图）相较理想 TLS 的 0 仍偏大。

```python
plt.figure(figsize=(8, 8))

plt.subplot(211)
plt.semilogx(
    Om_list / np.sqrt(effective_gamma),
    abs(I_c) / kappa,
    "k",
    linestyle="dashed",
    label=r"JC $I_\mathrm{c}$",
)
plt.semilogx(
    Om_list / np.sqrt(effective_gamma),
    abs(I_inc) / kappa,
    "r",
    linestyle="dashed",
    label=r"JC $I_\mathrm{inc}$",
)
plt.xlabel(r"Driving strength [$\Gamma_\mathrm{eff}$]")
plt.ylabel(r"Normalized Flux [$\kappa$]")
plt.legend(loc=2)

plt.subplot(212)
plt.loglog(Om_list / np.sqrt(effective_gamma), g20, "k", linestyle="dashed")
lim = (1e-4, 2e0)
plt.ylim(lim)
plt.xlabel(r"Driving strength [$\Gamma_\mathrm{eff}$]")
plt.ylabel("$g^{(2)}(0)$");
```

### 计算同相干涉后的 JC 发射

现在用算符 $b$（而非 $\sqrt{\kappa}/2\,a$）重新计算相干/非相干分量及 $g^{(2)}(0)$：

$$g^{(2)}(0)=\frac{\langle b^\dagger b^\dagger b b \rangle}{\langle b^\dagger b \rangle^2}.$$ 

```python
def calculate_rho_ss_c(Om):
    H_c = Om * (a_r + a_r.dag()) + delta_s * a_r.dag() * a_r
    return steadystate(H_c, c_op_r)


rho_ss_c = [calculate_rho_ss_c(Om) for Om in Om_list]

# calculate list of interference values for all driving strengths
alpha_list = -np.array(expect(rho_ss_c, a_r))
alpha_c_list = alpha_list.conjugate()

# decompose emission for all driving strengths
g20_int = []
I_c_int = []
I_inc_int = []
for i, rho in enumerate(rho_ss):
    g20_int.append(
        expect(
            (a.dag() + alpha_c_list[i])
            * (a.dag() + alpha_c_list[i])
            * (a + alpha_list[i])
            * (a + alpha_list[i]),
            rho,
        )
        / expect((a.dag() + alpha_c_list[i]) * (a + alpha_list[i]), rho) ** 2
    )
    I_c_int.append(
        expect(a.dag() + alpha_c_list[i], rho) * expect(a + alpha_list[i], rho)
    )
    I_inc_int.append(
        expect((a.dag() + alpha_c_list[i]) * (a + alpha_list[i]), rho) - I_c_int[-1]
    )
```

### 结果分析

此时代表 TLS 分解的红/蓝虚线，与“最优同相干涉后”的 JC 分解（红/蓝实线）匹配良好。
黑色虚线仍显示“无干涉”的 JC 相干分量，提醒其在大驱动下不饱和。
此外，加入干涉后 $g^{(2)}(0)$ 改善了多个数量级。

```python
plt.figure(figsize=(8, 8))

plt.subplot(211)
plt.semilogx(Om_list_TLS, abs(I_c_TLS), linestyle="dashed", label=r"TLS $I_\mathrm{c}$")
plt.semilogx(
    Om_list_TLS, abs(I_inc_TLS), "r", linestyle="dashed", label=r"TLS $I_\mathrm{inc}$"
)
plt.semilogx(
    Om_list / np.sqrt(effective_gamma),
    abs(I_c / effective_gamma),
    "k",
    linestyle="dashed",
    label=r"JC $I_\mathrm{c}$",
)
plt.semilogx(
    Om_list / np.sqrt(effective_gamma),
    abs(I_inc / effective_gamma),
    "r",
    label=r"JC $I_\mathrm{inc}$",
)
plt.semilogx(
    Om_list / np.sqrt(effective_gamma),
    abs(I_c_int / effective_gamma),
    "b",
    label=r"JC w/ homodyne $I_\mathrm{c}$",
)
plt.semilogx(Om_list / np.sqrt(effective_gamma), abs(I_inc_int / effective_gamma), "r")
plt.ylim(5e-4, 0.6)
plt.xlabel(r"Driving strength [$\Gamma_\mathrm{eff}$]")
plt.ylabel(r"Normalized flux [$\Gamma_\mathrm{eff}$]")
plt.legend(loc=2)

plt.subplot(212)
plt.loglog(Om_list / np.sqrt(effective_gamma), g20, "k", linestyle="dashed", label="JC")
plt.loglog(
    Om_list / np.sqrt(effective_gamma),
    g20_int,
    "blueviolet",
    label="JC w/ interference",
)
plt.ylim(lim)
plt.xlabel(r"Driving strength [$\Gamma_\mathrm{eff}$]")
plt.ylabel(r"$g^{(2)}(0)$")
plt.legend(loc=4);
```

### 含延迟的二阶相干函数

进一步考虑随延迟时间变化的二阶相干函数：

$$g^{(2)}(\tau)=\lim_{t\rightarrow\infty}\frac{\langle b^\dagger(t)b^\dagger(t+\tau)b(t+\tau)b(t)\rangle}{\langle b^\dagger(t)b(t)\rangle^2},$$

并演示在同相干涉语境下的计算方式。

```python
# first calculate the steady state
H = (
    Om * (a + a.dag())
    + g * (sm.dag() * a + sm * a.dag())
    + delta_s * (sm.dag() * sm + a.dag() * a)
    - detuning * sm.dag() * sm
)
rho0 = steadystate(H, c_op)

taulist = np.linspace(0, 5 / effective_gamma, 1000)

# next evolve the states according the quantum regression theorem

# ...with the b operator
corr_vec_int = expect(
    (a.dag() + alpha.conjugate()) * (a + alpha),
    mesolve(
        H,
        (a + alpha) * rho0 * (a.dag() + alpha.conjugate()),
        taulist,
        c_op,
        [],
        options={"atol": 1e-13, "rtol": 1e-11},
    ).states,
)
n_int = expect(rho0, (a.dag() + alpha.conjugate()) * (a + alpha))

# ...with the a operator
corr_vec = expect(
    a.dag() * a,
    mesolve(
        H,
        a * rho0 * a.dag(),
        taulist,
        c_op,
        [],
        options={"atol": 1e-12, "rtol": 1e-10},
    ).states,
)
n = expect(rho0, a.dag() * a)

# ...perform the same for the TLS comparison
H_TLS = Om * (sm_TLS + sm_TLS.dag()) * np.sqrt(effective_gamma)
c_ops_TLS = [sm_TLS * np.sqrt(effective_gamma)]
rho0_TLS = steadystate(H_TLS, c_ops_TLS)
corr_vec_TLS = expect(
    sm_TLS.dag() * sm_TLS,
    mesolve(H_TLS, sm_TLS * rho0_TLS * sm_TLS.dag(), taulist, c_ops_TLS, []).states,
)
n_TLS = expect(rho0_TLS, sm_TLS.dag() * sm_TLS)
```

### 与 TLS 关联函数对比

在中等驱动下，JC 关联函数（黑色虚线）明显偏离 TLS（紫色点虚线）。
而经过最优同相干涉后，发射关联（紫色实线）与理想 TLS 匹配良好。

```python
plt.figure(figsize=(8, 5))

(l1,) = plt.plot(
    taulist * effective_gamma,
    corr_vec_TLS / n_TLS**2,
    "blueviolet",
    linestyle="dotted",
    label="TLS",
)
plt.plot(
    taulist * effective_gamma, corr_vec / n**2, "k", linestyle="dashed", label="JC"
)
plt.plot(
    taulist * effective_gamma,
    corr_vec_int / n_int**2,
    "blueviolet",
    label="JC w/ interference",
)
plt.xlabel(r"$\tau$ [$1/\Gamma_\mathrm{eff}$]")
plt.ylabel(r"$g^{(2)}(\tau)$")
plt.legend(loc=2);
```

## 版本

```python
about()
```

## 测试

```python
# g20 of TLS should be approx. 0
g20_TLS = (
    expect(sm_TLS.dag() * sm_TLS.dag() * sm_TLS * sm_TLS, rho_ss_TLS)
    / expect(sm_TLS.dag() * sm_TLS, rho_ss_TLS) ** 2
)
assert np.allclose(g20_TLS, 0)

# g20 of homodyne interf. should be less than 1 while JC whitout it not necessarily
assert np.all(np.less(g20_int, 1))
assert not np.all(np.less(g20, 1))

# Homodyne interf. and TLS should match and the other pairs shouldn't
assert np.allclose(corr_vec_int / n_int**2, corr_vec_TLS / n_TLS**2, atol=1e-1)
assert not np.allclose(corr_vec_int / n_int**2, corr_vec / n**2, atol=1e-1)
assert not np.allclose(corr_vec_TLS / n_TLS**2, corr_vec / n**2, atol=1e-1)

# Traces of all states should be approx. 1
assert np.allclose(rho0.tr(), 1)
assert np.allclose(rho0_TLS.tr(), 1)
for rho in rho_ss:
    assert np.allclose(rho.tr(), 1)
for rho in rho_ss_c:
    assert np.allclose(rho.tr(), 1)
for rho in rho_ss_TLS:
    assert np.allclose(rho.tr(), 1)
```
