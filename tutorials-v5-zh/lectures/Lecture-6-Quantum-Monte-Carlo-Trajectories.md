---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 讲座 6 - 量子蒙特卡洛轨迹


Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

本系列讲座由 J.R. Johansson 开发，原始讲义 notebook 可在[这里](https://github.com/jrjohansson/qutip-lectures)查看。

这里是为适配当前 QuTiP 版本而稍作修改的版本。
你可以在 [qutip-tutorials 仓库](https://github.com/qutip/qutip-tutorials)中找到这些讲座。
本讲座及其他教程 notebook 的索引页见 [QuTiP Tutorial 网页](https://qutip.org/tutorials.html)。

```python
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import about, basis, destroy, expect, mcsolve, mesolve, steadystate

%matplotlib inline
```

## 量子蒙特卡洛轨迹方法简介

量子蒙特卡洛轨迹方法用于描述与环境相互作用的量子系统中，
单次实现态矢量 $\left|\psi(t)\right>$ 的运动方程。
其波函数动力学由薛定谔方程给出：

<center>
$\displaystyle\frac{d}{dt}\left|\psi(t)\right> = - \frac{i}{\hbar} H_{\rm eff} \left|\psi(t)\right>$
</center>

其中哈密顿量是有效哈密顿量，除系统哈密顿量 $H(t)$ 外，
还包含来自系统-环境相互作用的非厄米项：

<center>
$\displaystyle H_{\rm eff}(t) = H(t) - \frac{i\hbar}{2}\sum_n c_n^\dagger c_n$
</center>

由于有效哈密顿量非厄米，波函数范数会随时间下降。
对小时间步 $\delta t$ 的一阶近似为
$\langle\psi(t+\delta t)|\psi(t+\delta t)\rangle \approx 1 - \delta p\;\;\;$，其中

<center>
$\displaystyle \delta p = \delta t \sum_n \left<\psi(t)|c_n^\dagger c_n|\psi(t)\right>$
</center>

利用范数下降来判断何时施加“量子跃迁”：
将 $\delta p$ 与区间 [0,1] 的随机数比较。
若范数低于随机阈值，则在 $t+\delta t$ 施加一次跃迁，
新波函数为

<center>
$\left|\psi(t+\delta t)\right> = c_n \left|\psi(t)\right>/\left<\psi(t)|c_n^\dagger c_n|\psi(t)\right>^{1/2}$ 
</center>

其中塌缩算符 $c_n$ 随机选择，且选择概率加权为

<center>
$\displaystyle P_n = \left<\psi(t)|c_n^\dagger c_n|\psi(t)\right>/{\delta p}$ 
</center>



## 腔中单光子 Fock 态衰减

该蒙特卡洛仿真展示了热环境中腔 Fock 态 $\left|1\right>$ 的衰减，
热环境平均占据数为 $n=0.063$。

耦合强度取腔衰减时间常数 $T_c = 0.129$ 的倒数。

参数对应 S. Gleyzes 等在 Nature 446, 297 (2007) 的实验，
我们将进行与该论文实验结果对应的仿真：

```python
Image(filename="images/exdecay.png")
```

### 问题参数

```python
N = 4  # number of basis states to consider
kappa = 1.0 / 0.129  # coupling to heat bath
nth = 0.063  # temperature with <n>=0.063

tlist = np.linspace(0, 0.6, 100)
```

## 创建算符、哈密顿量与初态

这里创建该问题所需算符和状态的 QuTiP `Qobj` 表示。

```python
a = destroy(N)  # cavity destruction operator
H = a.dag() * a  # harmonic oscillator Hamiltonian
psi0 = basis(N, 1)  # initial Fock state with one photon: |1>
```

## 创建描述耗散的塌缩算符列表

```python
# collapse operator list
c_op_list = []

# decay operator
c_op_list.append(np.sqrt(kappa * (1 + nth)) * a)

# excitation operator
c_op_list.append(np.sqrt(kappa * nth) * a.dag())
```

## 蒙特卡洛仿真

开始蒙特卡洛仿真，
并分别在 1、5、15、904 条轨迹下计算光子数算符期望值
（与上方实验结果比较）。

```python
ntraj = [1, 5, 15, 904]  # list of number of trajectories to avg. over
mc = []

for n in ntraj:
    result = mcsolve(H, psi0, tlist, c_op_list, e_ops=[a.dag() * a], ntraj=n)
    mc.append(result)
```

此时期望值位于数组 ``mc.expect[idx][0]`` 中，
其中 ``idx`` 取 ``[0,1,2,3]``，对应平均轨迹数 ``1, 5, 15, 904``。
下面绘制每个 ``idx`` 下 ``mc.expect[idx][0]`` 与 ``tlist`` 的关系。


## Lindblad 主方程仿真与稳态

为了与蒙特卡洛轨迹平均结果比较，
这里也计算 Lindblad 主方程动力学。
理论上当轨迹数趋于无穷时，
其结果应与蒙特卡洛平均一致。

```python
# run master equation to get ensemble average expectation values
me = mesolve(H, psi0, tlist, c_op_list, e_ops=[a.dag() * a])

# calculate final state using steadystate solver
final_state = steadystate(H, c_op_list)  # find steady-state
# find expectation value for particle number
fexpt = expect(a.dag() * a, final_state)
```

## 绘制结果

```python
leg_prop = matplotlib.font_manager.FontProperties(size=10)

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 12))

fig.subplots_adjust(hspace=0.1)  # reduce space between plots

for idx, n in enumerate(ntraj):

    axes[idx].step(tlist, mc[idx].expect[0], "b", lw=2)
    axes[idx].plot(tlist, me.expect[0], "r--", lw=1.5)
    axes[idx].axhline(y=fexpt, color="k", lw=1.5)

    axes[idx].set_yticks(np.linspace(0, 2, 5))
    axes[idx].set_ylim([0, 1.5])
    axes[idx].set_ylabel(r"$\left<N\right>$", fontsize=14)

    if idx == 0:
        axes[idx].set_title("Ensemble Averaging of Monte Carlo Trajectories")
        axes[idx].legend(
            ("Single trajectory", "master equation", "steady state"),
            prop=leg_prop
        )
    else:
        axes[idx].legend(
            ("%d trajectories" % n, "master equation", "steady state"),
            prop=leg_prop
        )

axes[3].xaxis.set_major_locator(plt.MaxNLocator(4))
axes[3].set_xlabel("Time (sec)", fontsize=14);
```

### 软件版本：

```python
about()
```
