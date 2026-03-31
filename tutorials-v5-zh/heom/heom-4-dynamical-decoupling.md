---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# HEOM 4：非马尔可夫环境中的动力学退耦

+++

## 引言

参考 [Lorenza Viola 与 Seth Lloyd](https://arxiv.org/abs/quant-ph/9803057)，我们讨论一个动力学退耦示例。
我们选择执行 $\pi$ 旋转的驱动脉冲，并在脉冲之间保留短时间区间让热浴引起退相干。

先展示等间隔脉冲的标准案例，再考虑“最优” Uhrig 间隔（[Götz S. Uhrig, Phys. Rev. Lett. 98, 100504 (2007)](https://arxiv.org/abs/quant-ph/0609203)）。

+++

## 设置

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from qutip import (DrudeLorentzEnvironment, QobjEvo, about,
                   basis, expect, ket2dm, sigmax, sigmaz)
from qutip.solver.heom import HEOMSolver

from IPython.display import display
from ipywidgets import IntProgress

%matplotlib inline
```

## 求解器选项

```{code-cell} ipython3
# Solver options:

# The max_step must be set to a short time than the
# length of the shortest pulse, otherwise the solver
# might skip over a pulse.

options = {
    "nsteps": 1500,
    "store_states": True,
    "rtol": 1e-12,
    "atol": 1e-12,
    "max_step": 1 / 20.0,
    "method": "vern9",
    "progress_bar": "enhanced",
}
```

## 系统与热浴定义

现在定义系统、热浴与 HEOM 参数。系统是静止单量子比特（$H=0$），热浴为 Drude-Lorentz 谱的玻色热浴。

```{code-cell} ipython3
# Define the system Hamlitonian.
#
# The system isn't evolving by itself, so the Hamiltonian is 0 (with the
# correct dimensions):

H_sys = 0 * sigmaz()
```

```{code-cell} ipython3
# Define some operators with which we will measure the system
# 1,1 element of density matrix - corresponding to groundstate
P11p = basis(2, 0) * basis(2, 0).dag()
P22p = basis(2, 1) * basis(2, 1).dag()
# 1,2 element of density matrix  - corresponding to coherence
P12p = basis(2, 0) * basis(2, 1).dag()
```

```{code-cell} ipython3
# Properties for the Drude-Lorentz bath

lam = 0.0005
gamma = 0.005
T = 0.05

# bath-system coupling operator:
Q = sigmaz()

# number of terms to keep in the expansion of the bath correlation function:
Nk = 3

env = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T)
env_approx = env.approximate(method="pade", Nk=Nk)
```

```{code-cell} ipython3
# HEOM parameters

# number of layers to keep in the hierarchy:
NC = 6
```

为实现与环境的动力学退耦，我们对系统施加与 $\sigma_x$ 耦合的含时脉冲驱动。通常会把脉冲面积设为 $\pi/2$，使脉冲翻转量子比特态。

下面定义一个返回脉冲函数的函数：

```{code-cell} ipython3
def drive(amplitude, delay, integral):
    """ Coefficient of the drive as a function of time.

        The drive consists of a series of constant pulses with
        a fixed delay between them.

        Parameters
        ----------
        amplitude : float
            The amplitude of the drive during the pulse.
        delay : float
            The time delay between successive pulses.
        integral : float
            The integral of the pulse. This determines
            the duration of each pulse with the duration
            equal to the integral divided by the amplitude.
    """
    duration = integral / amplitude
    period = duration + delay

    def pulse(t):
        t = t % period
        if t < duration:
            return amplitude
        return 0

    return pulse


H_drive = sigmax()
```

## 绘制谱密度

先绘制 Drude-Lorentz 热浴的谱密度：

```{code-cell} ipython3
wlist = np.linspace(0, 0.5, 1000)
J = env.spectral_density(wlist)
J_approx = env_approx.spectral_density(wlist)

fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes.plot(wlist, J, "r", linewidth=2)
axes.plot(wlist, J_approx, "b--", linewidth=2)

axes.set_xlabel(r"$\omega$", fontsize=28)
axes.set_ylabel(r"J", fontsize=28);
```

## 快脉冲与慢脉冲下的动力学退耦

现在可以开始研究如何通过驱动实现环境退耦。

先用快速、大幅值脉冲，再用较慢、小幅值脉冲。快速脉冲退耦更有效、能更久保持相干；慢脉冲也有帮助但效果较弱。

先模拟快速脉冲：

```{code-cell} ipython3
# Fast driving (quick, large amplitude pulses)

tlist = np.linspace(0, 400, 1000)

# start with a superposition so there is something to dephase!
rho0 = (basis(2, 1) + basis(2, 0)).unit()
rho0 = ket2dm(rho0)

# without pulses
hsolver = HEOMSolver(H_sys, (env_approx, Q), NC, options=options)
outputnoDD = hsolver.run(rho0, tlist)

# with pulses
drive_fast = drive(amplitude=0.5, delay=20, integral=np.pi / 2)
H_d = QobjEvo([H_sys, [H_drive, drive_fast]])

hsolver = HEOMSolver(H_d, (env_approx, Q), NC, options=options)
outputDD = hsolver.run(rho0, tlist)
```

再看更长更慢的脉冲：

```{code-cell} ipython3
# Slow driving (longer, small amplitude pulses)

# without pulses
hsolver = HEOMSolver(H_sys, (env_approx, Q), NC, options=options)
outputnoDDslow = hsolver.run(rho0, tlist)

# with pulses
drive_slow = drive(amplitude=0.01, delay=20, integral=np.pi / 2)
H_d = QobjEvo([H_sys, [H_drive, drive_slow]])

hsolver = HEOMSolver(H_d, (env_approx, Q), NC, options=options)
outputDDslow = hsolver.run(rho0, tlist)
```

现在把所有结果与脉冲形状一起画出来：

```{code-cell} ipython3
def plot_dd_results(outputnoDD, outputDD, outputDDslow):
    fig, axes = plt.subplots(2, 1, sharex=False, figsize=(12, 12))

    # Plot the dynamic decoupling results:

    tlist = outputDD.times

    P12 = basis(2, 1) * basis(2, 0).dag()
    P12DD = expect(outputDD.states, P12)
    P12noDD = expect(outputnoDD.states, P12)
    P12DDslow = expect(outputDDslow.states, P12)

    plt.sca(axes[0])
    plt.yticks([0, 0.25, 0.5], [0, 0.25, 0.5])

    axes[0].plot(
        tlist, np.real(P12DD),
        'green', linestyle='-', linewidth=2, label="HEOM with fast DD",
    )
    axes[0].plot(
        tlist, np.real(P12DDslow),
        'blue', linestyle='-', linewidth=2, label="HEOM with slow DD",
    )
    axes[0].plot(
        tlist, np.real(P12noDD),
        'orange', linestyle='--', linewidth=2, label="HEOM no DD",
    )

    axes[0].locator_params(axis='y', nbins=3)
    axes[0].locator_params(axis='x', nbins=3)

    axes[0].set_ylabel(r"$\rho_{01}$", fontsize=30)

    axes[0].legend(loc=4)
    axes[0].text(0, 0.4, "(a)", fontsize=28)

    # Plot the drive pulses:

    pulse = [drive_fast(t) for t in tlist]
    pulseslow = [drive_slow(t) for t in tlist]

    plt.sca(axes[1])
    plt.yticks([0, 0.25, 0.5], [0, 0.25, 0.5])

    axes[1].plot(
        tlist, pulse,
        'green', linestyle='-', linewidth=2, label="Drive fast",
    )
    axes[1].plot(
        tlist, pulseslow,
        'blue', linestyle='--', linewidth=2, label="Drive slow",
    )

    axes[1].locator_params(axis='y', nbins=3)
    axes[1].locator_params(axis='x', nbins=3)

    axes[1].set_xlabel(r'$t\bar{V}_{\mathrm{f}}$', fontsize=30)
    axes[1].set_ylabel(r'Drive amplitude/$\bar{V}_{\mathrm{f}}$', fontsize=30)

    axes[1].legend(loc=1)
    axes[1].text(0, 0.4, "(b)", fontsize=28)

    fig.tight_layout()
```

```{code-cell} ipython3
plot_dd_results(outputnoDD, outputDD, outputDDslow)
```

## 非等间隔脉冲

+++

接下来考虑非等间隔脉冲。

我们不再画完整时间曲线，而是只考察总时间 $T$、100 个脉冲后的最终相干性。通过改变环境宽度来说明：当热浴很宽时，Uhrig 序列（即非均匀间隔）可能优于等间隔脉冲。

我们使用如下形式定义第 $j$ 个脉冲后的累计延迟（而非等间隔）：

$$
    \sin^2(\frac{\pi}{2} \frac{j}{N + 1})
$$

这只是描述变化延迟的一种便捷方式。也可以选用其它单调递增函数来表示累计延迟（但未必同样有效）。

```{code-cell} ipython3
def cummulative_delay_fractions(N):
    """ Return an array of N + 1 cummulative delay
        fractions.

        The j'th entry in the array should be the sum of
        all delays before the j'th pulse. The last entry
        should be 1 (i.e. the entire cummulative delay
        should have been used once the sequence of pulses
        is complete).

        The function should be monotonically increasing,
        strictly greater than zero and the last value
        should be 1.

        This implementation returns:

            sin((pi / 2) * (j / (N + 1)))**2

        as the cummulative delay after the j'th pulse.
    """
    return np.array(
        [np.sin((np.pi / 2) * (j / (N + 1)))**2 for j in range(0, N + 1)]
    )


def drive_opt(amplitude, avg_delay, integral, N):
    """ Return an optimized distance pulse function.

        Our previous pulses were evenly spaced. Here we
        instead use a varying delay after the j'th pulse.

        The cummulative delay is described by the function
        ``cummulative_delay_fractions`` above.
    """
    duration = integral / amplitude
    cummulative_delays = N * avg_delay * cummulative_delay_fractions(N)

    t_start = cummulative_delays + duration * np.arange(0, N + 1)
    t_end = cummulative_delays + duration * np.arange(1, N + 2)

    def pulse(t):
        if any((t_start <= t) & (t <= t_end)):
            return amplitude
        return 0.0

    return pulse
```

绘制累计延迟并查看其形状。可见累计延迟从 $0$ 开始，到 $1$ 结束，且单调递增，满足要求。

在同一坐标轴上，也绘制各个第 $j$ 个延迟相对于平均延迟的比例。

```{code-cell} ipython3
def plot_cummulative_delay_fractions(N):
    cummulative = cummulative_delay_fractions(N)
    individual = (cummulative[1:] - cummulative[:-1]) * N
    plt.plot(np.arange(0, N + 1), cummulative, label="Cumulative delay")
    plt.plot(np.arange(0, N), individual, label="j'th delay")
    plt.xlabel("j")
    plt.ylabel("Fraction of delay")
    plt.legend()


plot_cummulative_delay_fractions(100)
```

再把前十个等间隔脉冲与最优间隔脉冲画在一起比较：

```{code-cell} ipython3
def plot_even_and_optimally_spaced_pulses():
    amplitude = 10.0
    integral = np.pi / 2
    duration = integral / amplitude
    delay = 1.0 - duration

    tlist = np.linspace(0, 10, 1000)

    pulse_opt = drive_opt(amplitude, delay, integral, 100)
    pulse_eq = drive(amplitude, delay, integral)

    plt.plot(
        tlist, [pulse_opt(t) for t in tlist], label="opt",
    )
    plt.plot(
        tlist, [pulse_eq(t) for t in tlist], label="eq",
    )
    plt.legend(loc=4)


plot_even_and_optimally_spaced_pulses()
```

现在比较两套延迟方案在 100 个脉冲后维持相干性的效果。

我们将对一组 $\lambda$ 与 $\gamma$ 参数进行模拟，展示随着热浴谱函数宽度增加，非均匀间隔延迟何时变得更优。

```{code-cell} ipython3
# Bath parameters to simulate over:

# We use only two lambdas and two gammas so that the notebook executes
# quickly:

lams = [0.005, 0.0005]
gammas = np.linspace(0.005, 0.05, 2)

# But one can also extend the lists to larger ones:
#
# lams = [0.01, 0.005, 0.0005]
# gammas = np.linspace(0.005, 0.05, 10)

# Setup a progress bar:

progress = IntProgress(min=0, max=(2 * len(lams) * len(gammas)))
display(progress)


def simulate_100_pulses(lam, gamma, T, NC, Nk):
    """ Simulate the evolution of 100 evenly and optimally spaced pulses.

        Returns the expectation value of P12p from the final state of
        each evolution.
    """
    rho0 = (basis(2, 1) + basis(2, 0)).unit()
    rho0 = ket2dm(rho0)

    N = 100  # number of pulses to simulate
    avg_cycle_time = 1.0  # average time from one pulse to the next
    t_max = N * avg_cycle_time

    tlist = np.linspace(0, t_max, 100)

    amplitude = 10.0
    integral = np.pi / 2
    duration = integral / amplitude
    delay = avg_cycle_time - duration

    env = DrudeLorentzEnvironment(lam=lam, gamma=gamma, T=T)
    env_approx = env.approximate("pade", Nk=Nk)
    # Equally spaced pulses:

    pulse_eq = drive(amplitude, delay, integral)
    H_d = QobjEvo([H_sys, [H_drive, pulse_eq]])

    hsolver = HEOMSolver(H_d, (env_approx, Q), NC, options=options)
    result = hsolver.run(rho0, tlist)

    P12_eq = expect(result.states[-1], P12p)
    progress.value += 1

    # Non-equally spaced pulses:

    pulse_opt = drive_opt(amplitude, delay, integral, N)
    H_d = QobjEvo([H_sys, [H_drive, pulse_opt]])

    hsolver = HEOMSolver(H_d, (env_approx, Q), NC, options=options)
    result = hsolver.run(rho0, tlist)

    P12_opt = expect(result.states[-1], P12p)
    progress.value += 1

    return P12_opt, P12_eq


# We use NC=2 and Nk=2 to speed up the simulation:

P12_results = [
    list(zip(*(
        simulate_100_pulses(lam=lam_, gamma=gamma_, T=0.5, NC=2, Nk=2)
        for gamma_ in gammas
    )))
    for lam_ in lams
]
```

现在我们已经得到 $\rho_{01}$ 的期望值，接着按每个 $\lambda$ 绘制其随 $\gamma$ 的变化。可看到在每种情况下，当 $\gamma$ 足够小时，非均匀间隔脉冲会成为更优选择：

```{code-cell} ipython3
fig, axes = plt.subplots(1, 1, sharex=False, figsize=(10, 7))
colors = ["green", "red", "blue"]

for i in range(len(lams)):
    color = colors[i % len(colors)]
    axes.plot(
        gammas, np.real(P12_results[i][0]),
        color, linestyle='-', linewidth=2,
        label=f"Optimal DD [$\\lambda={lams[i]}$]",
    )
    axes.plot(
        gammas, np.real(P12_results[i][1]),
        color, linestyle='-.', linewidth=2,
        label=f"Even DD [$\\lambda={lams[i]}$]",
    )

axes.set_ylabel(r"$\rho_{01}$")
axes.set_xlabel(r"$\gamma$")
axes.legend(fontsize=16)

fig.tight_layout();
```

到这里，你已经了解了如何通过动力学退耦把量子比特与环境分离。

+++

## 关于

```{code-cell} ipython3
about()
```
