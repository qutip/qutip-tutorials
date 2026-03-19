---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 讲座：单光子干涉


作者：Amélie Orban（编辑）, Inga Schöyen, Alessandro Delmonte, Alexander Rihane。
<br> 本教程基于 2021 年 6 月在马斯特里赫特大学 Maastricht Science Programme 完成的项目 <em>使用 QuTiP（Python 量子工具箱）模拟单光子干涉</em>。
<br> 最后更新：09/11/21。

```python
import matplotlib.pyplot as plt
import qutip
from IPython.display import Image
from numpy import cos, exp, pi, random, sin, sqrt
from qutip import Qobj, basis

%matplotlib inline
```

### 目录
* [引言](#section0)
* [1. 理论背景](#section1)
* [2. 构成元件的代码实现](#section2)
* [3. 单光子干涉实验模拟](#section3)
* [4. 单光子干涉实验的变体](#section4)
* [结论与要点](#section5)


### 引言 <a class="anchor" id="section0"></a>


波粒二象性是量子力学的核心概念之一，它说明每个量子粒子（或量子对象）都可同时用“粒子性”和“波动性”来描述，而任何单一经典图像都不足以完整刻画量子对象。

类似于托马斯·杨在 1801 年进行的双缝实验，光的波粒二象性也可通过单光子干涉实验（此处为模拟）来观察。单个看似“粒子”的光子会与自身发生干涉，这是一种本质上“波动”的性质，因此它同时展现了粒子与波的特征。


本讲通过模拟一个单光子穿过偏振干涉仪的量子光学实验，研究单光子干涉现象。
* 如前所述，被模拟的实验与杨氏实验类似：它同样通过让光子走两条不同路径来产生干涉。经典图像中，一束光会被分成两束，其中一束走更长光程，随后借助分束器与偏振分析器再次合并。
* 路径长度变化会导致相位差，从而形成干涉条纹。
* 若把这一直觉直接套用到单光子情形，就会出现疑问：单光子是不可再分的光量子，要出现干涉似乎必须先被“分成两部分”。由于光子逐个通过装置且不可再分，表面上它们似乎无法彼此干涉。
* 但它们确实会干涉。标准解释是：发生干涉的是光子的概率波函数与其自身，而不是两个彼此独立的光子态之间的干涉。


### 1. 理论背景 <a class="anchor" id="section1"></a>


本节从理论角度介绍单光子干涉实验的装置，并说明需要模拟的各个光学元件。每个元件都用作用于量子态的 Jones 矩阵来表示。

该实验装置与相应理论主要参考 Mark Beck 的《Quantum Mechanics, Theory and Experiment》（*Beck, M. (2012). Quantum mechanics, theory and experiment. Oxford University Press.*）第 3 章实验 6。


#### 半波片 $\frac{\lambda}{2}$


波片利用介质对不同偏振方向折射率不同这一性质来改变光的偏振。
* 波片有两条互相正交的轴：快轴对应较小折射率（沿该方向传播更快），慢轴与快轴正交，折射率更大。
* 折射率差异会使两正交偏振分量在传播中获得不同相位，从而产生二者之间的相对相位差。
* 波片正是利用这种相对相位差来改变总偏振态。

此处使用半波片，其快慢轴之间的相对相位差等于半个波长。
* 半波片快轴与水平面夹角设为 $45^\circ$。
* 入射波的一半沿快轴形成 $|+45\rangle$ 偏振，另一半沿慢轴形成 $|-45\rangle$ 偏振，二者相对相位差为 $\pi$。
* 半波片的作用是按其取向将线偏振旋转一定角度（并不是把波真正分裂为两个正交分量）。

$45^\circ$ 取向半波片的数学表达式如下：

\begin{equation}
J_{\lambda/2 \hspace{1mm} 45^\circ} = \left[\begin{array}{cc} \cos(2\cdot45) & \sin(2\cdot45)  \\ \sin(2\cdot45) & -\cos(2\cdot45) \end{array}\right] = \left[\begin{array}{cc} 0 & 1  \\ 1 & 0 \end{array}\right]
\end{equation}


#### 偏振分析器 PA$_{HV}$ 与 PA$_{45}$


偏振分析器（也称偏振分束棱镜）既可将光束分解为两个正交偏振分量，也会使这两个分量产生不同的空间位移。
* 一般偏振光会被分解为沿不同方向传播的两束光。若分析器两出射面平行，则两束出射光互相平行但存在横向位移。

当偏振分析器相对水平面垂直放置，使水平传播光正入射时，会把光分成水平与垂直分量，此时记为 PA$_{HV}$。
* 在该情形下，$|V\rangle$ 分量直通不偏折，而 $|H\rangle$ 分量进出器件时会偏折，最终与 $|V\rangle$ 平行但有位移并带相位变化。
* 该元件也可用于把两正交分量重新合并，过程与分束相反。
* 表示 PA$_{HV}$ 的 Jones 矩阵由水平偏振器、垂直偏振器与相移矩阵组合而成，用以刻画其分束与相移作用：

\begin{equation}
J_{PA_{HV}} = J_{V} + J_{\phi}J_{H} = \left[\begin{array}{cc} 0 & 0  \\ 0 & 1 \end{array}\right] + \left[\begin{array}{cc} e^{i \phi} & 0  \\ 0 & 1 \end{array}\right]\left[\begin{array}{cc} 1 & 0  \\ 0 & 0 \end{array}\right] = \left[\begin{array}{cc} e^{i \phi} & 0  \\ 0 & 1 \end{array}\right]
\end{equation}

若将 PA$_{HV}$ 旋转 $45^\circ$，入射光会被分解为 $|+45\rangle$ 与 $|-45\rangle$ 分量，此时记为 PA$_{45}$。
* 在该情形下，$|+45\rangle$ 分量直通不偏折，而 $|-45\rangle$ 分量会偏折后与前者平行输出并产生位移。
* PA$_{45}$ 的 Jones 矩阵同样由多个矩阵组合：$45^\circ$ 与 $-45^\circ$ 线偏振器以及相移矩阵，其表达式为：

\begin{equation}
\begin{split}
J_{PA_{45}} &= J_{+45} + J_{\phi}J_{-45} \\
&= \left[\begin{array}{cc} \cos^2(45) & \cos(45)\sin(45)  \\ \cos(45)\sin(45) & \sin^2(45) \end{array}\right]
+ \left[\begin{array}{cc} e^{i \phi} & 0  \\ 0 & 1 \end{array}\right]\left[\begin{array}{cc} \cos^2(-45) & \cos(-45)\sin(-45)  \\ \cos(-45)\sin(-45) & \sin^2(-45) \end{array}\right] \\
&= \left[\begin{array}{cc} \frac{1}{2}(e^{i \phi} +1) & \frac{1}{2}(1-e^{i \phi})  \\ 0 & 1 \end{array}\right]
\end{split}
\end{equation}


#### 完整实验装置


下图所示实验装置流程如下。
* 首先制备 $|+45\rangle$ 偏振光子，并将其送入由 PA$_{HV}$、半波片、第二个 PA$_{HV}$ 以及末端 PA$_{45}$ 组成的干涉仪。
* 第一个 PA$_{HV}$ 把入射光子分解为 $|H\rangle$ 与 $|V\rangle$ 两个正交分量并产生位移。
* 两分量平行传播后通过半波片。加入半波片的作用之一是让两臂光程实现等效平衡。
* 如前所述，快轴为 $45^\circ$ 的半波片会把线偏振旋转 $90^\circ$，使 $|H\rangle$ 与 $|V\rangle$ 互换，从而使两臂行为更对称，并在初始设置下不引入相对相位差。
* 随后第二个 PA$_{HV}$ 将两分量重新合并。
* 最后，PA$_{45}$ 将末态分解到 $|+45\rangle$ 与 $|-45\rangle$ 两个端口，便于测量两个输出强度随相位变化的调制。

整个装置构成一个干涉仪。末端叠加态会发生干涉，其输出强度取决于二者相对相位（等价地，取决于两臂光程差）。
* 这个相对相位（初始为零）可通过调节 PA$_{HV}$ 改变。
* 倾斜一个或两个 PA$_{HV}$ 都会改变两臂相位差，因为相对相移与 PA$_{HV}$ 倾角成正比（其本质是改变光程）。
* 当 $\phi=0$ 时应出现相长干涉，输出集中在 $+45^\circ$ 端口；当 $\phi=\pi$ 时应出现相消干涉，输出集中在 $-45^\circ$ 端口。
* 之所以能观测到干涉，关键在于最后的 PA$_{45}$：它把原本正交（本来不会互相干涉）的偏振态投影到 $\pm45^\circ$ 轴上，从而显现干涉。


```python
Image(filename="images/single-photon-interference-setup.jpg", width=700, embed=True)
```


需要强调：通过这些元件的是单个光子，因此这是量子过程。在“分解为正交分量”这件事上，必须区分经典波与单光子。
* 经典波会被确定性地分为两部分，且分量大小按偏振展开系数比例分配（$\psi = c_H |H\rangle + c_V |V\rangle$）。
* 单光子则不同，由于不可再分，所谓“分束”体现为随机结果，无法确定每个光子必定从哪个端口输出。
* 一般态中 $|H\rangle$ 与 $|V\rangle$ 的权重表示光子走对应路径的**概率**。
* 同时要注意，光子态在整个干涉仪内并不会真的裂成两半，它始终是一个整体量子态。


### 2. 构成元件的代码实现 <a class="anchor" id="section2"></a>


#### 基矢
先定义偏振基矢：$H/V$ 基与 $\pm45$ 基。任意偏振态都可由这些基矢线性组合表示。

```python
# HV basis

# horizontal polarization
H = basis(2, 0)
# vertical polarization
V = basis(2, 1)
```

```python
# +45/-45 basis (in terms of HV basis)

# +45 polarization
p45 = 1 / sqrt(2) * (H + V)
# -45 polarization
n45 = 1 / sqrt(2) * (H - V)
```

#### 偏振分析器（HV）1 号
该元件使 H 偏振相对于 V 偏振产生相位差，来源于干涉仪两臂光程差。
<br> 对单光子而言，态并未真正分裂，装置中始终只有一个量子态，因此 PA$_{HV}$ 只改变各分量相位。

```python
# Polarization analyzer (HV) n掳1

phaseshift1 = pi / 4  # CONSTANT
# should depend on real size of the setup (here: arbitrarily chosen)
PA_HV1 = Qobj([[exp(1j * phaseshift1), 0], [0, 1]])
PA_HV1
```

#### 半波片 $\frac{\lambda}{2}$
这里快轴与水平面夹角为 $45^\circ$。
<br> 对单光子而言，通过该半波片的效果是 $|H\rangle$ 与 $|V\rangle$ 偏振分量互换。

```python
# Half-wave plate

胃 = pi / 4  # fast axis orientation
# (!) numpy calculates with rad
halfwave = Qobj([[cos(2 * 胃), sin(2 * 胃)], [sin(2 * 胃), -cos(2 * 胃)]])

"""
removes very small elements
(numerical artifacts from the finite precision of the computer)
"""
halfwave.tidyup()
```

#### 偏振分析器（HV）2 号
与前一个分析器类似，它也会让 H 偏振相对 V 偏振产生相位差。
* 初始时，PA$_{HV2}$ 的相移应设为与 PA$_{HV1}$ 相同。此时两者处于同一倾斜位置，总体相对相移为零。
* 随后可改变 PA$_{HV2}$ 的相移，以观察不同强度的干涉。
* 两个分析器引入相移的差值会改变分量间**相对**相位，而正是这个相对相位决定干涉行为。

```python
# Polarization analyzer (HV) n掳2

phaseshift2 = pi / 4  # CHANGE TO CHANGE INTERFERENCE
# should depend on real size of the setup
PA_HV2 = Qobj([[exp(1j * phaseshift2), 0], [0, 1]])

PA_HV2
```

#### 偏振分析器（45）
PA$_{45}$ 的作用是让光子最终以 $|+45\rangle$ 或 $|-45\rangle$ 偏振态之一输出。
* 根据一般偏振态在各基矢上的系数，可计算光子以对应偏振态输出的概率。
* 严格来说它也会给 $|-45\rangle$ 相对 $|+45\rangle$ 引入相移（源于不同光程），但此处忽略该效应，因为这是最后一个元件，我们只关心输出属于哪一偏振态，而不关心谁先到达。
* 因而该分析器在模拟中承担“读出实验结果”的角色。
* 单光子干涉在通过最后这个分析器前就已形成（更准确地说，在第二个 PA$_{HV}$ 后对应的重叠阶段可见），此后附加相移不再是核心因素。

```python
# Polarization analyzer (45)

# linear Polarizer, transmission axis +45 wrt horizontal
胃 = pi / 4
Pp45 = Qobj([[cos(胃) ** 2, cos(胃) * sin(胃)], [cos(胃) * sin(胃), sin(胃) ** 2]])
# linear Polarizer, transmission axis -45 wrt horizontal
胃 = -pi / 4
Pn45 = Qobj([[cos(胃) ** 2, cos(胃) * sin(胃)], [cos(胃) * sin(胃), sin(胃) ** 2]])


def PA_45(vector):
    p45_comp = Pp45 * vector  # retrieve only +45 component
    n45_comp = Pn45 * vector  # retrieve only -45 component
    return p45_comp, n45_comp
```

### 3. 单光子干涉实验模拟 <a class="anchor" id="section3"></a>


**定义初始变量：**
* 先设定 PA$_{HV2}$ 相移扫描的最小值、最大值和步数。模拟会遍历这些相移并重复实验，以得到干涉与相对相移之间的关系。步数越高，结果分辨率越高。
* 还需设置每个相移值对应的迭代次数。由于输出偏振结果由概率决定，必须进行足够多重复实验，才能得到可靠统计。

```python
psi_0 = p45  # define the initial state (+45 vector)

phaseshift2_init = pi / 4  # initial value
phaseshift2_max = 8 * pi
n = 100  # resolution of 蠁 (amount of steps)
step = (
    phaseshift2_max - phaseshift2_init
) / n  # interval divided by number of small steps we want
N_init = 1000  # number of iterations (range(N) -> 0 to N-1, both included)

# create x and y coords arrays to store the values needed to plot output graph
x_coords = []  # relative phase shift
y1_coords = []  # amount of photons in +45
y2_coords = []  # amount of photons in -45
```

**编写 *for 循环* 遍历 PA$_{HV2}$ 的所有相移取值：**
<br> 在该循环中，光子通过干涉仪的过程由一个等效矩阵模拟。之后利用 PA$_{45}$ 计算对应偏振输出概率，进而得到实验统计结果。
* 首先定义输出计数变量，用于记录某一相移下多次测量中输出为 $|+45\rangle$ 与 $|-45\rangle$ 的光子数。
* 再设定当前相移，并将其加入横坐标数组。
* 依据该相移构造 PA$_{HV2}$，从而得到把所有元件按正确次序合并后的总等效矩阵。
* 用总矩阵作用于初态，得到送入 PA$_{45}$ 进行判定的末态。
* 提取 $|+45\rangle$ 与 $|-45\rangle$ 分量，先求其向量范数再平方，得到对应概率。

外层 *for 循环* 内还有一个内层循环，用于模拟最终测量。光子落在 $|+45\rangle$ 还是 $|-45\rangle$ 是由概率控制的随机过程。
* 为此，生成 1 到 100 的随机数（对应 $100\%$ 概率空间）。
* 若随机数不大于 $|+45\rangle$ 分量的百分比概率，则记为输出 $|+45\rangle$；否则记为输出 $|-45\rangle$。
* 每次测量结果都累加到前述输出计数变量中。
* 完成该相移下全部迭代后，把计数加入两条纵坐标数组（分别对应 $|+45\rangle$ 与 $|-45\rangle$ 输出）。
* 随后在下一轮外循环开始时重置计数，并对新相移重复流程。

```python
for i in range(n + 1):
    output_p45 = 0
    output_n45 = 0

    phaseshift2 = phaseshift2_init + i * step
    x_coords.append(
        (phaseshift2 - phaseshift1) / pi
    )  # add realtive phase shift to x coords
    # create corresponding PA_HV2
    PA_HV2 = Qobj([[exp(1j * phaseshift2), 0], [0, 1]])
    EffM = PA_HV2 * halfwave * PA_HV1  # define the effective matrix

    # apply the effective matrix to the initial state to get the final state
    psi_final = EffM * psi_0

    psi_p45 = PA_45(psi_final)[0]  # retrieve +45 and -45 components
    psi_n45 = PA_45(psi_final)[1]

    # probab is rounded up to 5 decimals to avoid machine precision artifacts
    proba_p45 = round(psi_p45.norm() ** 2, 5)
    proba_n45 = round(psi_n45.norm() ** 2, 5)

    for j in range(N_init):
        """
        generates random number between 1 and 100 (both included),
        100 because 100% of the photons need to come out in either
        +45 or -45 state
        """
        a = random.randint(1, 100)
        if a <= proba_p45 * 100:
            output_p45 = output_p45 + 1
        else:
            output_n45 = output_n45 + 1

    y1_coords.append(output_p45)
    y2_coords.append(output_n45)
```

**绘制输出图：**
<br> 模拟过程中会持续生成用于作图的坐标数组。基于这些数据即可可视化干涉强度对相对相移（或等价地对两臂光程差）的依赖关系。

```python
plt.plot(x_coords, y1_coords, "b.", markersize=9, label="Photons in state +45")
plt.plot(x_coords, y2_coords, "r.", markersize=9, label="Photons in state -45")
legend = plt.legend(loc="upper center", fontsize="x-large")
plt.ylim([-100, N_init + 500])
plt.xlabel("Relative phase shift (multiples of 蟺)")
plt.ylabel("Amount of photons detected")
plt.title("Amount of photons existing in |+/-45> states");
```

结果显示出清晰的干涉图样：随着相对相移增大，测得光子在 $|+45\rangle$ 与 $|-45\rangle$ 两种输出之间振荡切换。


### 4. 单光子干涉实验的变体 <a class="anchor" id="section4"></a>


为了更深入理解干涉条纹为何出现以及实验的量子本质，我们可以考察一些变体情形。

本节展示：当叠加态塌缩时（这里指光子“不再同时走两条路径”），干涉条纹会消失。实现方式是阻断第一个偏振分析器的竖直或水平输出端，使光子只能走单一路径。相较前面的完整干涉模拟，需要做如下修改：
* 主要改动在第一个偏振分析器：根据阻断端口不同，把对应偏振分量在后续计算中置零。
* 另一个关键差异是最终通过整个干涉仪的光子总数。初始 $|+45\rangle$ 光子经 PA$_{HV1}$ 后有 50% 概率成为 $|V\rangle$、50% 概率成为 $|H\rangle$。若实验次数足够多，可视为两端口各占一半；阻断其中一路后，只有一半光子能继续传播。因此，原先 1000 次有效输出在该设置下变为 500 次，这要求把每个相移值的迭代次数减半。对应地，输出为 $|+45\rangle$ 与 $|-45\rangle$ 的概率和不再是 1（100%），而是 0.5；缺失的 0.5 对应在第一个分析器处被阻断的概率。
* 除上述修改外，其余代码保持不变。


#### 第一个 PA$_{HV}$ 的 V 输出端被阻断

```python
# Polarization analyzer (HV) n掳1, with V output port BLOCKED

phaseshift1 = pi / 4  # CONSTANT
# should depend on real size of the setup (here: arbitrarily chosen)
PA_HV1vb = Qobj([[exp(1j * phaseshift1), 0], [0, 0]])
```

```python
psi_0 = p45  # Defining the initial state (+45 vector)

phaseshift2_init = pi / 4  # initial value
phaseshift2_max = 8 * pi
n = 100  # resolution of 蠁 (amount of steps)
step = (
    phaseshift2_max - phaseshift2_init
) / n  # interval divided by number of small steps we want

# number of iterations (range(N) -> 0 to N-1, both included)
N_init = 1000

x_coords = []  # create x- and y- coords. arrays
# (x = phase shift of 2nd PA_HV,
# y1 = amount of photons in +45,
# y2 = amount of photons in -45)
y1_coords = []
y2_coords = []

for i in range(n + 1):
    output_p45 = 0
    output_n45 = 0

    phaseshift2 = phaseshift2_init + i * step
    # add realtive phase shift to x coords
    x_coords.append((phaseshift2 - phaseshift1) / pi)
    # create corresponding PA_HV2
    PA_HV2 = Qobj([[exp(1j * phaseshift2), 0], [0, 1]])
    EffM = PA_HV2 * halfwave * PA_HV1vb  # Defining the effective matrix

    """
    Applying the effective matrix to the initial state to get the final state
    """
    psi_final = EffM * psi_0
    psi_p45 = PA_45(psi_final)[0]  # Determining the probabilities
    psi_n45 = PA_45(psi_final)[1]
    proba_p45 = round(psi_p45.norm() ** 2, 5)
    proba_n45 = round(psi_n45.norm() ** 2, 5)

    if (proba_p45 + proba_n45) == 1:
        N = N_init  # all of the photons get to the end
    else:
        """
        half of the photons are blocked (should only get N/2 in the ouput)
        -> total prob should be 0.5
        """
        N = int(N_init / 2)

    for j in range(N):
        """
        generates random number between 1 and 50 (both included),
        50 because 50% of the photons need to come out in either
        +45 or -45 state (since other 50% was blocked)
        """
        a = random.randint(1, 50)
        if a <= proba_p45 * 100:
            output_p45 = output_p45 + 1
        else:
            output_n45 = output_n45 + 1

    y1_coords.append(output_p45)
    y2_coords.append(output_n45)

plt.plot(x_coords, y1_coords, "b.", markersize=9, label="Photons in state +45")
plt.plot(x_coords, y2_coords, "r.", markersize=9, label="Photons in state -45")
legend = plt.legend(loc="upper center", fontsize="x-large")
plt.ylim([0, 1000])
plt.xlabel("Relative phase shift (multiples of 蟺)")
plt.ylabel("Amount of photons detected")
plt.title("Amount of photons existing in |+/-45> states (V port blocked)");
```

#### 第一个 PA$_{HV}$ 的 H 输出端被阻断

```python
# Polarization analyzer (HV) n掳1, with H output port BLOCKED

PA_HV1hb = Qobj([[0, 0], [0, 1]])
```

```python
psi_0 = p45  # Defining the initial state (+45 vector)

phaseshift2_init = pi / 4  # initial value
phaseshift2_max = 8 * pi
n = 100  # resolution of 蠁 (amount of steps)
step = (
    phaseshift2_max - phaseshift2_init
) / n  # interval divided by number of small steps we want

# number of iterations (range(N) -> 0 to N-1, both included)
N_init = 1000

"""
create x- and y- coords. arrays (x = phase shift of 2nd PA_HV,
y1 = amount of photons in +45, y2 = amount of photons in -45)
"""
x_coords = []
y1_coords = []
y2_coords = []

for i in range(n + 1):
    output_p45 = 0
    output_n45 = 0

    phaseshift2 = phaseshift2_init + i * step
    # add realtive phase shift to x coords
    x_coords.append((phaseshift2 - phaseshift1) / pi)
    # create corresponding PA_HV2
    PA_HV2 = Qobj([[exp(1j * phaseshift2), 0], [0, 1]])
    # Defining the effective matrix
    EffM = PA_HV2 * halfwave * PA_HV1vb

    """
     Applying the effective matrix to the initial state to get the final state
    """
    psi_final = EffM * psi_0
    psi_p45 = PA_45(psi_final)[0]  # Determining the probabilities
    psi_n45 = PA_45(psi_final)[1]
    proba_p45 = round(psi_p45.norm() ** 2, 5)
    proba_n45 = round(psi_n45.norm() ** 2, 5)

    if (proba_p45 + proba_n45) == 1:
        N = N_init  # all of the photons get to the end
    else:
        """
        half of the photons are blocked (should only get N/2 in the ouput)
        -> total probability should be 0.5
        """
        N = int(N_init / 2)
    """
    # generates random number between 1 and 50 (both included),
    50 because 50% of the photons need to come out in either
    +45 or -45 state (since other 50% was blocked)
    """
    for j in range(N):
        a = random.randint(1, 50)
        if a <= proba_p45 * 100:
            output_p45 = output_p45 + 1
        else:
            output_n45 = output_n45 + 1

    y1_coords.append(output_p45)
    y2_coords.append(output_n45)

plt.plot(x_coords, y1_coords, "b.", markersize=9, label="Photons in state +45")
plt.plot(x_coords, y2_coords, "r.", markersize=9, label="Photons in state -45")
legend = plt.legend(loc="upper center", fontsize="x-large")
plt.ylim([0, 1000])
plt.xlabel("Relative phase shift (multiples of 蟺)")
plt.ylabel("Amount of photons detected")
plt.title("Amount of photons existing in |+/-45> states (H port blocked)");
```

### 结论与要点 <a class="anchor" id="section5"></a>


结果表明：当两条路径都畅通时，会出现明显干涉图样。随着相对相位变化，探测到的光子在 $|+45\rangle$ 与 $|-45\rangle$ 之间振荡。需要注意的是，在干涉仪内部两束光处于正交偏振，因此本身并不直接显示干涉；可见干涉是在末端 PA$_{45}$ 处显现的，因为此处把水平/垂直偏振投影到 $\pm45^\circ$ 轴上。干涉强度随两路径相位差振荡变化。

当阻断第一个偏振分析器的任一输出端后，在相对相移变化过程中，探测到的光子数基本保持常量（仅有随机涨落），不再出现干涉条纹。阻断 V 路和阻断 H 路的结果一致。这说明一旦叠加态塌缩，干涉就不会出现。


**关键结论：**
* 该实验展示了光子可同时呈现粒子性与波动性，波粒二象性不是二选一，而是统一存在。
* *那么在与“另一半”重合前，光子到底走了哪条路？* 前文已说明光子并不会真的分裂或再合并，而是始终处于单一演化态；但干涉图样取决于两臂光程差（即两条路径共同决定）。因此，若不引入光子波函数的“同时走两条路径”叠加描述，就无法解释观测到的干涉。该叠加态会持续到测量或外部退相干导致其塌缩。
* 通过阻断一路，光子无法处于路径叠加态，因而不能与自身干涉；同时输出光子数减半，这与原先 50/50 分路概率一致。即便如此，仍可在两种偏振态上看到 50/50 分布，说明量子态可在一个叠加被破坏后仍保留另一个叠加。
* 该模拟表明，单粒子量子干涉可用逐事件（event-by-event）方式描述，不必依赖更复杂的波场或完整含时系统框架即可展示核心机制。其数学本质可以由简单向量与矩阵运算给出。但若要进行真实物理预测，仍需纳入相干长度等实验限制，否则模拟结论可能不具物理意义。

```python
qutip.about()
```
