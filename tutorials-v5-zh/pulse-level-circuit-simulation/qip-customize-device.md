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

# 自定义脉冲级仿真
Author: Boxi Li (etamin1201@gmail.com)

本笔记演示如何在 qutip-qip 中自定义脉冲级模拟器，分为三部分：
1. 自定义哈密顿量模型
2. 自定义编译器
3. 自定义噪声

```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, fidelity, sigmax, sigmay, sigmaz, tensor, about
from qutip_qip.circuit import QubitCircuit
from qutip_qip.compiler import GateCompiler, Instruction, SpinChainCompiler
from qutip_qip.device import Model, ModelProcessor
from qutip_qip.noise import Noise
```

## 自定义哈密顿量模型


先从哈密顿量模型自定义开始。模型由 `Model` 类实例表示。
处理器模型的最小要求是：
- 指定硬件参数（编译器将用其计算脉冲强度）；
- 指定物理系统的控制哈密顿量（`Processor` 将通过 `get_control` 访问）。

```python
class MyModel(Model):
    def __init__(
        self, num_qubits, dims=None, h_x=1.0, h_z=1.0, g=0.1, t1=None, t2=None
    ):
        super().__init__(num_qubits, dims=dims)
        self.params = {
            "sz": [h_z] * num_qubits,
            "sx": [h_x] * num_qubits,
            "g": [g] * num_qubits,
            #  Will be accessed by the noise module.
            "t1": t1,
            "t2": t2,
        }
        # Control Hamiltonians
        _two_qubit_operator = tensor([sigmax(), sigmax()]) + tensor(
            [sigmay(), sigmay()]
        )
        self.controls = {}
        self.controls.update(
            {f"sx{n}": (2 * np.pi * sigmax(), n) for n in range(num_qubits)}
        )
        self.controls.update(
            {f"sz{n}": (2 * np.pi * sigmaz(), n) for n in range(num_qubits)}
        ),
        self.controls.update(
            {
                f"g{n}": (2 * np.pi * _two_qubit_operator, [n, n + 1])
                for n in range(num_qubits - 1)
            }
        ),

    def get_control(self, label):
        """
        The mandatory method. It returns a pair of Qobj and int representing
        the control Hamiltonian and the target qubit.
        """
        return self.controls[label]

    def get_control_labels(self):
        """
        It returns all the labels of available controls.
        """
        return self.controls.keys()

    def get_control_latex(self):
        """
        The function returns a list of dictionaries, corresponding to the latex
        representation of each control. This is used in the plotting.
        Controls in each dictionary will be plotted in a different colour.
        See examples later in this notebook.
        """
        return [
            {f"sx{n}": r"$\sigma_x^%d$" % n for n in range(num_qubits)},
            {f"sy{n}": r"$\sigma_z^%d$" % n for n in range(num_qubits)},
            {f"g{n}": r"$g_{%d}$" % (n) for n in range(num_qubits - 1)},
        ]
```

这是一个由 $n$ 个量子比特排成链的一维系统（与 [spin chain model](https://qutip-qip.readthedocs.io/en/stable/apidoc/qutip_qip.device.html?highlight=spinchain#qutip_qip.device.SpinChainModel) 相同），可控制三类哈密顿量：每个比特上的 $\sigma_x$、$\sigma_z$，以及相邻比特间 $\sigma_x\sigma_x+\sigma_y\sigma_y$ 作用：

$$
H = \sum_{j=0}^{n-1} c_{1,j}(t) \cdot h_x^{j}\sigma_x^{j} + \sum_{j=0}^{n-1} c_{2,j}(t) \cdot h_z^{j}\sigma_z^{j}
+ \sum_{j=0}^{n-2} c_{3,j}(t)\cdot g^{j}(\sigma_x^{j}\sigma_x^{j+1}+\sigma_y^{j}\sigma_y^{j+1})
$$

其中 $h_x,h_z,g$ 是硬件参数，$c_{i,j}(t)$ 是含时控制脉冲系数。
该哈密顿量与 QuTiP 线性自旋链模型一致。
一般硬件参数可随比特变化；这里为简化，仅用三个统一数值 $h_x,h_z,g$。

要模拟自定义量子设备，可把模型传入 `ModelProcessor`。
`ModelProcessor` 面向具体物理模型（区别于任意哈密顿量的最优控制），并提供仿真所需通用方法。 

```python
num_qubits = 2
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0, h_z=1.0, g=0.1))
```

在 `set_up_ops` 中定义控制哈密顿量并初始化控制脉冲。
可通过以下方式获得“脉冲标签 -> 脉冲位置”的映射：

```python
processor.get_control_labels()
```

也可按标签访问某个控制哈密顿量：

```python
sx0 = processor.get_control("sx0")
sx0
```

在 qutip-qip 0.1 中模型直接定义在 `Processor` 内。
0.2 中仍可按下方写法实现，效果与上面等价：会自动创建并保存一个 `Model` 实例。

```python
class MyProcessor(ModelProcessor):
    """
    Custom processor built using ModelProcessor as the base class.
    This custom processor will inherit all the methods of the base class
    such as setting up of the T1 and T2 decoherence rates in the simulations.

    In addition, it is possible to write your own functions to add control
    pulses.

    Args:
        num_qubits (int): Number of qubits in the processor.
        t1, t2 (float or list): The T1 and T2 decoherence rates for the
    """

    def __init__(self, num_qubits, h_x, h_z, g, t1=None, t2=None):
        super(MyProcessor, self).__init__(
            num_qubits, t1=t1, t2=t2
        )  # call the parent class initializer
        # The control pulse is discrete or continuous.
        self.pulse_mode = "discrete"
        self.model.params.update(
            {
                # can also be different for each qubit
                "sz": [h_z] * num_qubits,
                "sx": [h_x] * num_qubits,
                "g": [g] * num_qubits,
            }
        )
        # The dimension of each controllable quantum system
        self.model.dims = [2] * num_qubits
        self.num_qubits = num_qubits
        self.set_up_ops()  # set up the available Hamiltonians

    def set_up_ops(self):
        """
        Sets up the control operators.
        """
        for m in range(self.num_qubits):
            # sigmax pulse on m-th qubit with the corresponding pulse
            self.add_control(2 * np.pi * sigmax(), m, label="sx" + str(m))
        # sz
        for m in range(self.num_qubits):
            self.add_control(2 * np.pi * sigmaz(), m, label="sz" + str(m))
        # interaction operator
        operator = tensor([sigmax(), sigmax()]) + tensor([sigmay(), sigmay()])
        for m in range(self.num_qubits - 1):
            self.add_control(2 * np.pi * operator, [m, m + 1],
                             label="g" + str(m))
```

### 加载并编译量子线路



先定义量子线路。这里使用双比特线路包含两个 X 门，作用于 $|00\rangle$ 后得到 $|11\rangle$。

```python
circuit = QubitCircuit(num_qubits)
circuit.add_gate("X", targets=1)
circuit.add_gate("X", targets=0)
circuit
```

线路绘图参考[该笔记](../quantum-circuits/quantum-gates.md)。


要把量子线路转换为哈密顿量模型，需要编译器。下一节会详细讲自定义编译器。
由于此处采用自旋链模型，我们直接“借用”其编译器。

```python
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0, h_z=1.0, g=0.1))
processor.native_gates = ["ISWAP", "RX", "RZ"]

# processor.num_qubits, processor.params
# access directly the information in the model.
compiler = SpinChainCompiler(processor.num_qubits, processor.params)

processor.load_circuit(circuit, compiler=compiler)
result = processor.run_state(init_state=basis([2, 2], [0, 0]))
result.states[-1]
```

编译后的脉冲系数会保存在处理器里，可通过以下方式读取：

```python
sx1_pulse = processor.find_pulse("sx1")
print(sx1_pulse.coeff)
print(sx1_pulse.tlist)
```

这是一个从 0 到 0.25 的矩形脉冲。

#### 注意

对离散脉冲，时间序列长度通常比系数少 1，因为要指定脉冲起止边界。
若两者长度相同，则 `coeff` 最后一个元素会被忽略。
后文会看到连续脉冲，此时 `coeff` 与 `tlist` 长度相同。

为直观展示控制脉冲，可通过定义 `get_operators_labels` 提供 LaTeX 标签，然后绘制编译结果。

```python
processor.plot_pulses()
plt.show()
```

## 自定义编译器

不同量子硬件实现同一逻辑门的方式不同。
即使同一平台，不同实现也会导致不同性能。
最简单实现是矩形脉冲（如上），但真实控制信号往往是连续包络。
下面展示如何用高斯类脉冲自定义编译器。

典型编译函数形如 `XX_compiler(self, gate, args)`，接受 `gate`（待编译门）与 `args`（额外参数字典，如 `Processor.params`）。

每个门的编译函数返回一个 `Instruction` 对象，包含输入门、时间序列与脉冲系数。

下面先给出矩形脉冲示例。

```python
def rz_compiler(gate, args):
    """
    Compiles the RZ gate to an instruction for a pulse.

    Args:
        gate (qutip_qip.circuit.Gate): A qutip Gate object.
        args:(dict): A dictionary for compilation arguments e.g.
                     hardware parameters.

    Returns:
        Instruction (qutip_qip.compiler.instruction.Instruction):
        An instruction to implement a gate containing the control
        pulses.
    """
    tlist = np.array([1.0])
    coeff = np.array([0.0, 0.25])
    # instruction is an object that includes the pulse coefficient
    # and time sequence
    pulse_info = [("sz0", coeff)]
    return [Instruction(gate, tlist, pulse_info)]
```

接着替换为连续脉冲。这里定义 `single_qubit_compiler`：
对 RX/RY 门，从参数中读取最大驱动强度并计算对应时间序列与脉冲振幅。

为简化，使用父类 `GateCompiler` 提供的 [`generate_pulse_shape`](https://qutip-qip.readthedocs.io/en/stable/apidoc/qutip_qip.compiler.html?highlight=generate_pulse_shape#qutip_qip.compiler.GateCompiler.generate_pulse_shape)。

```python
class MyCompiler(GateCompiler):  # compiler class
    def __init__(self, num_qubits, params):
        super(MyCompiler, self).__init__(num_qubits, params=params)
        # pass our compiler function as a compiler for X gate.
        self.gate_compiler["X"] = self.single_qubit_compiler
        self.gate_compiler["Y"] = self.single_qubit_compiler
        self.args.update({"params": params})

    def single_qubit_compiler(self, gate, args):
        """
        Compiler for the X and Y gate.
        """
        targets = gate.targets
        if gate.name == "Z":
            pulse_prefix = "sz"
            pulse_strength = args["params"]["sz"][targets[0]]
        elif gate.name == "X":
            pulse_prefix = "sx"
            pulse_strength = args["params"]["sx"][targets[0]]
        coeff, tlist = self.generate_pulse_shape(
            "hann",  # Scipy Hann window
            100,  # 100 sampling point
            maximum=pulse_strength,
            area=(np.pi / 2)
            / (
                2 * np.pi
            ),  # 1/2 becuase we use sigmax as the operator instead of sigmax/2
        )
        pulse_info = [(pulse_prefix + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]
```

`pulse_mode` 可设为 `"discrete"` 或 `"continuous"`。
连续脉冲下，QuTiP 求解器会用三次样条插值。
为减小边界效应，建议连续脉冲在起点和终点都平滑收敛到 0。
下面引入 T1 退相干，对比调度前后末态保真度。

```python
gauss_compiler = MyCompiler(processor.num_qubits, processor.params)
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0,
                                         h_z=1.0, g=0.1, t1=300))
processor.pulse_mode = "continuous"
tlist, coeff = processor.load_circuit(
    circuit, compiler=gauss_compiler, schedule_mode=False
)
print
result = processor.run_state(init_state=basis([2, 2], [0, 0]))
print(
    "fidelity without scheduling:", fidelity(result.states[-1],
                                             basis([2, 2], [1, 1]))
)
```

```python
processor.plot_pulses(use_control_latex=False)
plt.show()
```

也可启用调度器，以更短执行时间运行线路。

```python
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0,
                                         h_z=1.0, g=0.1, t1=300))
tlist, coeffs = processor.load_circuit(
    circuit, schedule_mode="ASAP", compiler=gauss_compiler
)
processor.pulse_mode = "continuous"
result = processor.run_state(init_state=basis([2, 2], [0, 0]))
print("fidelity with scheduling:", fidelity(result.states[-1],
                                            basis([2, 2], [1, 1])))
```

```python
processor.plot_pulses(use_control_latex=False)
plt.show()
```

### 定义自己的量子门
qutip 预定义门数量有限，因此建议按需自定义门。下面演示参数化门的定义。首先定义门函数：

```python
def mygate(theta=None):
    # We just call the Molmer Sorensen gate as an example.
    # If you do not want to run the circuit at the gate matrix level
    # (circuit.run), no need for this function,
    # otherwise you will need to define this python function
    # that returns the Qobj of the gate
    from qutip_qip.operations import molmer_sorensen

    return molmer_sorensen(theta, 2, targets=[0, 1])


circuit = QubitCircuit(2)
# no need for this if you don't use circuit.run
circuit.user_gates = {"MYGATE": mygate}
circuit.add_gate("X", targets=1)
circuit.add_gate("MYGATE", targets=[0, 1], arg_value=3 * np.pi / 2)
circuit.add_gate("X", targets=0)
circuit
# You may see a warning because MYGATE is not found in defined
# LaTeX gate names, just ignore it.
```

```python
circuit.run(basis([2, 2], [0, 0]))
```

下一步是为该门定义编译函数。真实 MS 门编译较复杂，这里仅用一个三角脉冲作示例，重点展示如何在编译时获取参数 `theta`。

```python
def mygate_compiler(gate, args):
    targets = gate.targets  # target qubit

    theta = gate.arg_value
    coeff1 = np.concatenate([np.linspace(0, 10, 50),
                             np.linspace(10, 0, 50), [0]]) / 50
    coeff2 = np.concatenate([np.linspace(0, 10, 50),
                             np.linspace(10, 0, 50), [0]]) / 50
    #  save the information in a tuple (pulse_name, coeff)
    pulse_info = [
        ("sx" + str(targets[0]), theta * coeff1),
        ("sx" + str(targets[1]), theta * coeff2),
    ]
    tlist = np.linspace(0, 1, len(coeff1))
    return [Instruction(gate, tlist, pulse_info)]
```

```python
gauss_compiler = MyCompiler(processor.num_qubits, processor.params)
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0,
                                         h_z=1.0, g=0.1, t1=300))
gauss_compiler.gate_compiler["MYGATE"] = mygate_compiler
processor.pulse_mode = "continuous"
tlist, coeff = processor.load_circuit(circuit, compiler=gauss_compiler)
processor.plot_pulses()
plt.show()
```

## 自定义噪声
除预定义噪声（如 T1、T2、控制幅度随机噪声，见[指南](https://qutip-qip.readthedocs.io/en/stable/qip-processor.html)）外，也可自定义噪声。这里给出两个例子：
- 系统性（与脉冲无关）噪声
- 脉冲相关噪声

先简述仿真框架的数据结构：
控制量以 `Pulse` 对象列表存储在 Processor 中。每个 Pulse 包含理想控制、控制噪声与退相干部分。
系统性噪声保存在标签为 `"system"` 的 `Pulse` 中，表示系统本征动力学；
脉冲相关噪声则添加到对应控制 `Pulse` 上。

噪声定义通过 `UserNoise` 子类完成，需实现两部分：
- 初始化：声明噪声属性（如频率、幅度）；
- `get_noisy_dynamics`：接收控制脉冲 `pulses`、表示系统性噪声的占位 `Pulse`，以及系统维度（此处双比特 `[2,2]`）。


```python
class Extral_decay(Noise):
    def __init__(self, arg):
        self.arg = arg
        pass

    def get_noisy_dynamics(self, dims, pulses, systematic_noise):
        pass
```

### 系统性噪声

先看系统性噪声示例：给相邻比特加入常量强度 ZZ 串扰，步骤如下：

- 定义噪声类
- 用给定耦合强度初始化噪声对象
- 像平常一样定义 Processor，并把噪声加进去

下面在“双 X 门”线路上测试加入该噪声后的保真度。

```python
circuit = QubitCircuit(2)
circuit.add_gate("X", targets=1)
circuit.add_gate("X", targets=0)
```

```python
class ZZ_crosstalk(Noise):
    def __init__(self, strength):
        self.strength = strength

    def get_noisy_dynamics(self, dims, pulses, systematic_noise):
        zz_operator = tensor([sigmaz(), sigmaz()])
        for i in range(len(dims) - 1):
            systematic_noise.add_control_noise(
                self.strength * zz_operator, targets=[i, i + 1],
                tlist=None, coeff=True
            )  # constant, always 1


crosstalk_noise = ZZ_crosstalk(strength=1.0)
```

```python
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0, h_z=1.0, g=0.1))
processor.add_noise(crosstalk_noise)  # The noise is added to the processor
gauss_compiler = MyCompiler(processor.num_qubits, processor.params)
tlist, coeff = processor.load_circuit(circuit, compiler=gauss_compiler)

result = processor.run_state(init_state=basis([2, 2], [0, 0]))
print(
    "Final fidelity with ZZ crosstalk:",
    fidelity(result.states[-1], basis([2, 2], [1, 1])),
)
```

### 脉冲相关噪声
第二个例子演示如何添加“与控制脉冲线性相关”的额外振幅阻尼通道。
具体地，当控制脉冲 `sx` 打开时，对应退相干也同时开启。
其湮灭算符系数与控制脉冲幅度成正比。
该噪声可叠加在默认 T1/T2 噪声之上。

```python
class Extral_decay_2(Noise):
    def __init__(self, ratio):
        self.ratio = ratio

    def get_noisy_dynamics(self, dims, pulses, systematic_noise):
        from qutip import destroy

        op = destroy(2)
        for pulse in pulses:  # iterate for all pulses
            if (
                "sx" in pulse.label and pulse.coeff is not None
            ):  # if it is a sigma-x pulse and is not empty
                pulse.add_lindblad_noise(
                    op,
                    targets=pulse.targets,
                    tlist=pulse.tlist,
                    coeff=self.ratio * pulse.coeff,
                )
                # One can also use add_control_noise here
                # to add additional Hamiltonian as noise (see next example).


extral_decay = Extral_decay_2(0.3)
```

```python
processor = ModelProcessor(model=MyModel(num_qubits, h_x=1.0, h_z=1.0, g=0.1))
processor.add_noise(extral_decay)
gauss_compiler = MyCompiler(processor.num_qubits, processor.params)
tlist, coeff = processor.load_circuit(circuit, compiler=gauss_compiler)

result = processor.run_state(init_state=basis([2, 2], [0, 0]))
print(
    "Final fidelity with pulse-dependent decoherence:",
    fidelity(result.states[-1], basis([2, 2], [1, 1])),
)
```

```python
about()
```
