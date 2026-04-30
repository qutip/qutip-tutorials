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

# 随机基准测试仿真

在本示例中，我们复现 [Piltz et. al.](https://www.nature.com/articles/ncomms5679?origin=ppub) 图 3a 使用的随机基准测试实验。

注意：这个示例计算量较大。完整仿真建议使用 [joblib](https://joblib.readthedocs.io/en/latest/) 并行计算；不过运行演示版不需要它。


```python
import matplotlib.pyplot as plt
import numpy as np
from qutip import (Qobj, SolverOptions, about, basis, fock_dm, qeye,
                   sigmax, sigmay, sigmaz, tensor)
from qutip_qip.circuit import QubitCircuit
from qutip_qip.compiler import GateCompiler, Instruction
from qutip_qip.device import Model, ModelProcessor
from qutip_qip.noise import Noise
from qutip_qip.operations import Gate
from scipy.optimize import curve_fit
```

我们构造一个双量子比特 `Processor`。第二个量子比特相对第一个失谐 $\delta=1.852$ MHz。
对第一个量子比特施加一串 Rabi 频率 $\Omega=20$ KHz、相位随机的 $\pi$ 脉冲。
再定义噪声，使同一脉冲也作用在第二个量子比特上。由于失谐，该脉冲不会翻转第二个比特，但会造成扩散行为，使第二个比特相对于初态的平均保真度下降。

这里我们用双量子比特 `Processor` 复现该现象。
初始态保真度为 0.975，模拟哈密顿量
\begin{align}
H=\Omega(t)(\sigma^x_0 + \lambda \sigma^x_1) + \delta\sigma^z_1
,
\end{align}
其中 $\lambda$ 是串扰脉冲振幅比。


下面先构建一个名为 `MyModel` 的哈密顿量模型。
为简化起见，仅包含两个单比特控制哈密顿量：$\sigma_x$ 与 $\sigma_y$。
然后定义两类旋转门 RX、RY 的编译流程。
此外还定义了一个带混合 X/Y 正交分量的旋转门，参数为相位 $\phi$，形式是 $\cos(\phi)\sigma_x+\sin(\phi)\sigma_y$。
这个门会在后面的自定义噪声示例中使用。

随后用该模型初始化 `ModelProcessor`。
`ModelProcessor` 内置了默认仿真流程（如 `load_circuit`）。
由于硬件原生门是绕 $x$ 与 $y$ 轴的旋转，我们在属性 `native_gates` 中声明它们。
在提供该原生门集合后，绕 $z$ 轴旋转会自动分解为绕 $x$、$y$ 轴旋转。
这里定义了一个先做 $\pi/2$ 旋转再做 Z 门的线路。
编译后的脉冲可见图中 Z 门已被分解为绕 $x$、$y$ 轴的旋转。

```python
class MyModel(Model):
    """A custom Hamiltonian model with sigmax and sigmay control."""

    def get_control(self, label):
        """
        Get an available control Hamiltonian.
        For instance, sigmax control on the zeroth
        qubits is labeled "sx0".

        Args:
            label (str): The label of the Hamiltonian

        Returns:
            The Hamiltonian and target qubits as a tuple
            (qutip.Qobj, list).
        """
        targets = int(label[2:])
        if label[:2] == "sx":
            return 2 * np.pi * sigmax() / 2, [targets]
        elif label[:2] == "sy":
            return 2 * np.pi * sigmay() / 2, [targets]
        else:
            raise NotImplementedError("Unknown control.")


class MyCompiler(GateCompiler):
    """
    Custom compiler for generating pulses from gates using
    the base class GateCompiler.

    Args:
        num_qubits (int): The number of qubits in the processor
        params (dict): A dictionary of parameters for gate pulses
                       such as the pulse amplitude.
    """

    def __init__(self, num_qubits, params):
        super().__init__(num_qubits, params=params)
        self.params = params
        self.gate_compiler = {
            "ROT": self.rotation_with_phase_compiler,
            "RX": self.single_qubit_gate_compiler,
            "RY": self.single_qubit_gate_compiler,
        }

    def generate_pulse(self, gate, tlist, coeff, phase=0.0):
        """Generates the pulses.

        Args:
            gate (qutip_qip.circuit.Gate): A qutip Gate object.
            tlist (array): A list of times for the evolution.
            coeff (array): An array of coefficients for the gate pulses
            phase (float): The value of the phase for the gate.

        Returns:
            Instruction (qutip_qip.compiler.instruction.Instruction):
            An instruction to implement a gate containing the control pulses.
        """
        pulse_info = [
            # (control label, coeff)
            ("sx" + str(gate.targets[0]), np.cos(phase) * coeff),
            ("sy" + str(gate.targets[0]), np.sin(phase) * coeff),
        ]
        return [Instruction(gate, tlist=tlist, pulse_info=pulse_info)]

    def single_qubit_gate_compiler(self, gate, args):
        """Compiles single-qubit gates to pulses.

        Args:
            gate (qutip_qip.circuit.Gate): A qutip Gate object.

        Returns:
            Instruction (qutip_qip.compiler.instruction.Instruction):
            An instruction to implement a gate containing the control pulses.
        """
        # gate.arg_value is the rotation angle
        tlist = np.abs(gate.arg_value) / self.params["pulse_amplitude"]
        coeff = self.params["pulse_amplitude"] * np.sign(gate.arg_value)
        if gate.name == "RX":
            return self.generate_pulse(gate, tlist, coeff, phase=0.0)
        elif gate.name == "RY":
            return self.generate_pulse(gate, tlist, coeff, phase=np.pi / 2)

    def rotation_with_phase_compiler(self, gate, args):
        """Compiles gates with a phase term.

        Args:
            gate (qutip_qip.circuit.Gate): A qutip Gate object.

        Returns:
            Instruction (qutip_qip.compiler.instruction.Instruction):
            An instruction to implement a gate containing the control pulses.
        """
        # gate.arg_value is the pulse phase
        tlist = self.params["duration"]
        coeff = self.params["pulse_amplitude"]
        return self.generate_pulse(gate, tlist, coeff, phase=gate.arg_value)


# Define a circuit and run the simulation
num_qubits = 1

circuit = QubitCircuit(1)
circuit.add_gate("RX", targets=0, arg_value=np.pi / 2)
circuit.add_gate("Z", targets=0)

myprocessor = ModelProcessor(model=MyModel(num_qubits))
myprocessor.native_gates = ["RX", "RY"]

mycompiler = MyCompiler(num_qubits, {"pulse_amplitude": 0.02})

myprocessor.load_circuit(circuit, compiler=mycompiler)
result = myprocessor.run_state(basis(2, 0))

fig, ax = myprocessor.plot_pulses(figsize=(5, 3), dpi=120,
                                  use_control_latex=False)
ax[-1].set_xlabel("$t$")
fig.tight_layout()
```

接下来定义自定义噪声对象 `ClassicalCrossTalk`，其基类为 `Noise`。
`get_noisy_dynamics` 会在仿真中被调用，以生成含噪哈密顿量模型。
这里的噪声模型是：把同样的驱动哈密顿量加到相邻量子比特上，强度与原控制脉冲强度成比例。
量子比特跃迁频率失谐通过给处理器加入 $\sigma_z$ 漂移哈密顿量实现，频率设为 1.852 MHz。




```python
class ClassicalCrossTalk(Noise):
    def __init__(self, ratio):
        self.ratio = ratio

    def get_noisy_dynamics(self, dims=None, pulses=None,
                           systematic_noise=None):
        """Adds noise to the control pulses.

        Args:
            dims: Dimension of the system, e.g., [2,2,2,...] for qubits.
            pulses: A list of Pulse objects, representing the compiled pulses.
            systematic_noise: A Pulse object with no ideal control,
            used to represent pulse-independent noise such as decoherence
            (not used in this example).
        Returns:
            pulses: The list of modified pulses according to the noise model.
            systematic_noise: A Pulse object (not used in this example).
        """
        for i, pulse in enumerate(pulses):
            if "sx" not in pulse.label and "sy" not in pulse.label:
                continue  # filter out other pulses, e.g. drift
            target = pulse.targets[0]
            if target != 0:  # add pulse to the left neighbour
                pulses[i].add_control_noise(
                    self.ratio * pulse.qobj,
                    targets=[target - 1],
                    coeff=pulse.coeff,
                    tlist=pulse.tlist,
                )
            if target != len(dims) - 1:  # add pulse to the right neighbour
                pulses[i].add_control_noise(
                    self.ratio * pulse.qobj,
                    targets=[target + 1],
                    coeff=pulse.coeff,
                    tlist=pulse.tlist,
                )
        return pulses, systematic_noise
```

最后定义一个随机线路，由一串相位随机的 $\pi$ 旋转脉冲组成。
驱动脉冲是持续时间 $25\,\mu\rm{s}$、Rabi 频率 20 KHz 的 $\pi$ 脉冲。
该随机基准协议可用于研究相邻量子比特上由经典串扰引起的退相干。

两个量子比特初始在 $|00\rangle$，保真度为 0.975。
线路结束后测量第二个量子比特布居。
若没有串扰，它应保持在基态；而串扰会引起扩散行为并导致保真度下降。

完整仿真需重复 1600 次以得到平均保真度，可能耗时数小时。
因此下面代码只取 $t=250$ 的两个样本。完整版本写在注释中。

```python
def single_crosstalk_simulation(num_gates):
    """
    A single simulation, with num_gates representing the number of rotations.

    Args:
        num_gates (int): The number of random gates to add in the simulation.

    Returns:
        result (qutip.solver.Result):
            A qutip Result object obtained from any of the
            solver methods such as mesolve.
    """
    # Qubit-0 is the target qubit. Qubit-1 suffers from crosstalk.
    num_qubits = 2
    myprocessor = ModelProcessor(model=MyModel(num_qubits))
    # Add qubit frequency detuning 1.852MHz for the second qubit.
    myprocessor.add_drift(2 * np.pi * (sigmaz() + 1) / 2 * 1.852, targets=1)
    myprocessor.native_gates = None  # Remove the native gates
    mycompiler = MyCompiler(num_qubits,
                            {"pulse_amplitude": 0.02, "duration": 25})
    myprocessor.add_noise(ClassicalCrossTalk(1.0))
    # Define a randome circuit.
    gates_set = [
        Gate("ROT", 0, arg_value=0),
        Gate("ROT", 0, arg_value=np.pi / 2),
        Gate("ROT", 0, arg_value=np.pi),
        Gate("ROT", 0, arg_value=np.pi / 2 * 3),
    ]
    circuit = QubitCircuit(num_qubits)
    for ind in np.random.randint(0, 4, num_gates):
        circuit.add_gate(gates_set[ind])
    # Simulate the circuit.
    myprocessor.load_circuit(circuit, compiler=mycompiler)
    init_state = tensor(
        [Qobj([[init_fid, 0], [0, 0.025]]),
         Qobj([[init_fid, 0], [0, 0.025]])]
    )
    # increase the maximal allowed steps
    options = SolverOptions(nsteps=10000)
    e_ops = [tensor([qeye(2), fock_dm(2)])]  # observable

    # compute results of the run using a solver of choice
    result = myprocessor.run_state(
        init_state, solver="mesolve", options=options, e_ops=e_ops
    )
    # measured expectation value at the end
    result = result.expect[0][-1]
    return result


# The full simulation may take several hours
# so we just choose num_sample=2 and num_gates=250 as a test
num_sample = 2
fidelity = []
fidelity_error = []
init_fid = 0.975
num_gates_list = [250]

# The full simulation is defined in the commented lines below.

# from joblib import Parallel, delayed  # for parallel simulations
# num_sample = 1600
# num_gates_list = [250, 500, 750, 1000, 1250, 1500]

for num_gates in num_gates_list:
    expect = [single_crosstalk_simulation(num_gates)
              for i in range(num_sample)]
    fidelity.append(np.mean(expect))
    fidelity_error.append(np.std(expect) / np.sqrt(num_sample))
```

下面画出一组已记录结果作为示意。

```python
# Recorded result of a full simulation
num_gates_list = [250, 500, 750, 1000, 1250, 1500]
fidelity = [
    0.9566768747558925,
    0.9388905075892828,
    0.9229470389282218,
    0.9075513000339529,
    0.8941659320508855,
    0.8756519016627652,
]

fidelity_error = [
    0.00042992029265330223,
    0.0008339882813741004,
    0.0012606632769758602,
    0.0014643550337816722,
    0.0017695604671714809,
    0.0020964978542167617,
]


def rb_curve(x, a):
    return (1 / 2 + np.exp(-2 * a * x) / 2) * 0.975


pos, cov = curve_fit(rb_curve, num_gates_list, fidelity, p0=[0.001])

xline = np.linspace(0, 1700, 200)
yline = rb_curve(xline, *pos)

fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
ax.errorbar(
    num_gates_list, fidelity, yerr=fidelity_error, fmt=".",
    capsize=2, color="slategrey"
)
ax.plot(xline, yline, color="slategrey")
ax.set_ylabel("Average fidelity")
ax.set_xlabel(r"Number of $\pi$ rotations")
ax.set_xlim((0, 1700));
```

```python
about()
```
