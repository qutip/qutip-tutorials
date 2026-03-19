---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# <code>QobjEvo</code>：含时量子对象
Made by Eric Giguere, updated by Jake Lishman

```python
# Basic setup
import numpy as np
import qutip

size = 4
t = 1.0
a = qutip.destroy(size)
ad = qutip.create(size)
n = qutip.num(size)
Id = qutip.qeye(size)
```

## 目录

- [动机](#Motivation)
- [实例化](#Instantiation)
    * [函数形式时间依赖](#Function-time-dependence)
    * [字符串形式时间依赖](#String-time-dependence)
    * [数组形式时间依赖](#Array-dependence)
- [求值](#Evaluation)
- [编译](#Compilation)
- [参数](#Arguments)
    * [函数形式](#Function-form)
    * [字符串形式](#String-form)
    * [使用对象](#Using-objects)
    * [动态参数](#Dynamic-arguments)
- [数学运算](#Mathematics)
- [超算符](#Superoperators)


## 动机
`Qobj` 是通用量子对象，但它只能表示常量对象。需要表示含时对象时，相关函数通常接收 `(Qobj, <time-dependence>)` 元组列表，其中时间依赖可以是函数、字符串或数组。函数内部会把这些内容转换为新的含时类 `QobjEvo`。之后 QuTiP 会在该表示上做编译和优化，以保证求解器运行更快。

你并不“必须”手动使用 `QobjEvo`。但如果你要在长时间积分中重复使用大型算符，可以自行实例化并手动调用 C 编译方法。这会带来几秒的一次性启动开销，但后续每次使用都会更快。如果一次积分会持续约 15 分钟以上，这个权衡通常是值得的，即使对象不复用也常常有收益。


## 实例化

`QobjEvo` 可直接用传给 `mesolve` 的那种含时列表来实例化。它甚至不一定真的含时，不过常量对象相比变系数对象在这里收益较小。支持的三类标量时间依赖是：
  - 函数
  - 字符串
  - 数组

它可表示如下形式的对象
$$
A(t) = \sum_k f_k(t) A_k
$$
其中 $f_k(t)$ 是含时标量，$A_k$ 是常量 `Qobj`。对应列表形式为
```
[A0, [A1, f1], [A2, f2], ...]
```
其中 `Ak` 是常量 `Qobj`，`fk` 是可选形式之一的时间依赖。

也可以通过 `Qobj` 与封装后时间系数相乘来构造 `QobjEvo`：
```
A0 + A1 * qutip.coefficient(f1, ...) + ...
```

```python
constant_form = qutip.QobjEvo([n])
```

<!-- #region -->
### 函数形式时间依赖

这应是一个合法 Python 函数，签名为
```python
(t: float, ...) -> complex
```
其中 `t` 是时间。还可以加入额外参数，并在不重建 `QobjEvo` 的情况下修改。返回值是 $f_k$ 的复数值。后文会进一步介绍额外参数。
<!-- #endregion -->

```python
def cos_t(t):
    return np.cos(t)


function_form = n + (a + ad) * qutip.coefficient(cos_t)
```

如果你需要更复杂形式（例如带内部状态记忆，或参数固定后不再变化的函数族），可使用实现了 `__call__` 的类。

```python
class callable_time_dependence:
    def __init__(self, add):
        self.add = add

    def __call__(self, t, args):
        return self.add + np.cos(t)


callable_form = qutip.QobjEvo([n, [a + ad, callable_time_dependence(2)]])
```

### 字符串形式时间依赖

这应是一个合法、可求值为 `complex` 的单个 Python 表达式。粗略来说，只要 `eval(x)` 能得到合法复数，通常都可以。除 `t`（时间）外，作用域内还预定义了以下符号：
```
sin  cos  tan   asin  acos  atan
sinh cosh tanh  asinh acosh atanh  
exp  log  log10 erf   zerf  sqrt  
real imag conj  abs   norm  arg
proj pi
```
此外，`np` 指向 `numpy`，`spe` 指向 `scipy.special`。

```python
string_form = qutip.QobjEvo([n, [a + ad, "cos(t)"]])
```

### 数组形式时间依赖

如果时间系数计算代价较高，可以预先在不同时间点计算函数值并以数组形式传入；再通过 `QobjEvo` 构造函数的 `tlist` 关键字参数传入对应时间列表。中间时刻将使用三次样条插值。

`tlist` 必须有序，但不必等间隔。若多项都用数组形式，它们必须共用同一个 `tlist`（构造器只接收一个）。

```python
tlist = np.linspace(0, 10, 101)
values = np.cos(tlist)

array_form = n + (a + ad) * qutip.coefficient(values, tlist=tlist)
```

## 求值

无论使用哪种时间依赖形式，甚至是否为常量对象，都可以像函数一样调用 `QobjEvo` 实例获取某时刻的值。返回值是 `Qobj`。

```python
constant_form(2)
```

```python
function_form(2)
```

```python
callable_form(2)
```

```python
string_form(2)
```

```python
array_form(2)
```


```python
string_form(4)
```

## 参数

可通过 `args` 字典向系数函数或字符串表达式传递数据。对于函数形式，`args` 作为显式参数传入；对于字符串形式，`args` 就像额外变量定义。换言之，如果 `args={'x': 1}`，则字符串 `'x + 2'` 会返回有效值。  
`args` 的键应始终是合法 Python 标识符字符串，且不应以下划线（`_`）开头。

值可以是任意类型，但若字符串依赖启用了 Cython 编译，除了合法 C 数值类型（含 `complex`）、`numpy` 数组，以及 Cython 能原生调用的对象外，其他类型通常会有明显性能损失。

可以在实例化时直接传入 `args`；这些值会用于每次调用，除非你在调用时用 `args` 临时覆盖。覆盖只在该次调用有效，后续调用仍使用初始化时的值。


### 函数形式

```python
def coeff_with_args(t, args):
    return t + args["delta"]


td_args = Id * qutip.coefficient(coeff_with_args, args={"delta": 1.0})
td_args(2)
```

```python
# Temporarily overriding the arguments.
td_args(2, delta=10)
```

```python
# A subsequent regular call will still use the args given at initialisation.
td_args(2)
```

### 字符串形式

```python
td_args_str = qutip.QobjEvo([Id, "t + delta"], args={"delta": 1.0})
td_args_str(2)
```

```python
td_args_str(2, {"delta": 10})
```

### 多参数

即使变量名相同，`QobjEvo` 各项也保有各自参数：

```python
def f(t, w):
    return np.exp(1j * np.pi * w * t)


qevo = (
    n
    + a * qutip.coefficient(f, args={"w": 0.5})
    + a.dag() * qutip.coefficient(f, args={"w": -0.5})
)
qevo(1)
```

```python
# However overwritting the args with change them all:
qevo(1, w=1)
```

### 使用对象

参数值不必是数字。即使是字符串表达式，也可接受 Cython 可原生调用的函数对象，例如 `numpy` 核心函数。

```python
td_args_str = qutip.QobjEvo([Id, "f(t)"], args={"f": np.cos})
td_args_str(0.0)
```

```python
td_args_str(np.pi)
```

### 动态参数

当 `QobjEvo` 在求解器中使用时，求解器状态、派生量或其他内部值可作为参数注入。

这些值通过求解器类的方法设置：

多数求解器支持：

  - `StateFeedback`：以 `Qobj` 或 qutip `Data` 形式提供状态。
  - `ExpectFeedback`：由状态计算得到的期望值。

此外，`mcsolve` 还支持 `CollapseFeedback`（获取塌缩列表），随机求解器支持 `WienerFeedback`（返回轨迹上的 Wiener 函数）。

它们都接受 `default` 输入，用于指定在求解器外访问 `QobjEvo` 时采用的默认值。该值必须是 `QobjEvo` 可接受的合法输入，因为它会在求解器初始化阶段使用。

```python
args = {"state": qutip.MESolver.StateFeedback(default=qutip.fock_dm(4, 2))}


def print_args(t, state):
    print(f"'state':\n{state}")
    return t + state.norm()


td_args = qutip.QobjEvo([Id, print_args], args=args)
td_args(0.5)
```

## 数学运算

`QobjEvo` 支持对含时量子对象有意义的基本数学运算：
  - `QobjEvo` 与 `Qobj` 的加法
  - `QobjEvo` 与 `Qobj` 的减法
  - 与 `QobjEvo`、`Qobj` 或标量的乘法
  - 被标量除法
  - 取负：`-x`
  - 共轭：`QobjEvo.conj()`
  - 伴随（dagger）：`QobjEvo.dag()`
  - 转置：`QobjEvo.trans`

```python
(array_form * 2)(0)
```

```python
(array_form + 1 + a)(0)
```

## 超算符

`qutip.superoperator` 里的函数同样可用于 `QobjEvo`。其中最重要的是 `liouvillian`：`mesolve` 的第一个参数可以直接传 Liouvillian（而不是分开的哈密顿量和塌缩算符），通常这样更快。

```python
liouv = qutip.liouvillian(array_form, c_ops=[constant_form])
liouv(0)
```

## 结语

```python
qutip.about()
```

```python

```
