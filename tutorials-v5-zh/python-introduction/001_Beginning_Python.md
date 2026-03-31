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

# Python 入门


P.D. Nation 和 J.R. Johansson

关于 QuTiP 的更多信息见 [http://qutip.org](http://qutip.org)


## 导入
这里导入后续要用到的函数。

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
```

## 用 Python 做简单计算


第一步，我们把交互式 Python 命令行工具 **iPython** 当作基础计算器来用。
加、减、乘和你在纸上写算式的方式基本一致。

```python
10 + 5
```

```python
10 - 157
```

```python
4 / 3
```

```python
(50 - 4) * 10 / 5
```

不过，像幂运算 $4^{4}$，写法会有所不同。

```python
4**4
```

我们也可以写成数学上等价的 $4^{4.0}$。但在计算机内部，结果类型与上面并不完全一样。

```python
4**4.0
```

## 整数与浮点数


计算机中的所有信息都必须以二进制（0 和 1）表示（例如 $461\rightarrow 111001101$）。
每个 0 或 1 叫作一个 **bit（比特）**。给定 $N$ 个比特，可以表示区间
$[0,2^{N-1}]$ 内的整数，其中 $-1$ 来自首位常用于表示正负号。

不过，在固定比特数下，不可能精确存储任意实数。
因此，对一个随机数进行二进制转换时，除非它能被 2 的幂精确表示，否则会出现精度损失，
这就是**舍入误差（roundoff error）**。


在计算机中处理数字时，通常要区分两类：


- **整数（Integers）** - （1,2,4,-586,..）属于**定点数（fixed-point）**，
定点的含义是小数位数固定（对整数来说小数位为 0）。这类数在计算机中可精确存储。


- **双精度/浮点数（Doubles/Floats）** - （3.141,0.21,-0.1,..）属于**浮点数（floating-point）**，
相当于科学计数法的二进制版本，例如 $c=2.99792458\times 10^{8}$。
Double（双精度）通常使用 64 bit，一般精确到约第 15 或 16 位十进制。
Float（单精度）通常使用 32 bit，一般精确到 6-7 位。
**严肃的科学计算通常需要结合整数与双精度（64 bit）浮点数**。

```python
7 + 0.000000000000001
```

```python
7 + 0.0000000000000001
```

```python
0.1 + 0.2
```

最后一个例子清楚地说明：计算机无法精确存储大多数十进制浮点数。
浮点精度损失可由**机器精度** $\epsilon_{\rm m}$ 来刻画，其定义为满足

$$1_{\rm c}+\epsilon_{\rm m}\neq 1_{\rm c}$$

的最小正数，其中下标 $1_{\rm c}$ 表示“计算机中的 1”。
因此任意真实数 $N$ 与其浮点表示 $N_{\rm c}$ 关系为

$$N_{\rm c}=N\pm \epsilon, \ \ \forall~|\epsilon|< \epsilon_{\rm m}.$$

**要点总结** - 所有不能由 2 的因子精确表示的双精度十进制数，都会在约第 15 位十进制出现误差。
如果不加注意，这会传递到你的数值解中。


## 用 NumPy 提升 Python 的数值能力


Python 原生数学能力主要限于基本算术。
因此我们会使用 NumPy 模块中的函数来做更复杂、更高效的计算。
我们已经在开头导入了 NumPy，可通过 `np` 调用。

现在可以做更丰富的数值运算：

```python
np.exp(2.34)
```

```python
np.sqrt(5)
```

```python
np.sinc(0.5)
```

## 变量


如果希望保存计算中的数值和结果，就需要用 `=` 定义变量：

```python
radius = 5
area = np.pi * radius**2
area
```

可以看到，变量名在 `=` 左侧，赋给它的值在右侧。
这里还使用了 NumPy 预定义的 `pi` 常量。变量随后可用于其他表达式。

如果一个已定义变量再次出现在 `=` 左侧，它的原值会被新值覆盖。

```python
x = 10
x = (x**2 + 25) / 10
x
```

这与数学方程 $10x=x^{2}+25$（解为 $x=5$）并不相同。
因此要记住，程序中的 `=` **不等同于**数学中的等号关系。


如果你在未定义变量前就直接使用它，会发生什么？

Python 会报错提示变量未定义。此外，Python 语言有一些保留字不能作为变量名：

    and, as, assert, break, class, continue, def, del, elif, else, except, 
    exec, finally, for, from, global, if, import, in, is, lambda, not, or,
    pass, print, raise, return, try, while, with, yield
    
除了这些保留字外，变量名可由字母或下划线 `\_` 开头，后接任意字母数字和 `\_`。
注意大小写会被视为不同变量。

```python
_freq = 8
Oscillator_Energy = 10
_freq * Oscillator_Energy
```

### 变量命名的一些规则


虽然 Python 中命名方式很多，但最好在项目内保持统一。
在本课程中，我们的变量名统一使用小写字符。


```python
speed_of_light = 2.9979 * 10**8
spring_constant = np.sqrt(2 / 5)
```

同时建议变量名尽量反映其物理含义。


## 字符串


很多时候我们希望打印文本、接收用户输入，或者直接把字母/单词当作变量处理（例如 DNA 分析）。
这些都可以通过**字符串（string）**完成。我们已经见过一个最简单的字符串：

```python
"Hello Class"
```

也可以用单引号，例如 `'Hello Class'`。

如果字符串本身要包含引号，就需要混用单双引号。

```python
"How was Hwajung's birthday party?"
```

和整数、浮点一样，字符串也可赋值给变量，甚至可直接拼接。

```python
a = "I like "  # There is a blank space at the end of this string.
b = "chicken and HOF"
a + b
```

注意变量 `a` 字符串末尾的空格，它让 `like` 和 `chicken` 之间自动分开。


如果我们想把字符串和整数/浮点一起输出，可使用内置 `print` 函数：

```python
temp = 23
text = "The temperature right now is"
print(text, temp)
```

注意 `print` 会自动在参数之间加空格。
`print` 可以接收任意数量的字符串、整数、浮点或其他变量，
并自动转换为字符串后输出。


## 列表


很多时候我们希望把多个变量放进一个对象中。
在 Python 中可用 **`list`** 数据类型实现。

```python
shopping_list = ["eggs", "bread", "milk", "bananas"]
```

如果想访问列表中某个元素，需要用方括号里的**索引（index）**。

```python
shopping_list[2]
```

我们看到索引 `2` 对应字符串 `"milk"`。虽然它是第三个元素，
但 Python（与 C 类似）把第一个元素索引定义为 `0`。

```python
shopping_list[0]
```

这一点很重要，刚开始需要适应一段时间。
如果想从后往前访问，可以用负索引：

```python
shopping_list[-1]
```

```python
shopping_list[-2]
```

如果想知道列表有多少元素，可用 `len`，它返回列表长度（整数）：

```python
len(shopping_list)
```

若要增删元素，可分别使用 `append` 和 `remove`：

```python
shopping_list.append("apples")
shopping_list
```

```python
shopping_list.remove("bread")
shopping_list
```

列表元素不必是同一数据类型，你可以混合任意类型：

```python
various_things = [1, "hello", -1.234, [-1, -2, -3]]
various_things
```

这些元素都能按通常方式访问：

```python
various_things[0]
```

```python
various_things[-1]
```

```python
various_things[3][1]
```

## 遍历列表与 Python 缩进规则


列表的核心价值之一是：你常常想对每个元素逐个执行相同操作。
这种过程叫**迭代（iteration）**，在 Python 中用 `for` 完成：

```python
items = [
    "four calling birds",
    "three french hens",
    "two turtle doves",
    "a partridge in a pear tree",
]
for thing in items:
    print(thing)
```

这里 `thing` 是循环变量，它依次取列表 `items` 中每个元素，再传给 `print`。
变量名可以自由命名。

```python
for variable in items:
    print(variable)
```

另一点很关键：冒号 `:` 之后的 `print` 语句必须缩进。
Python 语法要求冒号后跟随缩进代码块（**block**）。
如果不缩进，Python 会报错。


代码块是编程语言中用于组织结构和流程控制的基础。
在上例中，所有缩进内容都会对列表每个元素执行一次。

```python
for variable in items:
    print("My true love gave to me", variable)
```

## 列表切片


如果只想取列表中的一部分元素，可使用**切片（slicing）**。
切片适用于任何**序列（sequence）**，例如列表、字符串，以及后面会讲的数组。
考虑下面的 `shopping_list`：

```python
shopping_list = ["eggs", "bread", "milk", "bananas", "apples"]
```

获取第一个元素时，我们用单个索引：

```python
shopping_list[0]
```

如果要获取前三个元素，可写：

```python
shopping_list[0:3]
```

也可以取最后两个元素：

```python
shopping_list[-2:]
```

还可以更灵活，使用第三个参数设置步长，例如取偶数位元素：

```python
shopping_list[0::2]
```

## 条件语句


到这里我们看过多种数据类型（整数、浮点、列表、字符串），
但还没讨论如何比较两个变量。
例如如何判断两个整数 $a$ 和 $b$ 是否相等？如何判断 $a\ge b$？
这要靠**条件语句（conditional statements）**。
布尔逻辑基本比较操作包括：
相等（`==`）、不等（`!=`）、大于（`>`）、大于等于（`>=`）、小于（`<`）、小于等于（`<=`）。
这些操作接收两个变量，返回布尔值 `True` 或 `False`。例如：

```python
a = 5
b = 8
a > b
```

```python
c = 0
c <= 0, c >= 0
```

```python
a = 5
b = 6
a == b, a != b
```

还要注意，在 Python 中 `1` 和 `0` 分别等价于 `True` 和 `False`。

```python
t = True
f = False
t == 1, f == 0
```

我们也能把多个比较连起来：

```python
a = -1
b = 4
c = 10
d = 11
a < b < c != d
```

这些操作也可用于列表和字符串：

```python
[4, 5, 6] >= [4, 5, 7]
```

```python
[4, 5, 6] <= [4, 5, 7]
```

```python
"today" == "Today"
```

### 条件语句与流程控制


条件语句的核心用途是控制程序执行流程。
条件判断结果可配合 `if/else` 和 `while` 等语句使用。

```python
today = "friday"
if today == "friday":
    print("We have class today :(")  # this is a code block
else:
    print("No class today :)")  # this is also a code block
```

`if` 下方代码块仅在 `today=='friday'` 为 `True` 时执行。
若条件为 `False`，则执行 `else` 代码块。
还可以使用 `elif` 在 `if` 后追加多分支判断：

```python
today = "thursday"
if today == "friday":
    print("We have class today :(")
elif today == "thursday":
    print("Our assignment is due today :(")
else:
    print("No class today :)")
```

另一个重要流程控制结构是 **`while` 循环**：
只要循环开头条件为 `True`，代码块就会重复执行；条件变为 `False` 时终止。

```python
n = 0
while n <= 10:  # evaluate code block until n>10
    print("The current value of n is:", n)
    n = n + 1  # increase the value of n by 1
```

使用 `while` 时必须保证条件最终会变成 `False`，
否则程序会进入永不结束的**死循环（infinite loop）**。


### 示例：奇数与偶数


下面判断 [1,10] 之间每个数是奇数还是偶数。

```python
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    if np.remainder(n, 2) == 0:
        print(n, "is even")
    else:
        print(n, "is odd")
```

手动输入长整数列表很麻烦。幸运的是，Python 内置了 `range` 函数，
能很方便地生成整数序列。因此上面代码可改写为：

```python
for n in range(1, 11):
    if np.remainder(n, 2) == 0:
        print(n, "is even")
    else:
        print(n, "is odd")
```

注意虽然上界写到 11，`range` 实际只到 10。
`range` 生成的序列**不包含终点**。
如果想从 0 开始，可写 `range(11)`。
也可以指定任意步长：

```python
for n in range(0, 11, 2):
    print(n)
```

`range` 返回的并不是列表，而是**生成器（generator）**对象。
通常它与 `for` 搭配使用最合适。


### 示例：斐波那契数列


遵循 Python 文档中的常见示例，我们计算斐波那契数列前 10 项：

```python
n = 10
fib = [0, 1]
for i in range(2, n):
    fib.append(fib[i - 1] + fib[i - 2])
print(fib)
```

如果愿意，也可以用 `while` 循环实现：

```python
n = 2
fib = [0, 1]
while n < 10:
    fib.append(fib[n - 1] + fib[n - 2])
    n = n + 1
print(fib)
```

## 编写脚本与函数


到目前为止，我们运行了不少代码片段，但还谈不上完整编程。
别忘了，Python 是一门脚本语言。
因此大多数时候，我们会编写包含常量、变量、数据结构、函数、注释等内容的**脚本**，
让它们完成更复杂任务。


###  脚本


Python 脚本本质上是一个以 **.py** 结尾、包含 Python 代码的文本文件。
Python 脚本也常被称为 Python **程序（program）**。
在任意编辑器中新建文件后，就可以输入 Python 命令。


在开始写脚本前，先看看推荐格式：

```python
# This is an example script for the P461 class
# Here we will calculate the series expansion
# for sin(x) up to an arbitrary order N.
#
# Paul Nation, 02/03/2014

N = 5  # The order of the series expansion
x = np.pi / 4.0  # The point at which we want to evaluate sine

ans = 0.0
for k in range(N + 1):
    ans = ans + (-1) ** k * x ** (1 + 2 * k) / factorial(1 + 2 * k)
print("Series approximation:", ans)
print("Error:", np.sin(x) - ans)
```

可以看到，脚本有四个主要部分：
第一，**注释**部分，说明脚本功能与创建时间。
在 Python 中注释以 `#` 开头，后续内容会被解释器忽略。
第二，导入所需模块/函数的部分。
第三，定义脚本要使用的常量，并尽量添加说明注释。
最后，主体计算代码放在这些部分之后。


### 函数


现在我们终于来到编程语言最核心的部分之一：**函数（function）**。
函数是一段完成特定任务的代码块。函数通常接收输入参数，
对输入做运算，再返回一个或多个结果。函数可重复使用，
也可在其他函数内部被调用。
下面把前面的 $sin(x)$ 脚本改写为函数形式：

```python
N = 5  # The order of the series expansion
x = np.pi / 4.0  # The point at which we want to evaluate sine


def sine_series(x, N):
    ans = 0.0
    for k in range(N + 1):
        ans = ans + (-1) ** k * x ** (1 + 2 * k) / factorial(1 + 2 * k)
    return ans


result = sine_series(x, N)
print("Series approximation:", result)
print("Error:", np.sin(x) - result)
```

可以看到，函数通过关键字 `def` 定义（即 define）。
后面是函数名与括号中的输入参数。
函数代码块结束后，用 `return` 指定输出变量或数据结构。
一个通用函数模板如下：

```python
def function_name(arg1, arg2):
    "Block of code to run"
    "..."
    return result
```

再次强调，冒号 `:` 后属于函数体的语句都必须缩进。
函数的优势在于：只需修改脚本顶部参数，就能复用同一段代码处理不同问题。

在函数内部定义的变量称为**局部变量（local variables）**，
只在该函数代码块内有效。前面的例子中，`k` 就是局部变量。
输入参数和返回值本身不属于局部变量。
函数执行结束后，局部变量会从内存中清除。
因此如果希望把结果带出函数，必须在结束前 `return`。


如果函数要返回多个结果，只需用逗号分隔即可。

```python
N = 100  # Number of points to generate


def random_coordinates(N):
    x_coords = []
    y_coords = []
    for n in range(N):
        xnew, ynew = np.random.random(2)
        x_coords.append(xnew)
        y_coords.append(ynew)
    return x_coords, y_coords


xc, yc = random_coordinates(N)
plt.plot(xc, yc, "ro", markersize=8)
plt.show()
```

```python
N = 20  # Number of points to generate


def random_coordinates(N):
    x_coords = []
    y_coords = []
    for n in range(N):
        xnew, ynew = np.random.random(2)
        x_coords.append(xnew)
        y_coords.append(ynew)
    return x_coords, y_coords


def dist2d(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def max_dist(xc, yc):
    max_dist = 0.0
    num_points = len(xc)
    for ii in range(num_points):
        for jj in range(num_points):
            dist = dist2d(xc[ii], yc[ii], xc[jj], yc[jj])
            if dist > max_dist:
                max_dist = dist
                xvals = [xc[ii], xc[jj]]
                yvals = [yc[ii], yc[jj]]
    return max_dist, xvals, yvals


xc, yc = random_coordinates(N)
max_dist, pnt1, pnt2 = max_dist(xc, yc)
plt.plot(xc, yc, "ro", markersize=8)
plt.plot(pnt1, pnt2, "b-", lw=2)
plt.show()
```

显然最后这个例子更复杂，尤其函数功能不看文档时不容易迅速理解。
哪怕是你自己写的函数，过一段时间也可能忘记用途。
所以应在脚本里写文档与注释。
下面通过 `max_dist` 展示如何为函数写规范说明：

```python
def max_dist(xc, yc):
    """
    Finds the maximum distance between any two points
    in a collection of 2D points.  The points corresponding
    to this distance are also returned.

    Parameters
    ----------
    xc : list
        List of x-coordinates
    yc : list
        List of y-coordinates

    Returns
    -------
    max_dist : float
        Maximum distance
    xvals : list
        x-coodinates of two points
    yvals : list
        y-coordinates of two points

    """
    max_dist = 0.0  # initialize max_dist
    num_points = len(xc)  # number of points in collection
    for ii in range(num_points):
        for jj in range(num_points):
            dist = dist2d(xc[ii], yc[ii], xc[jj], yc[jj])
            if dist > max_dist:
                max_dist = dist
                xvals = [xc[ii], xc[jj]]
                yvals = [yc[ii], yc[jj]]
    return max_dist, xvals, yvals
```

`"""..."""` 之间的内容称为 **docstring（文档字符串）**。
它能帮助不熟悉该函数的人快速理解：函数做什么、输入参数是什么、返回值是什么。
同时也建议在局部变量旁写必要注释，让读者知道变量作用。
虽然一开始看起来麻烦，但坚持写 docstring 会显著提升你的编程质量。


<h1 align="center">教程结束</h1> 
<h3 align="center"><a href="http://qutip.org/tutorials.html">返回 QuTiP 教程页面</a></h3> 
