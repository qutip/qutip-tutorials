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

```python
import numpy as np
```

# NumPy 数组入门


J.R. Johansson 和 P.D. Nation

关于 QuTiP 的更多信息见 [http://qutip.org](http://qutip.org)


## 引言


到目前为止，我们一直使用列表（list）来存放多个元素。但在数值计算里，列表并不理想。
例如，如果我想给一个数值列表的每个元素都加一，对于 `a = [1, 2, 3]`，我们不能直接写 `a + 1`。


相反，我们不得不写成

```python
a = [1, 2, 3]
for k in range(3):
    a[k] = a[k] + 1
```

如果希望同时对大量元素进行数值运算，或者在程序里构造向量和矩阵，使用列表会很快变得复杂。
这些能力以及更多特性，都可以通过 NumPy **数组（array）** 这一首选数据结构获得。


## NumPy 数组


在 Python 的数值处理场景中，几乎 100% 的时候都会使用 NumPy 模块中的数组来存储和操作数据。
NumPy 数组在使用方式上和 Python 列表很像，但底层是 C 语言数组，因此可以高效执行多维数值计算、
向量与矩阵运算以及线性代数操作。使用切片与**向量化（vectorization）**通常可以显著提速，
并替代许多用列表时不得不写的 for 循环。一般原则是：**for 循环越少，代码性能通常越高**。
要开始使用数组，我们可以先从一个简单列表出发，把它传给 `array` 函数。

```python
a = np.array([1, 2, 3, 4, 5, 6])
print(a)
```

现在我们已经创建了第一个整型数组。注意，打印时它看起来像列表，但数据结构完全不同。
我们也可以创建浮点数、复数甚至字符串数组。

```python
a = np.array([2.0, 4.0, 8.0, 16.0])
b = np.array([0, 1 + 0j, 1 + 1j, 2 - 2j])
c = np.array(["a", "b", "c", "d"])
print(a)
print(b)
print(c)
```

通常在 Python 里创建数组有三种方式：

- 先创建列表，再把列表传给 `array` 函数。

- 使用 NumPy 专门的数组创建函数：**zeros, ones, arange, linspace**。

- 从文件导入数据。


### 从列表创建数组


我们已经看过如何用简单列表创建数组。下面看如何构造更复杂的列表再转成数组。
比如，快速创建从 0 到 9 的列表：

```python
output = [n for n in range(10)]
print(output)
```

这段代码与下面较长写法完全等价：

```python
output = []
for n in range(10):
    output.append(n)
print(output)
```

把它转成数组也很简单：

```python
np.array(output)
```

或者更紧凑一些，直接在 `array` 函数里写列表推导式：

```python
np.array([n for n in range(10)])
```

这同样可以用于构造更复杂的数组：

```python
np.array([2.0 * k**0.563 for k in range(0, 10, 2)])
```

### NumPy 中的数组创建（更多可见 [NumPy 文档](http://docs.scipy.org/doc/numpy/reference/routines.array-creation.html)）


NumPy 提供了几个非常重要的数组创建函数，会让你的工作轻松很多。
例如，创建全零或全一数组非常直接：

```python
np.zeros(5)
```

```python
np.ones(10)
```

不过最常用的是 [**```arange```**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html#numpy.arange)
和 [**```linspace```**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)。
`arange` 类似 `range`，在区间内生成等间距数值；`linspace` 则在线性区间内生成指定数量的点。

```python
np.arange(5)
```

```python
np.arange(0, 10, 2)
```

```python
# make an array of 20 points linearly spaced from 0 to 10
np.linspace(0, 10, 20)
```

```python
np.linspace(-5, 5, 15)  # 15 points in range from -5 to 5
```

## 数组与列表的差异


在简单体验过数组后，现在来说明 NumPy 数组与 Python 列表的核心区别。


Python 列表非常通用，可以容纳任意数据类型的组合。
但 NumPy **数组只能包含一种数据类型**（整型、浮点、字符串、复数等）。
如果把不同类型混在一起，`array` 会发生**类型提升（upcast）**，把数据统一成同一类型。

```python
np.array([1, 2, 3.14])  # [int,int,float] -> [float,float,float]
```

整型和浮点之间的提升通常问题不大，但把字符串和数字混在数组里常会带来麻烦：

```python
np.array([1.0, 1 + 1j, "hello"])  # array data is upcast to strings
```

如果需要，我们也可以用 `dtype`（data type）关键字参数手动指定数组类型。
常见类型包括：`int, float, complex, bool, str, object` 等。
例如，把整型列表转换为浮点数组：

```python
np.array([1, 2, 3, 4, 5], dtype=float)
```

```python
np.arange(2, 10, 2, dtype=complex)
```

```python
np.array([k for k in range(10)], dtype=str)
```

与列表不同，**数组一旦创建，不能直接增删元素**。
因此创建前应先明确数组大小。


因为数组内部类型统一，像加法、乘法这类运算可以在 C 层实现，速度快且内存效率高。
数组上的数学操作是**逐元素（elementwise）**进行的，也就是每个元素以相同规则被处理。
这正是**向量化**的典型体现。例如：

```python
a = np.array([1, 2, 3, 4])
5.0 * a  # This gets upcasted because 5.0 is a float
```

```python
5 * a**2 - 4
```

回想一下，这些操作在 Python 列表上都不能直接工作。


## 在数组上使用 NumPy 函数


请记住，NumPy 内置了大量[数学函数集合](http://docs.scipy.org/doc/numpy/reference/routines.math.html)。
以数组为数据结构时，这些函数会更强大，因为它们能快速地对整个数组逐元素应用。
这同样是向量化，通常能显著提升代码速度。

```python
x = np.linspace(-np.pi, np.pi, 10)
np.sin(x)
```

```python
x = np.array([x**2 for x in range(4)])
np.sqrt(x)
```

```python
x = np.array([2 * n + 1 for n in range(10)])
sum(x)  # sums up all elements in the array
```

## 数组上的布尔运算


与其他数学函数类似，我们也可以把条件判断应用到数组上，检查每个元素是否满足表达式。
例如，要找出数组中小于 0 的元素位置，可以这样做：

```python
a = np.array([0, -1, 2, -3, 4])
print(a < 0)
```

结果是另一个布尔数组（`True/False`），表示对应元素是否小于 0。
同样地，我们也可以找出数组中的所有奇数：

```python
a = np.arange(10)
print((np.mod(a, 2) != 0))
```

## NumPy 数组切片


和列表一样，数组也可以切片，既可以取部分元素，也可以修改部分元素。
例如，我们从一个数组中每隔三个元素取一个：

```python
a = np.arange(20)
a[3::3]
```

现在把这些元素都设为 -1：

```python
a[3::3] = -1
print(a)
```

还可以用切片把数组反转：

```python
a = np.arange(10)
a[::-1]
```

最后，如果我们只想要满足某个条件的元素怎么办？
回忆一下，数组条件判断会返回布尔数组。
我们可以把这个布尔数组当索引，只取布尔值为 `True` 的元素。

```python
a = np.linspace(-10, 10, 20)
print(a[a <= -5])
```

但要注意，不能直接写多重区间条件：`print(a[-8 < a <= -5])`。

原因是：计算机无法把一组 `True/False` 布尔数组直接解释成一个单一真假值。


## 示例：重写埃拉托斯特尼筛法


下面我们用数组重写“埃拉托斯特尼筛法”中的大部分 for 循环，
替代原来基于列表的写法。这样代码更易读，而且在求较大素数时通常更快。
原始代码核心部分如下：

```python
N = 20
# generate a list from 2->N
numbers = []
for i in range(2, N + 1):  # This can be replaced by array
    numbers.append(i)
# Run Seive of Eratosthenes algorithm marking nodes with -1
for j in range(N - 1):
    if numbers[j] != -1:
        p = numbers[j]
        for k in range(j + p, N - 1, p):  # This can be replaced by array
            numbers[k] = -1
# Collect all elements not -1 (these are the primes)
primes = []
for i in range(N - 1):  # This can be replaced by array
    if numbers[i] != -1:
        primes.append(numbers[i])
print(primes)
```

使用数组后，代码可简化为：

```python
N = 20
# generate a list from 2->N
numbers = np.arange(2, N + 1)  # replaced for-loop with call to arange
# Run Seive of Eratosthenes algorithm
# by marking nodes with -1
for j in range(N - 1):
    if numbers[j] != -1:
        p = numbers[j]
        numbers[j + p: N - 1: p] = -1  # replaced for-loop by slicing array
# Collect all elements not -1 (these are the primes)
# Use conditional statement to get elements !=-1
primes = numbers[numbers != -1]
print(primes)
```

<h1 align="center">教程结束</h1> 
<h3 align="center"><a href="http://qutip.org/tutorials.html">返回 QuTiP 教程页面</a></h3> 
