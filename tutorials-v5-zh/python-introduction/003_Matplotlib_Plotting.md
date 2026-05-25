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
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, Image
from scipy.special import airy

%matplotlib inline
```

```python
HTML(
    """<script>
code_show=true;
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
}
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click
<a href="javascript:code_toggle()">here</a>."""
)
```

# 使用 Matplotlib 在 Python 中绘图


P.D. Nation 和 J.R. Johansson

关于 QuTiP 的更多信息见 [http://qutip.org](http://qutip.org)


## 引言


在今天的科学研究中，能够绘制高质量且信息充分的图像是一项基础能力。
图做得不好，工作展示效果也会受影响。好的可视化不仅能准确传达科学信息，
还能吸引读者关注你的研究。很多时候，图表质量会直接影响工作的整体传播效果。
因此，我们将学习如何使用 Python 模块 [Matplotlib](http://matplotlib.org)
绘制高质量、可用于发表的图。

```python
Image(filename="images/mpl.png", width=700, embed=True)
```

基础 2D 绘图
-----------------
在 Python 中生成图像前，我们需要先导入 Matplotlib 主模块。
这一步我们已在 notebook 开头完成。


### 绘制线条
绘制简单函数（例如正弦函数）非常容易。我们只需要两个数组：
一个存放 x 值，一个存放 f(x) 值。

```python
x = np.linspace(-np.pi, np.pi)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

这里 `plot` 命令负责生成图像，但只有执行 `show()` 才会显示。
如果需要，也可以添加坐标轴标签和标题。顺便把线条颜色改成红色并设置为虚线。

```python
x = np.linspace(-np.pi, np.pi)
y = np.sin(x)
plt.plot(x, y, "r--")  # make line red 'r' and dashed '--'
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin(x)")
plt.show()
```

其中 `'r'` 表示红色。你也可以使用内置的其他颜色：

- 'b': 蓝色
- 'g': 绿色
- 'r': 红色
- 'c': 青色
- 'm': 品红
- 'y': 黄色
- 'k': 黑色
- 'w': 白色


我们也可以通过 `color` 关键字参数指定线条颜色。

```python
x = np.linspace(-np.pi, np.pi)
y = np.sin(x)
# Here a string from 0->1 specifies a gray value.
plt.plot(x, y, "--", color="0.75")
plt.show()
```

```python
x = np.linspace(-np.pi, np.pi)
y = np.sin(x)
plt.plot(x, y, "-", color="#FD8808")  # We can also use hex colors if we want.
plt.show()
```

线型可从实线 `''` 或 `'-'` 改为虚线 `'--'`、点线 `'.'`、点划线 `'-.'`、
点加实线 `'.-'` 或小点线 `':'`。也可以使用 `linestyle` 或 `ls` 关键字参数。
下面用 **subplot** 函数展示这些样式。`subplot` 会按给定的行列网格显示多个子图。
只需一个 `show()` 就能一次性展示全部子图。为保证显示效果，我们还可用
`figure(figsize=...)` 控制图像宽高（单位英寸）。

```python
x = np.linspace(-np.pi, np.pi)
y = np.sin(x)
plt.figure(figsize=(12, 3))  # This controls the size of the figure
plt.subplot(2, 3, 1)  # This is the first plot in a 2x3 grid of plots
plt.plot(x, y)
plt.subplot(2, 3, 2)  # this is the second plot
plt.plot(x, y, linestyle="--")  # Demo using 'linestyle' keyword arguement
plt.subplot(2, 3, 3)
plt.plot(x, y, ".")
plt.subplot(2, 3, 4)
plt.plot(x, y, "-.")
plt.subplot(2, 3, 5)
plt.plot(x, y, ".-")
plt.subplot(2, 3, 6)
plt.plot(x, y, ls=":")  # Demo using 'ls' keyword arguement.
plt.show()
```

如果想调整线宽，可以使用 `linewidth` 或 `lw` 关键字参数，并传入浮点数。

```python
x = np.linspace(0, 10)
y = np.sqrt(x)
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(x, y)
plt.subplot(1, 3, 2)
plt.plot(x, y, linewidth=2)
plt.subplot(1, 3, 3)
plt.plot(x, y, lw=7.75)
plt.show()
```

如果要在同一张图上绘制多条曲线，可以多次调用 `plot`，也可以一次调用并传入多组数据。

```python
x = np.linspace(0, 10)
s = np.sin(x)
c = np.cos(x)
sx = x * np.sin(x)
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
# call three different plot functions
plt.plot(x, s, "b")
plt.plot(x, c, "r")
plt.plot(x, sx, "g")

plt.subplot(1, 2, 2)
# combine multiple lines in one call to plot
plt.plot(x, s, "b", x, c, "r", x, sx, "g")
plt.show()
```

### 设置坐标轴范围
假设我们希望限制 x 轴或 y 轴的显示范围（或同时限制）。
在 Matplotlib 中可通过 `xlim` 和 `ylim` 实现。

```python
x = np.linspace(-np.pi, np.pi, 100)
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(x, np.sin(x), lw=2)
plt.subplot(1, 3, 2)
plt.plot(x, np.sin(x), lw=2, color="#740007")
plt.xlim([-np.pi, np.pi])  # change bounds on x-axis to [-pi,pi]
plt.subplot(1, 3, 3)
plt.plot(x, np.sin(x), "^", ms=8, color="0.8")
plt.xlim([-1, 1])  # change bounds on x-axis to [-1,1]
plt.ylim([-0.75, 0.75])  # change bounds on y-axis to [-0.75,0.75]
plt.show()
```

### 保存图像
既然已经会画图，下一步通常就是保存到论文、报告或网页中。
这在 Matplotlib 中非常简单：调用 `savefig` 即可。

```python
x = np.linspace(-np.pi, np.pi, 100)
plt.figure(figsize=(12, 3))
plt.subplot(1, 3, 1)
plt.plot(x, np.sin(x), lw=2)
plt.subplot(1, 3, 2)
plt.plot(x, np.sin(x), lw=2, color="#740007")
plt.xlim([-np.pi, np.pi])
plt.subplot(1, 3, 3)
plt.plot(x, np.sin(x), "^", ms=8, color="0.8")
plt.xlim([-1, 1])
plt.ylim([-0.75, 0.75])
plt.savefig(
    "axes_example.png"
)  # Save the figure in PNG format in same directory as script
```

`savefig` 会按字符串中给出的文件名和扩展名保存图像。
文件名可自定义，但扩展名（此处是 `.png`）必须是 Matplotlib 支持的格式。
在本课程中我们主要使用 PNG（`.png`）和 PDF（`.pdf`）格式。


### 使用不同形状表示数据点
除了线和点，Matplotlib 还支持多种图形标记来表示数据点。
这些形状称为 **marker**，其颜色和大小也可控制。
标记类型很多（见 [markers 文档](http://matplotlib.org/api/markers_api.html)），
这里演示几种常见类型：`'*'` 星号、`'o'` 圆形、`'s'` 方形、`'+'` 加号，
并以 Airy 函数为例。

```python
x = np.linspace(-1, 1)
Ai, Aip, Bi, Bip = airy(x)
plt.plot(x, Ai, "b*", x, Aip, "ro", x, Bi, "gs", x, Bip, "k+")
plt.show()
```

我们还可以使用 `markersize` 或 `ms` 调整标记大小，
用 `markercolor` 或 `mc` 设置标记颜色。


其他类型的 2D 图
-----------------------
到目前为止我们主要使用 `plot` 函数来绘制二维图。
但 Matplotlib 还提供了很多其他 2D 图函数。
完整类型可参考 [matplotlib gallery](http://matplotlib.org/gallery.html)，
这里仅展示几个实用示例。

```python
x = np.linspace(-1, 1.0, 100)
plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
y = x + 0.25 * np.random.randn(len(x))
plt.scatter(x, y, color="r")  # plot collection of (x,y) points
plt.title("A scatter Plot")
plt.subplot(2, 2, 2)
n = np.array([0, 1, 2, 3, 4, 5])
plt.bar(
    n, n**2, align="center", width=1
)  # aligns the bars over the x-numbers, and width=dx
plt.title("A bar Plot")
plt.subplot(2, 2, 3)
plt.fill_between(
    x, x**2, x**3, color="green"
)  # fill between x**2 & x**3 with green
plt.title("A fill_between Plot")
plt.subplot(2, 2, 4)
plt.title("A hist Plot")
r = np.random.randn(50)  # generating some random numbers
plt.hist(r, color="y")  # create a histogram of the random number values
plt.show()
```

这些图形元素的颜色、大小等参数控制方式，和常规 `plot` 基本一致。


更多参考资料
----------------------------------

- [MatplotLib Gallery](http://matplotlib.org/gallery.html) : 展示 Matplotlib 能力的图形画廊。

- [Matplotlib Examples](http://matplotlib.org/examples/index.html) : 大量示例，演示不同绘图任务的实现方式。

- [Guide to 2D & 3D Plotting](http://nbviewer.jupyter.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb) : Robert Johansson 编写的 Matplotlib 绘图指南。


<h1 align="center">教程结束</h1> 
<h3 align="center"><a href="http://qutip.org">返回 QuTiP 网站</a></h3> 
