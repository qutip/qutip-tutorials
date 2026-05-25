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

# 动画演示
关于 QuTiP 的更多信息见 [http://qutip.org](http://qutip.org)


## 概览
QuTiP 提供动画函数，用于可视化量子动力学的时间演化。


```python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from qutip import (ket, basis, tensor, sigmaz, qeye, mesolve, anim_schmidt,
                   complex_array_to_rgb, spin_q_function,
                   anim_spin_distribution, about)
```


```python
# a magic command enabling you to see animations in your jupyter notebook
%matplotlib notebook
```


## 量子比特时间演化
考虑由两个量子比特组成的系统。
其哈密顿量为 $\sigma_z \otimes \mathbf{1}$，
初态为纠缠态 ($\left|10\right>$+$\left|01\right>$)/$\sqrt2$。
该算符作用于第一个量子比特，而第二个量子比特保持不变。


```python
# Hamiltonian
H = tensor(sigmaz(), qeye(2))

# initial state
psi0 = (ket('10')+ket('01')).unit()

# list of times for which the solver should store the state vector
tlist = np.linspace(0, 3*np.pi, 100)

results = mesolve(H, psi0, tlist, [], [])

fig, ani = anim_schmidt(results)
```


上述 magic 命令在某些环境中可能失效，例如 Linux 下运行 Jupyter 或使用 Google Colab。
这时可用下面代码。


```python
HTML(ani.to_jshtml())
```

## 带伴随图像的动画
你可以创建带附加静态图的动画。
注意：不能同时再附加其他动画对象。


```python
compl_circ = np.array([[(x + 1j*y) if x**2 + y**2 <= 1 else 0j
                        for x in np.arange(-1, 1, 0.005)]
                       for y in np.arange(-1, 1, 0.005)])

fig = plt.figure(figsize=(7, 3))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.imshow(complex_array_to_rgb(compl_circ, rmax=1, theme='light'),
           extent=(-1, 1, -1, 1))
plt.tight_layout()
fig, ani = anim_schmidt(results, fig=fig, ax=ax0)
```


## 自定义坐标轴对象
你可能想给动画添加标题和坐标轴标签。
方式与普通绘图一致。


```python
compl_circ = np.array([[(x + 1j*y) if x**2 + y**2 <= 1 else 0j
                        for x in np.arange(-1, 1, 0.005)]
                       for y in np.arange(-1, 1, 0.005)])

fig = plt.figure(figsize=(7, 3))
ax0 = plt.subplot(1, 2, 1)
ax1 = plt.subplot(1, 2, 2)
ax1.set_xlabel("x", fontsize=14)
ax1.set_ylabel("y", fontsize=14)
ax1.imshow(complex_array_to_rgb(compl_circ, rmax=1, theme='light'),
           extent=(-1, 1, -1, 1))
plt.tight_layout()
fig, ani = anim_schmidt(results, fig=fig, ax=ax0)
# add title
ax0.set_title('schmidt')
ax1.set_title('color circle')
```


## 保存
可将动画保存到环境中后再分享。
可用扩展名（gif、mp4 等）取决于环境配置。
更多细节见[官方文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.Animation.html)


```python
# ani.save("schmidt.gif")
```


## 其他动画
QuTiP 使用 `qutip.Qobj` 存储量子态，
但某些函数也会使用 `np.array` 存储数据。
例如 `qutip.spin_q_function` 返回在给定 $\theta$ 和 $\phi$ 网格上的
自旋 Husimi Q 函数矩阵值。
有些动画函数非常适合对这类数据可视化，下面给出一个简单示例。


```python
theta = np.linspace(0, np.pi, 90)
phi = np.linspace(0, 2 * np.pi, 90)
Ps = list()
for i in range(0, 121, 2):
    spin = np.cos(np.pi/2*i/60)*basis(2, 0)+np.sin(np.pi/2*i/60)*basis(2, 1)
    # output np.array matrix
    Q, THETA, PHI = spin_q_function(spin, theta, phi)
    Ps.append(Q)

fig, ani = anim_spin_distribution(Ps, THETA, PHI, projection='3d',
                                  colorbar=True)
```


# 版本信息

```python
about()
```
