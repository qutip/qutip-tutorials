---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 层级运动方程（HEOM）示例

“层级运动方程”（hierarchical equations of motion, HEOM）方法是一种强大的数值方法，
可用于求解与非马尔可夫、非微扰环境耦合的量子系统的动力学与稳态问题。
该方法最初发展于物理化学领域，随后也扩展到固体物理、光学、单分子电子学和生物物理等问题。

QuTiP 中 HEOM 的实现细节可见：https://arxiv.org/abs/2010.10806。

本示例集合来源于该论文，展示了如何使用 QuTiP 的 HEOM 对多种与玻色或费米热浴耦合的系统进行建模与动力学研究。

## Notebook 总览

<!-- markdown-link-check-disable -->

* [示例 1a：自旋-热浴模型（入门）](./heom-1a-spin-bath-model-basic.ipynb)

* [示例 1b：自旋-热浴模型（超强耦合）](./heom-1b-spin-bath-model-very-strong-coupling.ipynb)

* [示例 1c：自旋-热浴模型（欠阻尼情形）](./heom-1c-spin-bath-model-underdamped-sd.ipynb)

* [示例 1d：自旋-热浴模型（谱与关联函数拟合）](./heom-1d-spin-bath-model-ohmic-fitting.ipynb)

* [示例 1e：自旋-热浴模型（纯退相干）](./heom-1e-spin-bath-model-pure-dephasing.ipynb)

* [示例 2：Fenna-Mathews-Olsen 复合体（FMO）中的动力学](./heom-2-fmo-example.ipynb)

* [示例 3：量子热输运](./heom-3-quantum-heat-transport.ipynb)

* [示例 4：非马尔可夫环境中的动力学解耦](./heom-4-dynamical-decoupling.ipynb)

* [示例 5a：费米单杂质模型](./heom-5a-fermions-single-impurity-model.ipynb)

* [示例 5b：离散玻色模与杂质耦合 + 费米引线](./heom-5b-fermions-discrete-boson-model.ipynb)

<!-- markdown-link-check-enable -->
