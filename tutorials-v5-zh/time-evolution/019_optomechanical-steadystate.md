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

# 稳态：单光子强耦合区的光力学系统


P.D. Nation and J.R. Johansson

QuTiP 更多信息请见 [http://qutip.org](http://qutip.org)

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from qutip import (about, destroy, hinton, ptrace, qdiags, qeye, steadystate,
                   tensor, wigner, wigner_cmap, basis, fidelity, mesolve)

%matplotlib inline
```

## 光力学哈密顿量


光力学哈密顿量来自光学腔中光压作用；其中一个腔镜可以发生机械振动（可形变）。

```python
Image(filename="images/optomechanical_setup.png", width=500, embed=True)
```

设 $a^{+}$、$a$ 分别为腔模升/降算符，$b^{+}$、$b$ 分别为机械振子升/降算符，则在经典单色泵浦驱动下，光力学系统哈密顿量可写为


\begin{equation}
\frac{\hat{H}}{\hbar}=-\Delta\hat{a}^{+}\hat{a}+\omega_{m}\hat{b}^{+}\hat{b}+g_{0}(\hat{b}+\hat{b}^{+})\hat{a}^{+}\hat{a}+E\left(\hat{a}+\hat{a}^{+}\right),
\end{equation}


其中 $\Delta=\omega_{p}-\omega_{c}$ 是泵浦（$\omega_p$）与腔模（$\omega_c$）频率失谐，$g_0$ 是单光子耦合强度，$E$ 是泵浦幅度。已知在单光子强耦合区，当每个声子引起的腔频移大于腔线宽，即 $g_{0}/\kappa \gtrsim 1$（$\kappa$ 为腔衰减率），且单个光子使机械振子位移超过零点振幅，即 $g_{0}/\omega_{m} \gtrsim 1$（等价于 $g^{2}_{0}/\kappa\omega_{m} \gtrsim 1$）时，机械振子可在系统+环境动力学下被驱动到非经典稳态。下面我们将使用 QuTiP 的稳态求解器探索该状态，并比较不同求解器。


## 求解稳态密度矩阵


光力学系统（含环境）的稳态密度矩阵可由 Liouvillian 超算符 $\mathcal{L}$ 给出：

\begin{equation}
\frac{d\rho}{dt}=\mathcal{L}\rho=0\rho,
\end{equation}

其中 $\mathcal{L}$ 通常取 Lindblad 形式
\begin{align}
\mathcal{L}[\hat{\rho}]=&-i[\hat{H},\hat{\rho}]+\kappa \mathcal{D}\left[\hat{a},\hat{\rho}\right]\\
&+\Gamma_{m}(1+n_{\rm{th}})\mathcal{D}[\hat{b},\hat{\rho}]+\Gamma_{m}n_{\rm th}\mathcal{D}[\hat{b}^{+},\hat{\rho}], \nonumber
\end{align}

其中 $\Gamma_m$ 为机械振子与热环境的耦合强度，$n_{th}$ 为热环境平均占据数。按惯例，这里假设腔模耦合到真空。

虽然稳态本质上是一个本征值问题，但由于 $\mathcal{L}$ 的非厄米结构，以及随截断希尔伯特空间维度增加而恶化的条件数，数值求解并不简单。


## QuTiP v5.0+ 的稳态求解器


从 QuTiP 5.0 起，可用稳态求解方法包括：

- **direct**：直接 LU 分解。
- **eigen**：计算 $\mathcal{L}\rho$ 零本征值对应本征向量。
- **svd**：SVD 分解求解（仅稠密矩阵）。
- **power**：用逆幂法寻找零本征向量。

其中在 `direct` 与 `power` 方法下，可选如下分解 `solver`：

- **稠密求解器**：来自 `numpy.linalg`。
    - solve：通过 LAPACK `_gesv` 给出精确解。
    - lstsq：最小化 L2 范数的最优最小二乘解。
- **稀疏求解器**：来自 `scipy.sparse.linalg`。
    - spsolve：通过 UMFPACK 的精确解。
    - gmres：GMRES 迭代解。
    - lgmres：LGMRES 迭代解。
    - bicgstab：BICGSTAB 迭代解。
- **MKL 求解器**：由 `mkl` 提供的稀疏求解器。
    - mkl_spsolve：Intel MKL Pardiso 求解。


## 建模与求解


### 系统参数

```python
# System Parameters (in units of wm)
# -----------------------------------
Nc = 4  # Number of cavity states
Nm = 12  # Number of mech states
kappa = 0.3  # Cavity damping rate
E = 0.1  # Driving Amplitude
g0 = 2.4 * kappa  # Coupling strength
Qm = 0.3 * 1e4  # Mech quality factor
gamma = 1 / Qm  # Mech damping rate
n_th = 1  # Mech bath temperature
delta = -0.43  # Detuning
```

### 构造哈密顿量与塌缩算符

```python
# Operators
# ----------
a = tensor(destroy(Nc), qeye(Nm))
b = tensor(qeye(Nc), destroy(Nm))
num_b = b.dag() * b
num_a = a.dag() * a

# Hamiltonian
# ------------
H = -delta * (num_a) + num_b + g0 * (b.dag() + b) * num_a + E * (a.dag() + a)

# Collapse operators
# -------------------
cc = np.sqrt(kappa) * a
cm = np.sqrt(gamma * (1.0 + n_th)) * b
cp = np.sqrt(gamma * n_th) * b.dag()
c_ops = [cc, cm, cp]
```

### 运行稳态求解器

```python
# all possible methods
possible_methods = ["direct", "eigen", "svd", "power"]

# all possible solvers for direct (and power) method(s)
possible_solvers = [
    "solve",
    "lstsq",
    "spsolve",
    "gmres",
    "lgmres",
    "bicgstab",
    "mkl_spsolve",
]

# method and solvers used here
method = "direct"
solvers = ["spsolve", "gmres"]

mech_dms = []
for solver in solvers:
    if solver in ["gmres", "bicgstab"]:
        precond_options = {
            "permc_spec": "NATURAL",
            "diag_pivot_thresh": 0.1,
            "fill_factor": 100,
            "options": {"ILU_MILU": "smilu_2"},
        }
        solver_options = {
            "use_precond": True,
            "atol": 1e-15,
            "maxiter": int(1e5),
            **precond_options,
        }
        use_rcm = True
    else:
        solver_options = {}
        use_rcm = False

    start = time.time()
    rho_ss = steadystate(
        H,
        c_ops,
        method=method,
        solver=solver,
        use_rcm=use_rcm,
        **solver_options,
    )
    end = time.time()

    print(f"Solver: {solver}, Time: {np.round(end-start, 5)}")
    rho_mech = ptrace(rho_ss, 1)
    mech_dms.append(rho_mech)

rho_mech = mech_dms[0]
mech_dms = [mech_dm.data.as_ndarray() for mech_dm in mech_dms]
```

### 检查解的一致性


可通过比较机械子系统密度矩阵之差中的非零元素数（NNZ）来检查不同求解器的结果是否一致。

```python
for kk in range(len(mech_dms)):
    c = np.where(
            np.abs(mech_dms[kk].flatten() - mech_dms[0].flatten()) > 1e-5
        )[0]
    print(f"#NNZ for k = {kk} : {len(c)}")
```

## 绘制机械振子的 Wigner 函数


已知由于相位扩散，机械振子的密度矩阵在 Fock 基下近似对角。通过 `hinton()` 图可看到对角元素幅值更大，非对角项影响趋于消失。

```python
hinton(rho_mech.data.as_ndarray(), x_basis=[""] * Nm, y_basis=[""] * Nm);
```

不过在分解过程中仍会出现一些很小的非对角项，可用 `plt.spy()` 显示。

```python
plt.spy(rho_mech.data.as_ndarray(), markersize=1);
```

因此，为消除这一误差，我们显式取对角元并重构一个新算符。

```python
diag = rho_mech.diag()
rho_mech2 = qdiags(diag, 0, dims=rho_mech.dims, shape=rho_mech.shape)
hinton(rho_mech2, x_basis=[""] * Nm, y_basis=[""] * Nm);
```

接下来计算振子的 Wigner 函数，并检查是否存在负值区域。

```python
xvec = np.linspace(-20, 20, 256)
W = wigner(rho_mech2, xvec, xvec)
wmap = wigner_cmap(W, shift=-1e-5)
```

```python
fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(xvec, xvec, W, 256, cmap=wmap)
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
plt.colorbar(c, ax=ax);
```

## 关于

```python
about()
```

## 测试

```python
# 通过 mesolve 演化验证得到的稳态
psi0 = tensor(basis(Nc), basis(Nm))
rho0 = psi0 @ psi0.dag()
tlist = np.linspace(0, 1500, 1500)
rho_evolve = mesolve(H, rho0, tlist, c_ops)
rho_final = ptrace(rho_evolve.states[-1], 1)
assert fidelity(rho_mech, rho_final) > 0.99

# 稳态应为对角占优
rho_mat = np.abs(rho_mech.data.to_array())
assert np.all(2 * np.diag(rho_mat) >= np.sum(rho_mat, axis=1))

# Wigner 函数应出现负值（非经典性）
assert np.any(W < 0)
```
