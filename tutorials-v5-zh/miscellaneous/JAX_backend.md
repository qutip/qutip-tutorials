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

# qutip 的 qutip-jax JAX 后端

JAX 是一个类似 numpy 的库，可运行在 CPU、GPU 和 TPU 上，并支持自动微分。
qutip-jax 允许使用 JAX 数组存储 `Qobj` 数据，从而让 qutip 在 GPU 上运行。

该后端可与大多数 qutip 函数配合使用，但部分函数可能会在无提示下将数据转换为其他格式。
例如，使用 scipy ODE 时，状态会被转换为 numpy 数组。

对 qutip 函数的 `jit` 和 `grad` 支持仍属实验性。
在选项设置正确时，可在 GPU 上运行 `mesolve` 与 `sesolve`，同时启用编译与自动微分。
许多 `Qobj` 运算也已支持。

```python
import jax
import qutip
import qutip_jax  # noqa: F401
```

导入 `qutip_jax` 模块即可激活 JAX 后端。
随后会新增 `jax` 与 `jaxdia` 两种 qutip 数据格式：
- `"jax"`：将数据存储为稠密 JAX 数组。
- `"jaxdia"`：用 DIA（对角）格式表示稀疏数组。

```python
# Creating jax Qobj using the dtype argument
id_jax = qutip.qeye(3, dtype="jax")
id_jax.data_as("JaxArray")
```

```python
# Creating jax Qobj using a context manager
with qutip.CoreOptions(default_dtype="jaxdia"):
    id = qutip.qeye(3)
    a = qutip.destroy(3)

# Creating jax Qobj using manual conversion
sz = qutip.sigmaz().to("jaxdia")
sx = qutip.sigmax().to("jaxdia")

# Once created, most operations will conserve the data format
op = (sz & a) + (sx & id)
op
```

```python
# Many functions will do operations without converting its output to numpy
qutip.expect(op, qutip.rand_dm([2, 3], dtype="jax"))
```

`jit` 可用于多数线性代数函数：

```python
op = qutip.num(3, dtype="jaxdia")
state = qutip.rand_dm(3, dtype="jax")


@jax.jit
def f(op, state):
    return op @ state @ op.dag()


print(f(op, state))
%timeit op @ state @ op.dag()
%timeit f(op, state)
```

JAX 也可配合 `mesolve` 与 `sesolve`，并支持 `jax.jit` 与 `jax.grad`，
但必须满足以下条件：
- 必须使用 diffrax 的 ODE 求解器，而非 scipy 提供的求解器。
- `normalize_output` 必须为 `False`。
- QobjEvo 的系数应为 `jitted` 函数。
- `e_ops` 的 `isherm` 标志需预先设定。
- 若使用 `jit`，应使用类接口。
- 自动微分时应使用 `e_data`，而不是 `expect`。
- 所有算符和态都应使用 `jax` 或 `jaxdia` 格式。

```python
@jax.jit
def fp(t, w):
    return jax.numpy.exp(1j * t * w)


@jax.jit
def fm(t, w):
    return jax.numpy.exp(-1j * t * w)


@jax.jit
def cte(t, A):
    return A


with qutip.CoreOptions(default_dtype="jax"):
    H = qutip.num(10)
    c_ops = [qutip.QobjEvo([qutip.destroy(10), fm], args={"w": 1.0})]

H.isherm  # Precomputing the `isherm` flag

solver = qutip.MESolver(
    H, c_ops, options={"method": "diffrax", "normalize_output": False}
)


def final_expect(solver, rho0, t, w):
    result = solver.run(rho0, [0, t], args={"w": w}, e_ops=H)
    return result.e_data[0][-1].real


dfinal_expect_dt = jax.jit(
    jax.grad(final_expect, argnums=[2]), static_argnames=["solver"]
)

# TODO: use dfinal_expect_dt instead of final_expect when qutip-jax bug-fix
# dfinal_expect_dt(solver, qutip.basis(10, 8, dtype="jax"), 0.1, 1.0)
jax.grad(final_expect, argnums=[2])(solver, qutip.basis(10, 8, dtype="jax"), 0.1, 1.0)
```


```python
qutip.about()
```

