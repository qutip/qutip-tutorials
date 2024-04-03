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

# qutip-jax JAX backend for qutip

JAX is a numpy like libraries that can run on the CPU, GPU and TPU and support automatic differentiation.
qutip-jax allows JAX array to be used to store `Qobj`'s data allowing qutip to run on GPU.

This backend will work with all qutip functions, but some may convert data to other format without warning. For example using scipy ODE will conver the state to a numpy array.

Support for `jit` and `grad` with qutip's functions is experimental. When using the right options, it is possible to run `mesolve` and `sesolve` on GPU with compilation and auto-differentiation with them. Many `Qobj` operations are also supported.

```python
import jax
import qutip
import qutip_jax  # noqa: F401
```

The JAX backend is activated by importing the module. 
Then the formats `jax` and `jaxdia` are added to know qutip data types.
- `"jax"` store the data as a dense Jax Array.
- `"jaxdia"` represent sparse arrays in DIAgonal format.

```python
# Creating jax Qobj using the dtype argument
id_jax = qutip.qeye(3, dtype="jax")
id_jax.data_as("JaxArray")
```

```python
# Creating jax Qobj using context
with qutip.CoreOptions(default_dtype="jaxdia"):
    id = qutip.qeye(3)
    a = qutip.destroy(3)

# Creating jax Qobj using manual conversion
sz = qutip.sigmaz().to("jaxdia")
sx = qutip.sigmax().to("jaxdia")

# Once created, most operation will conserve the data format
op = (sz & a) + (sx & id)
op
```

```python
# Many functions will do operation without converting output to numpy
qutip.expect(op, qutip.rand_dm([2, 3], dtype="jax"))
```

`jit` can be used with most linear algebra functions:

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

JAX can be used with `mesolve` and `sesovle` in a way that support `jax.jit` and `jax.grad`, but specific options must be used:
- The ODE solver from diffrax must be used instead of those provided by scipy.
- `normalize_output` must be false
- Coefficient for QobjEvo must be `jitted` function.
- The isherm flag of e_ops must be pre-set.
- The class interface must be used for `jit`
- `e_data` must be used instead of expect for auto-differentiation.
- All operators and states must use `jax` or `jaxdia` format.

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
dfinal_expect_dt(solver, qutip.basis(10, 8, dtype="jax"), 0.1, 1.0)
```

If you are interested in contributing to qutip-jax

```python
qutip.about()
```

```python

```
