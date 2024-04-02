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

# <code>QobjEvo</code>: time-dependent quantum objects
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

## Contents

- [Motivation](#Motivation)
- [Instantiation](#Instantiation)
    * [Functional time dependence](#Function-time-dependence)
    * [String time dependence](#String-time-dependence)
    * [Array dependence](#Array-dependence)
- [Evaluation](#Evaluation)
- [Compilation](#Compilation)
- [Arguments](#Arguments)
    * [Function form](#Function-form)
    * [String form](#String-form)
    * [Using objects](#Using-objects)
    * [Dynamic arguments](#Dynamic-arguments)
- [Mathematics](#Mathematics)
- [Superoperators](#Superoperators)


## Motivation
A `Qobj` is a generic quantum object, but it only represents constant objects.  When we need to represent time-dependent objects, the relevant functions take a list of `(Qobj, <time-dependence>)` tuples, where the time dependence is a function, a string or an array.  Inside the function, this is then converted into a new, time-dependent class `QobjEvo`.  We then use this for several compilation and optimisation steps to ensure that the solvers run quickly.

You do not _need_ to use `QobjEvo`, but if you are planning to reuse some large operators for a long-running integration, you can instantiate one yourself and manually call the C compilation methods.  This will have a one-off start-up cost of a couple of seconds, but will make all uses of it faster afterwards.  This trade-off is often worthwhile if the integration will take over around 15 minutes, even if you do not reuse the object.


## Instantiation

`QobjEvo` is instantiated with the same time-dependent list that is passed to, say, `mesolve`.  This doesn't even _have_ to be time dependent, but constant ones will not have significant benefits over varying ones.  The three types of scalar time dependence are:
  - function
  - string
  - array

This can represent objects of the form
$$
A(t) = \sum_k f_k(t) A_k
$$
where the $f_k(t)$ are time-dependent scalars, and the $A_k$ are constant `Qobj` objects.  The list then looks like
```
[A0, [A1, f1], [A2, f2], ...]
```
where all the `Ak` are constant `Qobj`s, and the `fk` are time dependences in one of the available forms.

Alternatively, `QobjEvo` can be created by multiplication of `Qobj` with wrapped time dependences:
```
A0 + A1 * qutip.coefficient(f1, ...) + ...
```

```python
constant_form = qutip.QobjEvo([n])
```

<!-- #region -->
### Function time dependence

This should be a valid Python function with the signature
```python
(t: float, ...) -> complex
```
where `t` is the time. Additional arguments that can be changed without needing a new `QobjEvo` can be added. The return value is the complex value of $f_k$.  We'll look more at extra arguments later.
<!-- #endregion -->

```python
def cos_t(t):
    return np.cos(t)


function_form = n + (a + ad) * qutip.coefficient(cos_t)
```

If you need something more complex, such as a state with memory or to build a parametrised set of functions where the arguments will not change once set, you can use a class which implements `__call__`.

```python
class callable_time_dependence:
    def __init__(self, add):
        self.add = add

    def __call__(self, t, args):
        return self.add + np.cos(t)


callable_form = qutip.QobjEvo([n, [a + ad, callable_time_dependence(2)]])
```

### String time dependence

This should be a valid single Python expression that evaluates to a `complex`.  Roughly, if you could do `eval(x)` and get a valid `complex`, you will be fine.  In addition to `t` being the time, the following symbols are also defined in scope with their usual definitions:
```
sin  cos  tan   asin  acos  atan
sinh cosh tanh  asinh acosh atanh  
exp  log  log10 erf   zerf  sqrt  
real imag conj  abs   norm  arg
proj pi
```
In addition, `np` refers to `numpy` and `spe` to `scipy.special`.

```python
string_form = qutip.QobjEvo([n, [a + ad, "cos(t)"]])
```

### Array dependence

If the time dependence is particularly costly to compute, you may pass an array containing the value of the functions evaluated at different times, and separately pass the corresponding list of times to the `tlist` keyword argument of the `QobjEvo` constructor.  All times inbetween will be interpolated with cubic splines.

The times in `tlist` must be sorted, but they don't need to be evenly distributed.  If you use this for more than one entry, all entries must have the same values for `tlist` (you can only pass one).

```python
tlist = np.linspace(0, 10, 101)
values = np.cos(tlist)

array_form = n + (a + ad) * qutip.coefficient(values, tlist=tlist)
```

## Evaluation

No matter what type of time dependence was used, or even if only a constant was created, you can always call your `QobjEvo` instance like a function to get the value at that time.  This will return a `Qobj`.

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

## Arguments

Data can be passed to the coefficient functions or strings using an `args` dictionary.  In the functional form, this is passed as an explicit parameter.  In the string form, `args` acts like additional variable definitions for the scope of the function.  In other words, `'x + 2'` will work and return a value if, for example, `args` is `{'x': 1}`.  
The keys of `args` should always be strings representing valid Python identifiers (variable names), and they should not begin with an underscore (\_).

Values _can_ be any type, but if Cython compilation is used with string dependence, there will be a large performance penalty for using anything other than valid C numeric types (including `complex`), `numpy` arrays, or any object that Cython cannot call natively.

You can pass `args` directly at instantiation, and these will be used in every call, unless you specifically override them using the `args` keyword argument in the call.  The overriding is temporary, and all subsequent calls will use the values given at initialisation.


### Function form

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

### String form

```python
td_args_str = qutip.QobjEvo([Id, "t + delta"], args={"delta": 1.0})
td_args_str(2)
```

```python
td_args_str(2, {"delta": 10})
```

### Multiple arguments

Each term in the `QobjEvo` has its own arguments even if they share the variable name:

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

### Using objects

The argument value need not just be a number. Even strings can accept functions which Cython can call natively, such as the core `numpy` functions.

```python
td_args_str = qutip.QobjEvo([Id, "f(t)"], args={"f": np.cos})
td_args_str(0.0)
```

```python
td_args_str(np.pi)
```

### Dynamic arguments

When `QobjEvo` is used in the solvers, the solver states, derivated values or other internal values can be made available as arguments.

These values are set from method to the solver classes:

Most solver support:

  - `StateFeedback`: state as a `Qobj` or qutip `Data`.
  - `ExpectFeedback`: Expectation value computed from the state.

Additionnally `mcsolve` has `CollapseFeedback` to get the collaspe list and stochastic solvers have `WienerFeedback` that returns the Wiener function along the trajectory.

They all take a `default` input that specifies the value to be used when the `QobjEvo` is accessed outside of a solver. This value must be a valid input for the `QobjEvo` -- it will be used during solver setup.

```python
args = {"state": qutip.MESolver.StateFeedback(default=qutip.fock_dm(4, 2))}


def print_args(t, state):
    print(f"'state':\n{state}")
    return t + state.norm()


td_args = qutip.QobjEvo([Id, print_args], args=args)
td_args(0.5)
```

## Mathematics

`QobjEvo` supports the basic mathematical operations which make sense for time-dependent quantum objects:
  - addition of `QobjEvo` and `Qobj`
  - subtraction of `QobjEvo` and `Qobj`
  - product with `QobjEvo`, `Qobj` or scalars
  - division by a scalar
  - negation: `-x`
  - conjugation: `QobjEvo.conj()`
  - adjoint (dagger): `QobjEvo.dag()`
  - transpose: `QobjEvo.trans`

```python
(array_form * 2)(0)
```

```python
(array_form + 1 + a)(0)
```

## Superoperators

The functions in `qutip.superoperator` can also be used for `QobjEvo`.  Of particular importance is `liouvillian`, as `mesolve` can take the Liouvillian as its first argument (and will be faster this way), in place of separated Hamiltonians and collapse operators.

```python
liouv = qutip.liouvillian(array_form, c_ops=[constant_form])
liouv(0)
```

## Epilogue

```python
qutip.about()
```

```python

```
