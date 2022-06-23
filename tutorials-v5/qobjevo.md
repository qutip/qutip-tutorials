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

# `QobjEvo`: time-dependent quantum objects
Made by Eric Giguere, updated by Jake Lishman

```python
# Basic setup
import qutip
import numpy as np
size = 4
t = 1.0
a = qutip.destroy(size)
ad = qutip.create(size)
n = qutip.num(size)
I = qutip.qeye(size)
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

It is not recommended to mix formats within one `QobjEvo`, as the available optimisations will typically be reduced.

```python
constant_form = qutip.QobjEvo([n])
```

<!-- #region -->
### Function time dependence

This should be a valid Python function with the signature
```python
(t: float, args: dict) -> complex
```
where `t` is the time, `args` is a dictionary containing arguments which you can change without needing a new `QobjEvo`, and the return value is the complex value of $f_k$.  We'll look more at `args` later.
<!-- #endregion -->

```python
def cos_t(t, args):
    return np.cos(t)

function_form = qutip.QobjEvo([n, [a+ad, cos_t]])
```

If you need something more complex, such as a state with memory or to build a parametrised set of functions where the arguments will not change once set, you can use a class which implements `__call__`.

```python
class callable_time_dependence:
    def __init__(self, add):
        self.add = add
    
    def __call__(self, t, args):
        return self.add + np.cos(t)
    
callable_form = qutip.QobjEvo([n, [a+ad, callable_time_dependence(2)]])
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

String dependence can be compiled down to C code if Cython is available (or to pure Python if it isn't).  Compiling to C takes a comparatively long time, but if you plan to reuse the same time dependence a lot, this will produce the fastest code.

```python
string_form = qutip.QobjEvo([n, [a+ad, "cos(t)"]])
```

### Array dependence

If the time dependence is particularly costly to compute, you may pass an array containing the value of the functions evaluated at different times, and separately pass the corresponding list of times to the `tlist` keyword argument of the `QobjEvo` constructor.  All times inbetween will be interpolated with cubic splines.

The times in `tlist` must be sorted, but they don't need to be evenly distributed.  If you use this for more than one entry, all entries must have the same values for `tlist` (you can only pass one).

```python
tlist = np.linspace(0, 10, 101)
values = np.cos(tlist)
array_form = qutip.QobjEvo([n, [a+ad, values]], tlist=tlist)
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

## Compilation


If you have Cython available, `QobjEvo` can compile itself down to C code for speed.  This will be most successful with the string format of time dependence.  The compilation process is likely to be slow, but afterwards the `QobjEvo` should return values significantly faster.

Compilation is done with the `QobjEvo.compile()` method, and modifies the object in-place.  Calling the same method again will not force a recompilation.  This is important for re-using compiled `QobjEvo` objects - the solvers will not need to recompile each time the same object is passed.

```python
string_form.compile()
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
    return t + args['delta']

td_args = qutip.QobjEvo([I, coeff_with_args], args={'delta': 1.}) 
td_args(2)
```

```python
# Temporarily overriding the arguments.
td_args(2, args={"delta": 10})
```

```python
# A subsequent regular call will still use the args given at initialisation.
td_args(2)
```

### String form

```python
td_args_str = qutip.QobjEvo([I, "t + delta"], args={"delta": 1.}) 
td_args_str(2)
```

```python
td_args_str(2, args={"delta": 10})
```

### Using objects

The argument value need not just be a number.  Even Cython-compiled strings can accept functions which Cython can call natively, such as the core `numpy` functions.

```python
td_args_str = qutip.QobjEvo([I, "f(t)"], args={'f': np.cos}) 
td_args_str.compile()
td_args_str(0.)
```

```python
td_args_str(np.pi)
```

### Dynamic arguments

When `QobjEvo` is used in the solvers, certain dynamic arguments will be populated at each iteration, if and only if their names are present in the `args` dictionary used at `QobjEvo` intialisation.  The initial values of all of these dynamic arguments will be a representation of `0` in the corresponding type, for example `"state"` will be `qzero()` of the correct dimensions.

There several of these "magic" variables, mostly revolving around the state currently being evolved:
  - `"state"` or `"state_qobj"`: a `Qobj` of the current state.
  - `"state_mat"`: a dense 2D `np.ndarray` of the state as a matrix, similar to `state.full()`
  - `"state_vec"`: a dense 1D `np.ndarray` of the state as a vector.  This only generally makes sense for kets.
  - `"expect_op_<n>"`: `complex`, where `<n>` is an index into `e_ops`, the current expectation value of `e_ops[n]` (the `<>` should not appear, e.g. `"expect_op_0"`).
  - `"collapse"`: a `list` of `(t: float, n: int)` indicating the time `t` a collapse occurred, and which of `c_ops` caused it.  Only present when using `mcsolve`.


```python
args = {"state": None}

def print_args(t, args):
    print("\n".join([
        '"' + key + '":\n' + repr(value)
        for key, value in args.items()
    ]))
    return t

td_args = qutip.QobjEvo([I, print_args], args=args) 
td_args(0)
```

```python
# The `state` keyword argument is typically unused.
# Here it just simulates being inside a solver at a particular state.
td_args(0, state=qutip.basis(4,2))
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
