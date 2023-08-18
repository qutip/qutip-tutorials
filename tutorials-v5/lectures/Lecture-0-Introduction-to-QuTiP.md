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

# Lecture 0 - Introduction to QuTiP

Author: J. R. Johansson (robert@riken.jp), https://jrjohansson.github.io/

This lecture series was developed by J.R. Johannson. The original lecture notebooks are available [here](https://github.com/jrjohansson/qutip-lectures).

This is a slightly modified version of the lectures, to work with the current release of QuTiP. You can find these lectures as a part of the [qutip-tutorials repository](https://github.com/qutip/qutip-tutorials). This lecture and other tutorial notebooks are indexed at the [QuTiP Tutorial webpage](https://qutip.org/tutorials.html).

```python
import matplotlib.pyplot as plt
from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np
from IPython.display import Image
from qutip import (Qobj, about, basis, coherent, coherent_dm, create, destroy,
                   expect, fock, fock_dm, mesolve, qeye, sigmax, sigmay,
                   sigmaz, tensor, thermal_dm)

%matplotlib inline
```

## Introduction

QuTiP is a python package for calculations and numerical simulations of quantum systems.

It includes facilities for representing and doing calculations with quantum objects such state vectors (wavefunctions), as bras/kets/density matrices, quantum operators of single and composite systems, and superoperators (useful for defining master equations).

It also includes solvers for a time-evolution of quantum systems, according to: Schrodinger equation, von Neuman equation, master equations, Floquet formalism, Monte-Carlo quantum trajectors, experimental implementations of the stochastic Schrodinger/master equations.

For more information see the project web site at [qutip.org](https://qutip.org), and the
[QuTiP documentation](https://qutip.org/docs/latest/index.html).

### Installation

You can install QuTiP directly from `pip` by running:

`pip install qutip`

For further installation details, refer to the [GitHub repository](https://github.com/qutip/qutip).


## Quantum object class: `qobj`

At the heart of the QuTiP package is the `Qobj` class, which is used for representing quantum object such as states and operator.

The `Qobj` class contains all the information required to describe a quantum system, such as its matrix representation, composite structure and dimensionality.

```python
Image(filename="images/qobj.png")
```

### Creating and inspecting quantum objects


We can create a new quantum object using the `Qobj` class constructor, like this:

```python
q = Qobj([[1], [0]])

q
```

Here we passed python list as an argument to the class constructor. The data in this list is used to construct the matrix representation of the quantum objects, and the other properties of the quantum object is by default computed from the same data.

We can inspect the properties of a `Qobj` instance using the following class method:

```python
# the dimension, or composite Hilbert state space structure
q.dims
```

```python
# the shape of the matrix data representation
q.shape
```

```python
# the matrix data itself. in sparse matrix format.
q.data
```

```python
# get the dense matrix representation
q.full()
```

```python
# some additional properties
q.isherm, q.type
```

### Using `Qobj` instances for calculations

With `Qobj` instances we can do arithmetic and apply a number of different operations using class methods:

```python
sy = Qobj([[0, -1j], [1j, 0]])  # the sigma-y Pauli operator

sy
```

```python
sz = Qobj([[1, 0], [0, -1]])  # the sigma-z Pauli operator

sz
```

```python
# some arithmetic with quantum objects

H = 1.0 * sz + 0.1 * sy

print("Qubit Hamiltonian = \n")
H
```

Example of modifying quantum objects using the `Qobj` methods:

```python
# The hermitian conjugate
sy.dag()
```

```python
# The trace
H.tr()
```

```python
# Eigen energies
H.eigenenergies()
```

For a complete list of methods and properties of the `Qobj` class, see the [QuTiP documentation](https://qutip.org/docs/latest/index.html) or try `help(Qobj)` or `dir(Qobj)`.


## States and operators

Normally we do not need to create `Qobj` instances from stratch, using its constructor and passing its matrix represantation as argument. Instead we can use functions in QuTiP that generates common states and operators for us. Here are some examples of built-in state functions:

### State vectors

```python
# Fundamental basis states (Fock states of oscillator modes)

N = 2  # number of states in the Hilbert space
n = 1  # the state that will be occupied

basis(N, n)  # equivalent to fock(N, n)
```

```python
fock(4, 2)  # another example
```

```python
# a coherent state
coherent(N=10, alpha=1.0)
```

### Density matrices

```python
# a fock state as density matrix
fock_dm(5, 2)  # 5 = hilbert space size, 2 = state that is occupied
```

```python
# coherent state as density matrix
coherent_dm(N=8, alpha=1.0)
```

```python
# thermal state
n = 1  # average number of thermal photons
thermal_dm(8, n)
```

### Operators


#### Qubit (two-level system) operators

```python
# Pauli sigma x
sigmax()
```

```python
# Pauli sigma y
sigmay()
```

```python
# Pauli sigma z
sigmaz()
```

#### Harmonic oscillator operators

```python
#  annihilation operator

destroy(N=8)  # N = number of fock states included in the Hilbert space
```

```python
# creation operator

create(N=8)  # equivalent to destroy(5).dag()
```

```python
# the position operator is easily constructed from the annihilation operator
a = destroy(8)

x = a + a.dag()

x
```

#### Using `Qobj` instances we can check some well known commutation relations:

```python
def commutator(op1, op2):
    return op1 * op2 - op2 * op1
```

$[a, a^1] = 1$

```python
a = destroy(5)

commutator(a, a.dag())
```

**Ops...** The result is not identity! Why? Because we have truncated the Hilbert space. But that's OK as long as the highest Fock state isn't involved in the dynamics in our truncated Hilbert space. If it is, the approximation that the truncation introduces might be a problem.


$[x,p] = i$

```python
x = (a + a.dag()) / np.sqrt(2)
p = -1j * (a - a.dag()) / np.sqrt(2)
```

```python
commutator(x, p)
```

Same issue with the truncated Hilbert space, but otherwise OK.


Let's try some Pauli spin inequalities

$[\sigma_x, \sigma_y] = 2i \sigma_z$

```python
commutator(sigmax(), sigmay()) - 2j * sigmaz()
```

$-i \sigma_x \sigma_y \sigma_z = \mathbf{1}$

```python
-1j * sigmax() * sigmay() * sigmaz()
```

$\sigma_x^2 = \sigma_y^2 = \sigma_z^2 = \mathbf{1}$

```python
sigmax() ** 2 == sigmay() ** 2 == sigmaz() ** 2 == qeye(2)
```

## Composite systems

In most cases we are interested in coupled quantum systems, for example coupled qubits, a qubit coupled to a cavity (oscillator mode), etc.

To define states and operators for such systems in QuTiP, we use the `tensor` function to create `Qobj` instances for the composite system.

For example, consider a system composed of two qubits. If we want to create a Pauli $\sigma_z$ operator that acts on the first qubit and leaves the second qubit unaffected (i.e., the operator $\sigma_z \otimes \mathbf{1}$), we would do:

```python
sz1 = tensor(sigmaz(), qeye(2))

sz1
```

We can easily verify that this two-qubit operator does indeed have the desired properties:

```python
psi1 = tensor(basis(N, 1), basis(N, 0))  # excited first qubit
psi2 = tensor(basis(N, 0), basis(N, 1))  # excited second qubit
```

```python
# this should not be true,
# because sz1 should flip the sign of the excited state of psi1
sz1 * psi1 == psi1
```

```python
# this should be true, because sz1 should leave psi2 unaffected
sz1 * psi2 == psi2
```

Above we used the `qeye(N)` function, which generates the identity operator with `N` quantum states. If we want to do the same thing for the second qubit we can do:

```python
sz2 = tensor(qeye(2), sigmaz())

sz2
```

Note the order of the argument to the `tensor` function, and the correspondingly different matrix representation of the two operators `sz1` and `sz2`.

Using the same method we can create coupling terms of the form $\sigma_x \otimes \sigma_x$:

```python
tensor(sigmax(), sigmax())
```

Now we are ready to create a `Qobj` representation of a coupled two-qubit Hamiltonian: $H = \epsilon_1 \sigma_z^{(1)} + \epsilon_2 \sigma_z^{(2)} + g \sigma_x^{(1)}\sigma_x^{(2)}$

```python
epsilon = [1.0, 1.0]
g = 0.1

sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

H = epsilon[0] * sz1 + epsilon[1] * sz2 + g * tensor(sigmax(), sigmax())

H
```

To create composite systems of different types, all we need to do is to change the operators that we pass to the `tensor` function (which can take an arbitrary number of operator for composite systems with many components).

For example, the Jaynes-Cumming Hamiltonian for a qubit-cavity system:

$H = \omega_c a^\dagger a - \frac{1}{2}\omega_a \sigma_z + g (a \sigma_+ + a^\dagger \sigma_-)$

```python
wc = 1.0  # cavity frequency
wa = 1.0  # qubit/atom frenqency
g = 0.1  # coupling strength

# cavity mode operator
a = tensor(destroy(5), qeye(2))

# qubit/atom operators
sz = tensor(qeye(5), sigmaz())  # sigma-z operator
sm = tensor(qeye(5), destroy(2))  # sigma-minus operator

# the Jaynes-Cumming Hamiltonian
H = wc * a.dag() * a - 0.5 * wa * sz + g * (a * sm.dag() + a.dag() * sm)

H
```

Note that

$a \sigma_+ = (a \otimes \mathbf{1}) (\mathbf{1} \otimes \sigma_+)$

so the following two are identical:

```python
a = tensor(destroy(3), qeye(2))
sp = tensor(qeye(3), create(2))

a * sp
```

```python
tensor(destroy(3), create(2))
```

## Unitary dynamics

Unitary evolution of a quantum system in QuTiP can be calculated with the `mesolve` function.

`mesolve` is short for Master-eqaution solve (for dissipative dynamics), but if no collapse operators (which describe the dissipation) are given to the solve it falls back on the unitary evolution of the Schrodinger (for initial states in state vector for) or the von Neuman equation (for initial states in density matrix form).

The evolution solvers in QuTiP returns a class of type `Odedata`, which contains the solution to the problem posed to the evolution solver.

For example, considor a qubit with Hamiltonian $H = \sigma_x$ and initial state $\left|1\right>$ (in the sigma-z basis): Its evolution can be calculated as follows:

```python
# Hamiltonian
H = sigmax()

# initial state
psi0 = basis(2, 0)

# list of times for which the solver should store the state vector
tlist = np.linspace(0, 10, 100)

result = mesolve(H, psi0, tlist, [], [])
```

```python
result
```

The `result` object contains a list of the wavefunctions at the times requested with the `tlist` array.

```python
len(result.states)
```

```python
result.states[-1]  # the finial state
```

You can visualize the time evolution of the state.

```python
fig, ani = qutip.anim_matrix_histogram(result, limits=[0, 1], bar_style='abs', color_style='phase')
plt.close()
ani
```

### Expectation values

The expectation values of an operator given a state vector or density matrix (or list thereof) can be calculated using the `expect` function.

```python
expect(sigmaz(), result.states[-1])
```

```python
expect(sigmaz(), result.states)
```

```python
fig, axes = plt.subplots(1, 1)

axes.plot(tlist, expect(sigmaz(), result.states))

axes.set_xlabel(r"$t$", fontsize=20)
axes.set_ylabel(r"$\left<\sigma_z\right>$", fontsize=20);
```

If we are only interested in expectation values, we could pass a list of operators to the `mesolve` function that we want expectation values for, and have the solver compute then and store the results in the `Odedata` class instance that it returns.

For example, to request that the solver calculates the expectation values for the operators $\sigma_x, \sigma_y, \sigma_z$:

```python
result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])
```

Now the expectation values are available in `result.expect[0]`, `result.expect[1]`, and `result.expect[2]`:

```python
fig, axes = plt.subplots(1, 1)

axes.plot(tlist, result.expect[2], label=r"$\left<\sigma_z\right>$")
axes.plot(tlist, result.expect[1], label=r"$\left<\sigma_y\right>$")
axes.plot(tlist, result.expect[0], label=r"$\left<\sigma_x\right>$")

axes.set_xlabel(r"$t$", fontsize=20)
axes.legend(loc=2);
```

## Dissipative dynamics

To add dissipation to a problem, all we need to do is to define a list of collapse operators to the call to the `mesolve` solver.

A collapse operator is an operator that describes how the system is interacting with its environment.

For example, consider a quantum harmonic oscillator with Hamiltonian

$H = \hbar\omega a^\dagger a$

and which loses photons to its environment with a relaxation rate $\kappa$. The collapse operator that describes this process is

$\sqrt{\kappa} a$

since $a$ is the photon annihilation operator of the oscillator.

To program this problem in QuTiP:

```python
w = 1.0  # oscillator frequency
kappa = 0.1  # relaxation rate
a = destroy(10)  # oscillator annihilation operator
rho0 = fock_dm(10, 5)  # initial state, fock state with 5 photons
H = w * a.dag() * a  # Hamiltonian

# A list of collapse operators
c_ops = [np.sqrt(kappa) * a]
```

```python
tlist = np.linspace(0, 50, 100)

# request that the solver return the expectation value
# of the photon number state operator a.dag() * a
result = mesolve(H, rho0, tlist, c_ops, [a.dag() * a])
```

```python
fig, axes = plt.subplots(1, 1)
axes.plot(tlist, result.expect[0])
axes.set_xlabel(r"$t$", fontsize=20)
axes.set_ylabel(r"Photon number", fontsize=16);
```

### Installation information

```python
about()
```
