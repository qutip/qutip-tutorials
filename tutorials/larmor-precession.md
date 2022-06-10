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

# Schrödinger equation solver: Larmor precession

Author: C. Staufenbiel, 2022

### Introduction

This notebook guides you through the process of setting up a Schrödinger 
equation in QuTiP and using the corresponding solver to obtain the time 
evolution. We will investigate the example of the Larmor precession to 
explore the functionality of [`qutip.sesolve()`](https://qutip.org/docs/latest/apidoc/functions.html?highlight=sesolve#module-qutip.sesolve).

You can also find more on time evolutions with QuTiP [here](https://qutip.org/docs/latest/guide/guide-dynamics.html).

### Setup

First thing is to import the required functions, classes and modules.
```python
%matplotlib inline
from qutip import Qobj, sesolve, basis, Bloch, sigmaz, sigmay, QobjEvo
import qutip
import matplotlib.pyplot as plt
import numpy as np
```

We setup a arbitrary qubit state, which is in a superposition of the two qubit states. We use the `qutip.Bloch` class to visualize the state on the Bloch sphere.

```python
psi = (2.0 * basis(2, 0) + basis(2, 1)).unit()
b = Bloch()
b.add_states(psi)
b.show()
```

### Simulation with constant magnetic field

Let's define a simple Hamiltonian and use `qutip.sesolve` to solve the
Schrödinger equation. The Hamiltonian describes a constant magnetic field 
along the z-axis. We can describe this magnetic field by the corresponding 
Pauli matrix, which is defined as `qutip.sigmaz()` in QuTiP.

To solve the Schrödinger equation for this particular Hamiltonian, we have to pass the Hamiltonian, the initial state, the times for which we want to simulate the system, and a set of observables that we evaluate at these times.

Here, we are for example interested in the time evolution of the expectation value for $\sigma_y$. We pass these properties to `sesolve` in the following.

```python
# simulate the unitary dynamics
H = sigmaz()
times = np.linspace(0, 10, 100)
result = sesolve(H, psi, times, [sigmay()])
```

`result.expect` holds the expecation values for the times that we passed to `sesolve`. `result.expect` is a two dimensional array, where the first dimension refers to the different expectation operators that we passed to `sesolve` before. 

Above we passed `sigmay()` as the only expectation operator and therefore we can access its values by `result.expect[0]`. Below we plot the evolution of the expecation value.

```python
plt.plot(times, result.expect[0])
plt.xlabel('Time'), plt.ylabel('<sigma_y>')
plt.show()
```

Above we gave `sigmay()` as an operator to `sesolve` to directly calculate it's expectation value. If we pass an empty list at this argument to `sesolve` it will return the quantum state of the system for each time step in `times`. We can access the states by `result.states` and use them to calculate the expectation value manually.

```python
res = sesolve(H, psi, times, [])
exp_y = qutip.expect(sigmay(), res.states)
plt.plot(times, exp_y)
plt.xlabel('Time'), plt.ylabel('<sigma_y>')
plt.show()
```

## Simulation with varying magnetic field

Above we passed a constant Hamiltonian to `sesolve`. In QuTiP these constant operators are represented by `Qobj`. However, `sesolve` can also take time-dependent operators as an argument, which are represented by [`QobjEvo`](https://qutip.org/docs/latest/apidoc/classes.html?highlight=qobjevo#qutip.QobjEvo) in QuTiP. In this section we define the magnetic field with a linear and a periodic field strength, and observe the changes in the expecation value of $\sigma_y$.
You can find more information on `QobjEvo` in [this notebook](https://nbviewer.ipython.org/github/qutip/qutip-notebooks/blob/master/examples/qobjevo.ipynb).

We start by defining two functions for the field strength of the magnetic field. To be passed on to `QobjEvo` the functions need two arguments: the times and optional arguments.


```python
def linear(t, args):
    return 0.3*t

def periodic(t, args):
    return np.cos(0.5*t)

# Define QobjEvos
H_lin = QobjEvo([[sigmaz(), linear]], tlist=times)
H_per = QobjEvo([[sigmaz(), periodic]], tlist=times)
```

We can now continue as in the previous section and use `sesolve` to solve the Schrödinger equation.

```python
result_lin = sesolve(H_lin, psi, times, [sigmay()])
result_per = sesolve(H_per, psi, times, [sigmay()])


# Plot <sigma_y> for linear increasing field strength
plt.plot(times, result_lin.expect[0])
plt.xlabel('Time'), plt.ylabel('<sigma_y>')
plt.show()
```

We can see that the frequency of the Larmor precession increases with the time. This is a direct result of the time-dependent Hamiltonian. We can generate the same plot for the periodically varying field strength.

```python
plt.plot(times, result_per.expect[0])
plt.xlabel('Time'), plt.ylabel('<sigma_y>')
plt.show()
```

### Conclusion
We can use `sesolve` to solve unitary time evolutions. This is not only 
limited to constant Hamiltonians, but we can also make use of time-dependent Hamiltonians using `QobjEvo`. 

### About

```python
import qutip

qutip.about()
```

### Testing

This section can include some tests to verify that the expected outputs are
generated within the notebook. We put this section at the end of the notebook,
so it's not interfering with the user experience. Please, define the tests
using `assert`, so that the cell execution fails if a wrong output is generated.

```python
assert np.allclose(result.expect[0][0], 0)
assert np.allclose(result_lin.expect[0][0], 0)
assert np.allclose(result_per.expect[0][0], 0)
assert 1 == 1
```

```python

```
