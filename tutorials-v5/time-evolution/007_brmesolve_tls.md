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

<!-- #region -->
# Bloch-Redfield Solver: Two Level System

Author: C.Staufenbiel, 2022

with inspirations from the [`brmesolve notebook`](https://github.com/qutip/qutip-notebooks/blob/master/examples/brmesolve.ipynb) by P.D. Nation.


### Introduction

The Bloch-Redfield solver is another method to solve a master equation. In comparison to the Lindblad Master equation solver `qutip.mesolve()` the Bloch-Redfield solver `qutip.brmesolve()` differs in the description of the interaction with the environment. In `qutip.mesolve()` we described the dissipation by collapse operators, which not necessarily have a physical interpretation. The `qutip.brmesolve()` function requires the a dissipation description by the so-called *noise-power-spectrum*, which gives the intensity of the dissipation depending on the frequency $\omega$.

In this notebook we will introduce the basic usage of `qutip.brmesolve()` and compare it to `qutip.mesolve()`. For more information on the Bloch-Redfield solver see the follow-up notebooks and the [QuTiP Documentation of the functionality](https://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html).

### Imports
<!-- #endregion -->

```python
import matplotlib.pyplot as plt
from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np
from qutip import (about, basis, bloch_redfield_tensor, brmesolve, expect,
                   hinton, liouvillian, mesolve, plot_expectation_values,
                   sigmam, sigmax, sigmay, sigmaz, steadystate, anim_hinton)

%matplotlib inline
```




## Two-level system evolution

In this example we consider a simple two level system described by the Hamiltonian:

$$ H = \frac{\epsilon}{2} \sigma_z$$

Furthermore, we define a constant dissipation rate to the environment $\gamma$.

```python
epsilon = 0.5 * 2 * np.pi
gamma = 0.25
times = np.linspace(0, 10, 100)
```

Setup the Hamiltonian, initial state and collapse operators for the `qutip.mesolve()` function. We choose a superposition of states as initial state and want to observe the expectation values of $\sigma_x, \sigma_y, \sigma_z$.

```python
# Setup Hamiltonian and initial state
H = epsilon / 2 * sigmaz()
psi0 = (2 * basis(2, 0) + basis(2, 1)).unit()

# Setup the master equation solver
c_ops = [np.sqrt(gamma) * sigmam()]
e_ops = [sigmax(), sigmay(), sigmaz()]
result_me = mesolve(H, psi0, times, c_ops, e_ops)
```

For the `qutip.brmesolve` function we have to give the interaction of the system with the bath as a hermitian operator together with a noise power spectrum, which defines the strength of the interaction per frequency. Here we define a constant interaction whenever the frequency is positive and no dissipation for negative frequencies. This allows us to use `sigmax()` ( a hermitian operator) instead of the non-hermitian operator `sigmam` used above.

The usage of hermitian operators simplifies the internal numerical implementation and leads to vanishing cross-correlations between different environment operators (if multiple are given).

```python
a_op = [sigmax(), lambda w: gamma * (w > 0.0)]
```

Instead of the `c_ops` we now pass the `a_ops` to the Bloch-Redfield solver.

```python
result_brme = brmesolve(H, psi0, times, [a_op], e_ops)
```

We can now compare the expectation values for every operator we passed to the solvers in `e_ops`. As expected both solvers, `mesolve` and `brmesolve`, produce similar results.

```python
fig, axes = plot_expectation_values(
    [result_me, result_brme], ylabels=["<X>", "<Y>", "<Z>"]
)
for ax in axes:
    ax.legend(['mesolove', 'brmesolve'], loc='upper right')
```

## Storing States instead of expectation values
As for the other solvers provided in QuTiP, we can obtain the density matrices at each defined time step instead of some expectation values. To do so, we pass an empty list as `e_ops` argument. If you want to calculate expectation values (i.e. non-empty `e_ops`) and obtain the states at the same time you can also pass `options={"store_states": True}` to the solver functions.

```python
# run solvers without e_ops
me_s = mesolve(H, psi0, times, c_ops, e_ops=[])
brme_s = brmesolve(H, psi0, times, [a_op], e_ops=[])

# calculate expecation values
x_me = expect(sigmax(), me_s.states)
x_brme = expect(sigmax(), brme_s.states)

# plot the expectation values
plt.plot(times, x_me, label="ME")
plt.plot(times, x_brme, label="BRME")
plt.legend(), plt.xlabel("time"), plt.ylabel("<X>");
```

You can use the `qutip.anim_hinton()` function to visualize the time evolution. The animation shows the state is converging to the steadystate.

```python
fig, ani = anim_hinton(me_s)
plt.close()
ani
```

## Bloch-Redfield Tensor

We described the dynmamics of the system by the Bloch-Redfield master equation, which is constructed from the Bloch-Redfield tensor $R_{abcd}$ (see [documentation of Bloch-Redfield master equation](https://qutip.org/docs/latest/guide/dynamics/dynamics-bloch-redfield.html)). Hence the dynamics are determined by this tensor. We can calculate the tensor in QuTiP using the `qutip.bloch_redfield_tensor()` function. We have to pass the Hamiltonian of the system and the dissipation description in `a_ops` to construct $R_{abcd}$. Furthermore, the function gives us the **eigenstates of the Hamiltonian**, as they are calculated along the way.


```python
R, H_ekets = bloch_redfield_tensor(H, [a_op])

# calculate lindblad liouvillian from H
L = liouvillian(H, c_ops)
```

We can now use the Bloch-Redfield Tensor and the Lindblad Liouvillain to calculate the steadystate for both approaches. As we saw above the dynamics were the same for using the different solvers, hence we expect the steadystate to be equal too. We use the `qutip.hinton()` function to plot the steadystate density matrix for both approaches and can see that they are the same.

We have to transform the steadystate density matrix we obtain from the Bloch-Redfield tensor using the eigenstates of the Hamiltonian, as `R` is expressed in the eigenbasis of `H`.

```python
# Obtain steadystate from Bloch-Redfield Tensor
rhoss_br_eigenbasis = steadystate(R)
rhoss_br = rhoss_br_eigenbasis.transform(H_ekets, True)

# Steadystate from Lindblad liouvillian
rhoss_me = steadystate(L)

# Plot the density matrices using a hinton plot
fig, ax = hinton(rhoss_br)
ax.set_title("Bloch-Redfield steadystate")
fig, ax = hinton(rhoss_me)
ax.set_title("Lindblad-ME steadystate");
```

## About

```python
about()
```

## Testing

```python
# Verify that mesolve and brmesolve generate similar results
assert np.allclose(result_me.expect[0], result_brme.expect[0])
assert np.allclose(result_me.expect[1], result_brme.expect[1])
assert np.allclose(result_me.expect[2], result_brme.expect[2])
assert np.allclose(x_me, x_brme)

# assume steadystate is the same
assert np.allclose(rhoss_br.full(), rhoss_me.full())
```
