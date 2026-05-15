---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Calculating Time Propagators and Evolution with Dysolve

Authors:
- Mathis Beaudoin, 2025
- Eric Giguere, 2026       



The Dyson Solver (`Dysolve`) solves the Schrödinger equation by expanding the propagator in a Dyson series. This approach is specifically designed to handle driven system with high frequencies. Standard solvers (like Runge-Kutta) perform small, discrete time steps, which becomes smaller as the frequency of the drive increases. In contrast, the Dyson solver analytically calculates the evolution over a finite interval (possibly multiple periods), offering a more efficient approach for this class of problems.



### Introduction

This notebook shows how to compute time evolution and propagators with Dysolve using QuTiP. It computes time evolution for Hamiltonians of the form

$$H(t) = H_0 + \sum_i \exp(i \omega_i t)X_i$$ 

where $H_0$ is the base hamiltonian and $X_i$ are perturbations. 
It performs better than other general methods for this class of hamiltonians. 
For more details on Dysolve, see the corresponding guide in the documentation.

Dysolve can be used with the class `Dysolve` and the functions `dysolve` and `dysolve_propagator` from QuTiP.
They follow a similar structure to the function `sesolve` and `propagator`. 


First, we import the necessary packages.

```python
import time

import matplotlib.pyplot as plt
import numpy as np
import qutip
from qutip import basis, destroy, num, qeye, sigmax, sigmay, sigmaz
from qutip.solver.dysolve import Dysolve, dysolve, dysolve_propagator
from qutip.solver.propagator import propagator
from qutip.solver.sesolve import sesolve
```

### One qubit propagator example using `Dysolve`

We first define the base Hamiltonian $H_0$ and the drive: $X$ and $\omega$. Here we study the system:

$$H(t) = \sigma_z + \cos(10t)\sigma_x$$

```python
# Constant part of the Hamiltotinian
H_0 = sigmaz()
# The drive as a tuple (operator, frequency, form)
X = sigmax()
omega = 10.0
drive = (X, omega, "cos")
```

Some options can be defined. `order` specifies the order of approximation used when calculating a propagator. The higher this integer is, the more precise the results will be (at a cost of taking more time to calculate). `a_tol` is simply the absolute tolerance used in the calculations. Finally, a time propagator can be computed using subpropagators of time increment `step_size`. If `step_size` is set to 0.25, then the propagator $U(1, 0)$ will come from the multiplication of the supropagators $U(0.25, 0)$, $U(0.5, 0.25)$, $U(0.75, 0.5)$ and $U(1, 0.75)$. This allows for more precise results when the evolution is over a long period of time. In our case, we keep `a_tol` and `step_size` to their default value, but we change `order` to 5 to achieve a good balance of precision and speed for this example.

```python
options = {"max_order": 5}
```

Everything is now defined to initialize an instance.

```python
dy = Dysolve(H_0, [drive], options=options)
```

Then, to compute a time propagator, simply call the instance with a given initial time and final time. Also, only a final time can be given and, in that case, the initial time is considered to be 0.

```python
t_i = -1
t_f = 1
U = dy.propagator(t_f, t_i)
```

This returns a single time propagator $U(t_f = 1, t_i = -1)$. To verify that the $U$ is correct, we compare it to what `propagator` would return.

```python
# Solve using propagator


def X_coeff(t, omega):
    return np.cos(omega * t)


H = [H_0, [X, X_coeff]]
args = {"omega": omega}
options = {"atol": 1e-10, "rtol": 1e-8}
prop = propagator(H, [t_i, t_f], args=args, options=options)

# Comparison
with qutip.CoreOptions(atol=1e-10, rtol=1e-6):
    print(U == prop[1])
```

The comparison U == prop[1] returns True, indicating that up to the chossen tolerance, Dysolve and the standard propagator yield numerically identical results for the propagator from `t_i` to `t_f`. The `qutip.CoreOptions` are used to set the internal tolerances of the equal operation (atol for absolute tolerance, rtol for relative tolerance) in a similar way to `numpy.allclose`.


The first call to `Dysolve.propagator` can be slow as it pre-compute parts of the propagator for the `step_size`.
After this initial call, any subsequent calls for propagator over time interval that is multiple of that `step_size` will be faster, 
as the pre-computed parts can be reused. Calls for non-multiples of step_size will still incur a performance penalty due to new calculations.

```python
dy = Dysolve(H_0, [drive], options={"step_size": 0.1, "max_order": 5})
%time U = dy.propagator(0.10)  # first call, slow
%time U = dy.propagator(1.00)  # multiple of step_size. fast
%time U = dy.propagator(1.05)  # not a multiple of step_size. slow
```

### Two qubits example with `dysolve_propagator`

We proceed like the previous example.

```python
# Define the system
H_0 = (sigmax() & sigmaz()) + (qeye(2) & sigmay())
X = qeye(2) & sigmaz()
omega = 100.0

# Keep options to default
```

`dysolve_propagator` can take more than one time value. If a single time is passed, a single propagator $U(t,0)$ is returned. If a list of times is given, the function will return a list of propagator $[U(\text{times}[i], \text{times}[0])]$ for all time in `tlist`. 

```python
times = np.linspace(0, 1, 11)
Us = dysolve_propagator(
    H_0, [(X, omega, "sin")], times, {"order": 5, "step_size": 0.05}
)
```

Again, we compare the results with `propagator`.

```python
# Solve using propagator


def X_coeff(t, omega):
    return np.sin(omega * t)


H = [H_0, [X, X_coeff]]
args = {"omega": omega}
props = propagator(H, times, args=args, options={"atol": 1e-10, "rtol": 1e-8})

# Comparison
for i in range(11):
    print((Us[i] - props[i]).norm())
```

In this two-qubit example, we observe small non-zero differences when comparing the propagators from `dysolve_propagator` and `propagator`. This is expected due to the different underlying numerical methods and approximations used by each solver. The errors observed are reasonably small (on the order of $10^{-6}$ to $10^{-5}$), indicating that `dysolve_propagator` still provides results with acceptable precision for many applications.


`Dysolve.run` or `dysolve` are interfaces for state evolution using the dyson serie.

Reusing the last Hamiltonian:

```python
times = np.linspace(0, 1, 51)
psi0 = basis([2, 2], [0, 1])
e_op = sigmaz() & sigmaz()
dy_inst = Dysolve(H_0, [(X, omega, "sin")])
result_dysolve = dysolve(H_0, [(X, omega, "sin")], psi0, times, e_ops=[e_op])
result_dysolve_2 = dy_inst.run(psi0, times, e_ops=[e_op])
result_sesolve = sesolve(H, psi0, times, e_ops=[e_op], args=args)

plt.plot(result_dysolve.times, result_dysolve.expect[0], "+", label="dysolve")
plt.plot(
    result_dysolve_2.times,
    result_dysolve_2.expect[0],
    "--",
    label="Dysolve.run",
)
plt.plot(result_sesolve.times, result_sesolve.expect[0], ":", label="sesolve")
plt.xlabel("time")
plt.ylabel("expect")
plt.legend()
```

The plot above illustrates the evolution of an expectation value (`e_ops=(sigmaz() & sigmaz())`). The overlapping lines demonstrate that `dysolve`, `Dysolve.run`, and `sesolve` produce consistent results for the state evolution when appropriately configured.


## Drives with envelopes

`Dysolve` can also be used to solve more complex Hamiltonian where a slow-moving envelope is added to the drive:

$$H(t) = H_0 + \sum_i \exp(i \omega_i t)X_i * E_i(t)$$ 

With `Dysolve`, the envelope $E_i(t)$ is a function estimated to be constant over the `"step_size"` interval as part of its approximation. 
This is a key consideration when choosing `step_size`.

In QuTiP, the envelopes are `Coefficient` objects and support arguments as in other solver.
They are passed as the fourth value in the drives tuple. 
QuTiP provides the namedtuple `dysolve.Drive`, which is recommended for use instead of raw tuples to improve readability and avoid errors.


To present this feature, we will use a drive with Gaussian envelope centered at `t=0.5`, and another one with controllable strength `A` that can be passed as an argument. 

```python
H_0 = sigmaz() & sigmaz()
X_1 = sigmax() & qeye(2)
X_2 = qeye(2) & sigmax()
omega = 10.0
pulse = qutip.coefficient(lambda t: np.exp(-5 * (t - 0.5) ** 2))
control = qutip.coefficient(lambda t, A: A, args={"A": 0.0})
drive1 = dysolve.Drive(X_1, omega, envelope=pulse)
drive2 = dysolve.Drive(X_2, 0.0, "exp", envelope=control)

options = {"step_size": 0.001, "order": 5}

dy_inst = Dysolve(H_0, [drive1, drive2], options=options)

tlist = np.linspace(0, 1, 101)
e_ops = [(sigmaz() & qeye(2)) + (qeye(2) & sigmaz() * 2)]
psi0 = basis([2, 2], [1, 1])

result_dy = dy_inst.run(psi0, tlist, e_ops=e_ops)


def coeff(t):
    return np.exp(-5 * (t - 0.5) ** 2) * np.cos(omega * t)


result_ref = sesolve([H_0, [X_1, coeff]], psi0, tlist, e_ops=e_ops)

plt.plot(tlist, result_dy.expect[0], label="dysolve")
plt.plot(tlist, result_ref.expect[0], label="sesolve")
plt.xlabel("time")
plt.ylabel("expect")
plt.legend()
```

```python
result_dy = dy_inst.run(psi0, tlist, e_ops=e_ops, args={"A": 1.0})


def coeff(t):
    return np.exp(-5 * (t - 0.5) ** 2) * np.cos(omega * t)


result_ref = sesolve(
    [H_0 + X_2, [X_1, coeff]], psi0, tlist, e_ops=e_ops, args={"A": 1.0}
)

plt.plot(tlist, result_dy.expect[0], label="dysolve")
plt.plot(tlist, result_ref.expect[0], label="sesolve")
plt.xlabel("time")
plt.ylabel("expect")
plt.legend()
```

The first graph shows the evolution without the second perturbation, 
while the second show the evolution with `X_2` active.
Both graph are computed from the same `Dysolve` instance and changing the arguments do not require new pre-computation.
However in the first computation, even if `drive2` is effectively off, it is computed and affect evolution time.


## Comparisons with other solvers

QuTiP has three `...solve` functions to solve the Schrödinger equation: `sesolve`, `fsesolve` and `dysolve`.
Since drives are usually periodic perturbations, there is a large overlap between `fsesolve` and `dysolve` use cases.
Here are some pointers to choose the right solver for the problem at hand.

- `sesolve`: Generic solver, which will work with all systems.
  It works best with slowly varying Hamiltonians and needs to do more work as the frequency increases.
- `fsesolve`: Uses Floquet basis to solve periodic systems more efficiently.
  It only need to performs the evolution over one period to obtain the state at any time.
  Its efficiency is not significantly affected by the drive's frequency, and errors stable over time.
  It can solve any periodic system, even those with messy Fourier transforms.
- `dysolve`: Uses Dyson series to solve driven systems more efficiently.
  It converges faster at higher frequencies and supports non-periodic drives.
  It, however, needs a clean Fourier transform with few terms.


#### Example 1: Hamiltonian with a periodic square pulse

$$H = H_0 + X \cdot ((t \% 1.) < 0.5)$$

This is efficient to solve with `fsesolve`, because the evolution over the period is straightforward.   
But for `dysolve`, this would require the Fourier series to be truncated, resulting in low precision.

#### Example 2: Hamiltonian with a similar frequency drives

$$H = H_0 + X_1 \cdot \sin(100 \pi t) + X_2 \cdot \sin(101 \pi t) $$

This is easy to solve with `dysolve`: 2 high-frequency drives.  
However, for `fsesolve`, while the period of each drive individually is small (`~0.02`), the overall period is `2` and it's fastly moving during that interval making the integration hard.


With the right options, every solver can reach similar precision.

The Hamiltonian:

$$H = 1 + \sigma_x cos(\omega * t)$$

can be solved analytically to 

$$U = e^{-it} * \exp({-i\sigma_x sin(\omega t)/\omega})$$

We will solve this system with all 3 methods and compare computation time and numerical error.

```python
H0 = qeye(2)
X = sigmax()
w = 100.0
psi0 = basis(2, 1)
tlist = np.linspace(0, 10, 11)
H = H0 + X * qutip.coefficient(lambda t: np.cos(w * t))

ref = [np.exp(-1j * t) * (-1j / w * np.sin(w * t) * X).expm() @ psi0 for t in tlist]
start = time.perf_counter()
dy_res = dysolve(
    H0, [(X, w, "cos")], psi0, tlist, options={"order": 4, "step_size": 0.1}
).states
print(f"dysolve: {time.perf_counter() - start}")

start = time.perf_counter()
se_res = sesolve(H, psi0, tlist).states
print(f"sesolve, (default option): {time.perf_counter() - start}")

options = {
    "max_step": 0.001,
    "rtol": 1e-12,
    "atol": 1e-13,
    "nsteps": 10000,
}
start = time.perf_counter()
se_res_high_pre = sesolve(H, psi0, tlist, options=options).states
print(f"sesolve, (high precision): {time.perf_counter() - start}")

start = time.perf_counter()
fl_res = qutip.fsesolve(H, psi0, tlist, T=(2 * np.pi / w)).states
print(f"fsesolve: {time.perf_counter() - start}")

plt.semilogy(
    tlist[1:],
    [(r - s).norm() for r, s in zip(ref, dy_res)][1:],
    label="dysolve",
)
plt.semilogy(
    tlist[1:],
    [(r - s).norm() for r, s in zip(ref, se_res)][1:],
    label="sesolve, (default option)",
)
plt.semilogy(
    tlist[1:],
    [(r - s).norm() for r, s in zip(ref, se_res_high_pre)][1:],
    label="sesolve, (high precision)",
)
plt.semilogy(
    tlist[1:],
    [(r - s).norm() for r, s in zip(ref, fl_res)][1:],
    label="fsesolve",
)
plt.xlabel("time")
plt.ylabel("Numerical error")
plt.legend()
```

This plot provides a quantitative comparison of the numerical error for `dysolve`, `sesolve` (with default and high precision options), and `fsesolve` against an analytical solution. We can observe:

*   **`fsesolve`** Is the fastest, but has the highest numerical error.
*   **`dysolve`** performs well, showing good accuracy, especially given its advantages for high-frequency and non-periodic drives.
*   **`sesolve`** with default options shows higher error, highlighting its limitations for such driven systems. However, with `high_precision` options (e.g., smaller `max_step`, higher `nsteps`, and tighter `rtol`/`atol`), `sesolve` can achieve comparable accuracy to `dysolve`, but at a much higher computational cost.


## Dysolve method convergence

`Dysolve` precision depend on multiple factors, the principals are 
- `order`
- frequency
- options `"eigen"` and `"polar"`.

In this section, we will show how each affect the numerical error.

Let's first look at the `order` and frequency impact. `Dysolve` converges faster at higher frequencies.

```python
H_0 = sigmaz() & sigmaz()
X = (sigmax() & sigmax()) + (sigmaz() & sigmay())

for W in [0, 3, 10, 30, 100, 300]:
    drive = dysolve.Drive(X, W)
    U_expected = dysolve_propagator(
        H_0, [drive], 1, options={"order": 8, "step_size": 0.1}
    )
    err = []
    orders = [1, 2, 3, 4, 5, 6]
    for order in orders:
        U = dysolve_propagator(
            H_0, [drive], 1, options={"order": order, "step_size": 0.1}
        )
        err.append((U_expected - U).norm())
    plt.semilogy(orders, err, label=f"{W=}")
plt.xlabel("Order")
plt.ylabel("Error")
plt.legend()
```

We see that as the frequency increase, the error decrease for the same `order`.


#### options "eigen"

The computation of the dyson series require $H_0$ to be diagonal. QuTiP's `Dysolve` offers two ways to diagonalize it:
- `{"eigen": True}`: Computes the eigen decomposition and changes the basis of all operators.
- `{"eigen": False}`: Extracts the diagonal part of $H_0$ and add the non-diagonal part as a drive with frequency `0`.

As seen earlier, a drive with a frequency of `0` converges slowly. Therefore `{"eigen": False}` has greater numerical error.
However the computation of the dyson series is faster for sparses perturbations operators.
The change of basis from the eigen decomposition usually leaves the drive operators dense, resulting in slower computation.

#### options "polar"
With the options `{"polar": True}`, `scipy.linalg.polar` is used to ensure the propagator is unitary. This improves the precision when using low odd-order expansions.


To present the impact, we solve a non-diagonal systems with all combinaisons of these two options.
We show both computation time and numerical error per order.

We can obserse that `eigen=True` result in numerical error orders of magnitude smaller that when disabled, but with about twice the computation time.  
`polar` has much smaller impact on both the error and the timing, but can be important when using `order=1`.

```python
# Non-diagonal Hamiltonian
N = 6
H_0 = num(N) + destroy(N) ** 2 + destroy(N).dag() ** 2
X = destroy(N) + destroy(N).dag()
drive = dysolve.Drive(X, W)
U_expected = dysolve_propagator(
    H_0, [drive], 1, options={"order": 7, "step_size": 0.01}
)
orders = [1, 2, 3, 4, 5, 6]

for polar in [True, False]:
    for eigen in [True, False]:
        err = []
        start = time.perf_counter()
        for order in orders:
            U = dysolve_propagator(
                H_0,
                [drive],
                1,
                options={"order": order, "polar": polar, "eigen": eigen},
            )
            err.append((U_expected - U).norm())
        print(
            f"Computation with {polar=} and {eigen=} "
            f"took {time.perf_counter() - start} sec"
        )
        plt.semilogy(orders, err, label=f"{polar=}, {eigen=}")

plt.xlabel("order")
plt.ylabel("Error")
plt.legend()
```

As a last example, we will show the impact of `"step_size"`.

As expected, smaller steps result in better precision, but `step_size` generally has less impact than other parameters for fixed time evolution, especially when not using envelopes.

**`"step_size"` becomes very important when envelopes are used, as it dictates the interval over which the envelope is assumed constant.**

```python
H_0 = sigmaz() & sigmaz()
X = (sigmax() & sigmax()) + (sigmaz() & sigmay())
drive = dysolve.Drive(X, W)
U_expected = dysolve_propagator(
    H_0, [drive], 1, options={"order": 7, "step_size": 0.00001}
)

step_sizes = np.logspace(-5, 0, 16)
orders = [1, 2, 3, 4]
for order in orders:
    err = []
    for step_size in step_sizes:
        U = dysolve_propagator(
            H_0, [drive], 1, options={"order": order, "step_size": step_size}
        )
        err.append((U_expected - U).norm())
    plt.loglog(step_sizes, err, label=f"{order=}")
plt.xlabel("step_size")
plt.ylabel("Error")
plt.legend()
```

### About

```python
qutip.about()
```
