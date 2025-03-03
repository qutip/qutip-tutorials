---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Calculation of control fields for symplectic dynamics using L-BFGS-B algorithm


Alexander Pitchford (agp1@aber.ac.uk)


Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.optimize_pulse function.
The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which in this case is given by the
'Trace difference' norm.

This in a Symplectic quantum system example, with two coupled oscillators

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot

This example assumes that the example-control-pulseoptim-Hadamard has already been tried, and hence explanations in that notebook are not repeated here.

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulseoptim as cpo
import qutip_qtrl.symplectic as sympl
from qutip import Qobj, identity, about

example_name = "Symplectic"
%matplotlib inline
```

### Defining the physics

```python
# Drift
w1 = 1
w2 = 1
g1 = 0.5
A0 = Qobj(np.array([[w1, 0, g1, 0], [0, w1, 0, g1], [g1, 0, w2, 0], [0, g1, 0, w2]]))

# Control
Ac = Qobj(
    np.array(
        [
            [
                1,
                0,
                0,
                0,
            ],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
)
ctrls = [Ac]
n_ctrls = len(ctrls)

initial = identity(4)

# Target
a = 1
Ag = np.array([[0, 0, a, 0], [0, 0, 0, a], [a, 0, 0, 0], [0, a, 0, 0]])

Sg = Qobj(sympl.calc_omega(2).dot(Ag)).expm()
```

### Defining the time evolution parameters

```python
# Number of time slots
n_ts = 1000
# Time allowed for the evolution
evo_time = 10
```

### Set the conditions which will cause the pulse optimisation to terminate

```python
# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 30
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
```

### Set the initial pulse type

```python
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = "ZERO"
```

### Give an extension for output files

```python
# Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
```

### Run the optimisation

```python
# Note that this call uses
#    dyn_type='SYMPL'
# This means that matrices that describe the dynamics are assumed to be
# Symplectic, i.e. the propagator can be calculated using
# expm(combined_dynamics.omega*dt)
# This has defaults for:
#    prop_type='FRECHET'
# therefore the propagators and their gradients will be calculated using the
# Frechet method, i.e. an exact gradient
#    fid_type='TRACEDIFF'
# so that the fidelity error, i.e. distance from the target, is give
# by the trace of the difference between the target and evolved operators
result = cpo.optimize_pulse(
    A0,
    ctrls,
    initial,
    Sg,
    n_ts,
    evo_time,
    fid_err_targ=fid_err_targ,
    min_grad=min_grad,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    dyn_type="SYMPL",
    out_file_ext=f_ext,
    init_pulse_type=p_type,
    gen_stats=True,
)
```

### Report the results

```python
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print(
    "Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time))
)
```

### Plot the initial and final amplitudes

```python
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial Control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(
        result.time,
        np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
        where="post",
    )

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Amplitudes")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(
        result.time,
        np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
        where="post",
    )

plt.tight_layout()
plt.show()
```

### Versions

```python
about()
```

```python

```

```python

```
