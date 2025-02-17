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

# Calculation of control fields for Lindbladian dynamics using L-BFGS-B algorithm


Christian Arenz (christianarenz.ca@gmail.com), Alexander Pitchford (alex.pitchford@gmail.com)


Example to demonstrate using the control library to determine control pulses using the ctrlpulseoptim.optimize_pulse function. The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which in this case is given by the 'Trace difference' norm.

This in an open quantum system example, with a single qubit subject to an amplitude damping channel. The target evolution is the Hadamard gate. For a $d$ dimensional quantum system in general we represent the Lindbladian
as a $d^2 \times d^2$ dimensional matrix by creating the Liouvillian superoperator. Here done for the Lindbladian that describes the amplitude damping channel. Similarly the control generators acting on the qubit are also converted to superoperators. The initial and target maps also need to be in superoperator form. 

The user can experiment with the strength of the amplitude damping by changing the gamma variable value. If the rate is sufficiently small then the target fidelity can be achieved within the given tolerence. The drift Hamiltonian and control generators can also be swapped and changed to experiment with controllable and uncontrollable setups.

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot

For more background on the pulse optimisation see:
[QuTiP overview - Optimal Control](http://nbviewer.ipython.org/github/qutip/qutip-notebooks/blob/master/examples/example-optimal-control-overview.ipynb)  

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulseoptim as cpo
from qutip import (
    gates,
    identity,
    liouvillian,
    sigmam,
    sigmax,
    sigmay,
    sigmaz,
    sprepost,
    about,
)

example_name = "Lindblad"

%matplotlib inline
```

### Defining the physics

```python
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Sm = sigmam()
Si = identity(2)
# Hadamard gate
had_gate = gates.hadamard_transform(1)

# Hamiltonian
Del = 0.1  # Tunnelling term
wq = 1.0  # Energy of the 2-level system.
H0 = 0.5 * wq * sigmaz() + 0.5 * Del * sigmax()

# Amplitude damping#
# Damping rate:
gamma = 0.1
L0 = liouvillian(H0, [np.sqrt(gamma) * Sm])

# sigma X control
LC_x = liouvillian(Sx)
# sigma Y control
LC_y = liouvillian(Sy)
# sigma Z control
LC_z = liouvillian(Sz)

# Drift
drift = L0
# Controls - different combinations can be tried
ctrls = [LC_z, LC_x]
# Number of ctrls
n_ctrls = len(ctrls)

# start point for the map evolution
E0 = sprepost(Si, Si)

# target for map evolution
E_targ = sprepost(had_gate, had_gate)
```

### Defining the time evolution parameters

```python
# Number of time slots
n_ts = 10
# Time allowed for the evolution
evo_time = 2
```

### Set the conditions which will cause the pulse optimisation to terminate

```python
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 30
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
```

### Set the initial pulse type

```python
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = "RND"
```

### Give an extension for output files

```python
# Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)
```

### Run the optimisation

```python
# Note that this call will take the defaults
#    dyn_type='GEN_MAT'
# This means that matrices that describe the dynamics are assumed to be
# general, i.e. the propagator can be calculated using:
# expm(combined_dynamics*dt)
#    prop_type='FRECHET'
# and the propagators and their gradients will be calculated using the
# Frechet method, i.e. an exact gradent
#    fid_type='TRACEDIFF'
# and that the fidelity error, i.e. distance from the target, is give
# by the trace of the difference between the target and evolved operators
result = cpo.optimize_pulse(
    drift,
    ctrls,
    E0,
    E_targ,
    n_ts,
    evo_time,
    fid_err_targ=fid_err_targ,
    min_grad=min_grad,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
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
print("Initial fidelity error {}".format(result.initial_fid_err))
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
ax1.set_title("Initial control amps")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(
        result.time,
        np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
        where="post",
    )

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(
        result.time,
        np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
        where="post",
    )
fig1.tight_layout()
```

### Versions

```python
about()
```

```python

```
