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

# Calculation of control fields for Hadamard gate on single qubit using L-BFGS-B algorithm


Alexander Pitchford (agp1@aber.ac.uk)


Example to demonstrate using the control library to determine control
pulses using the ctrlpulseoptim.optimize_pulse_unitary function.
The (default) L-BFGS-B algorithm is used to optimise the pulse to
minimise the fidelity error, which is equivalent maximising the fidelity
to an optimal value of 1.

The system in this example is a single qubit in a constant field in z
with a variable control field in x
The target evolution is the Hadamard gate irrespective of global phase

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different initial (starting) pulse types can be tried.
The initial and final pulses are displayed in a plot

An in depth discussion of using methods of this type can be found in [1]

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulseoptim as cpo
from qutip import gates, identity, sigmax, sigmaz, about

example_name = "Hadamard"

%matplotlib inline
```

### Defining the physics


The dynamics of the system are governed by the combined Hamiltonian:
H(t) = H_d + sum(u1(t)*Hc1 + u2(t)*Hc2 + ....)
That is the time-dependent Hamiltonian has a constant part (called here the drift) and time vary parts, which are the control Hamiltonians scaled by some functions u_j(t) known as control amplitudes
In this case the drift is simply a rotation about z and the (time-varying) control is a rotation about x
In theory this system is fully controllable (irrespective of global phase) and so any unitary target could be chosen; we have chosen the Hadamard gate.

```python
# Drift Hamiltonian
H_d = sigmaz()
# The (single) control Hamiltonian
H_c = [sigmax()]
# start point for the gate evolution
U_0 = identity(2)
# Target for the gate evolution Hadamard gate
U_targ = gates.hadamard_transform(1)
```

### Defining the time evolution parameters


To solve the evolution the control amplitudes are considered constant within piecewise timeslots, hence the evolution during the timeslot can be calculated using U(t_k) = expm(-i*H(t_k)*dt). Combining these for all the timeslots gives the approximation to the evolution from the identity at t=0 to U(T) at the t=evo_time
The number of timeslots and evo_time have to be chosen such that the timeslot durations (dt) are small compared with the dynamics of the system.

```python
# Number of time slots
n_ts = 10
# Time allowed for the evolution
evo_time = 10
```

### Set the conditions which will cause the pulse optimisation to terminate


At each iteration the fidelity of the evolution is tested by comparaing the calculated evolution U(T) with the target U_targ. For unitary systems such as this one this is typically:
f = normalise(overlap(U(T), U_targ))
For details of the normalisation see [1] or the source code.
The maximum fidelity (for a unitary system) calculated this way would be 1, and hence the error is calculated as fid_err = 1 - fidelity. As such the optimisation is considered completed when the fid_err falls below such a target value.

In some cases the optimisation either gets stuck in some local minima, or the fid_err_targ is just not achievable, therefore some limits are set to the time/effort allowed to find a solution.

The algorithm uses gradients to direct its search for the minimum fidelity error. If the sum of all the gradients falls below the min_grad, then it is assumed some local minima has been found.

```python
# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
```

### Set the initial pulse type


The control amplitudes must be set to some initial values. Typically these are just random values for each control in each timeslot. These do however result in erratic optimised pulses. For this example, a solution will be found for any initial pulse, and so it can be interesting to look at the other initial pulse alternatives.

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


In this step the L-BFGS-B algorithm is invoked. At each iteration the gradient of the fidelity error w.r.t. each control amplitude in each timeslot is calculated using an exact gradient method (see [1]). Using the gradients the algorithm will determine a set of piecewise control amplitudes that reduce the fidelity error. With repeated iterations an approximation of the Hessian matrix (the 2nd order differentials) is calculated, which enables a quasi 2nd order Newton method for finding a minima.  The algorithm continues until one of the termination conditions defined above has been reached.

```python
result = cpo.optimize_pulse_unitary(
    H_d,
    H_c,
    U_0,
    U_targ,
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


Firstly the performace statistics are reported, which gives a breadown of the processing times. The times given are those that are associated with calculating the fidelity and the gradients. Any remaining processing time can be assumed to be used by the optimisation algorithm (L-BFGS-B) itself. In this example it can be seen that the majority of time is spent calculating the propagators, i.e. exponentiating the combined Hamiltonian.

The optimised U(T) is reported as the 'final evolution', which is essentially the string representation of the Qobj that holds the full time evolution at the point when the optimisation is terminated.

The key information is in the summary (given) last. Here the final fidelity is reported and the reasonn for termination of the algorithm.

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


Here the (random) starting pulse is plotted along with the pulse (control amplitudes) that was found to produce the target gate evolution to within the specified error.

```python
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
# ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(
    result.time,
    np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
    where="post",
)

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(
    result.time,
    np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
    where="post",
)
plt.tight_layout()
plt.show()
```

### Versions

```python
about()
```

### References


[1] Machnes et.al., DYNAMO - Dynamic Framework for Quantum Optimal Control. arXiv.1011.4874
