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

# Calculation of control fields for QFT gate on two qubits using the CRAB algorithm


Alexander Pitchford (agp1@aber.ac.uk)

<!-- #raw -->
Example to demonstrate using the CRAB [1][2] algorithm in the control library 
to determine control pulses using the ctrlpulseoptim.create_pulse_optimizer function to 
generate an Optimizer object, through which the configuration can be
manipulated before running the optmisation algorithm. In this case it is
demonstrated by modifying the CRAB pulse parameters to show how pulse constraints
for controls can be applied.

The system in this example is two qubits in constant fields in x, y and z
with a variable independant controls fields in x and y acting on each qubit
The target evolution is the QFT gate. The user can experiment with the
different:
    phase options - phase_option = SU or PSU
    propagtor computer type prop_type = DIAG or FRECHET
    fidelity measures - fid_type = UNIT or TRACEDIFF

The user can experiment with the timeslicing, by means of changing the
number of timeslots and/or total time for the evolution.
Different guess and ramping pulse parameters can be tried.
The initial and final pulses are displayed in a plot
<!-- #endraw -->

```python
import datetime

import matplotlib.pyplot as plt
import numpy as np

import qutip_qtrl.pulsegen as pulsegen
import qutip_qtrl.pulseoptim as cpo
from qutip import identity, sigmax, sigmay, sigmaz, tensor, about
from qutip_qip.algorithms import qft

example_name = "QFT"

```

### Defining the physics

```python
Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5 * identity(2)

# Drift Hamiltonian
H_d = 0.5 * (tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
# The (four) control Hamiltonians
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
n_ctrls = len(H_c)
# start point for the gate evolution
U_0 = identity(4)
# Target for the gate evolution - Quantum Fourier Transform gate
U_targ = qft(2)
```

### Defining the time evolution parameters

```python
# Number of time slots
n_ts = 200
# Time allowed for the evolution
evo_time = 10
```

### Set the conditions which will cause the pulse optimisation to terminate

```python
# Fidelity error target
fid_err_targ = 1e-3
# Maximum iterations for the optisation algorithm
max_iter = 20000
# Maximum (elapsed) time allowed in seconds
max_wall_time = 300
```

### Give an extension for output files

```python
# Set to None to suppress output files
f_ext = "{}_n_ts{}.txt".format(example_name, n_ts)
```

### Create the optimiser objects

```python
optim = cpo.create_pulse_optimizer(
    H_d,
    H_c,
    U_0,
    U_targ,
    n_ts,
    evo_time,
    fid_err_targ=fid_err_targ,
    max_iter=max_iter,
    max_wall_time=max_wall_time,
    alg="CRAB",
    dyn_type="UNIT",
    prop_type="DIAG",
    fid_type="UNIT",
    fid_params={"phase_option": "PSU"},
    gen_stats=True,
)
```

### Configure the pulses for each of the controls

```python
dyn = optim.dynamics

# Control 1
crab_pgen = optim.pulse_generator[0]
# Start from a ramped pulse
guess_pgen = pulsegen.create_pulse_gen("LIN", dyn=dyn, pulse_params={"scaling": 3.0})
crab_pgen.guess_pulse = guess_pgen.gen_pulse()
crab_pgen.scaling = 0.0
# Add some higher frequency components
crab_pgen.num_coeffs = 5

# Control 2
crab_pgen = optim.pulse_generator[1]
# Apply a ramping pulse that will force the start and end to zero
ramp_pgen = pulsegen.create_pulse_gen(
    "GAUSSIAN_EDGE", dyn=dyn, pulse_params={"decay_time": evo_time / 50.0}
)
crab_pgen.ramping_pulse = ramp_pgen.gen_pulse()

# Control 3
crab_pgen = optim.pulse_generator[2]
# Add bounds
crab_pgen.scaling = 0.5
crab_pgen.lbound = -2.0
crab_pgen.ubound = 2.0


# Control 4
crab_pgen = optim.pulse_generator[3]
# Start from a triangular pulse with small signal
guess_pgen = pulsegen.PulseGenTriangle(dyn=dyn)
guess_pgen.num_waves = 1
guess_pgen.scaling = 2.0
guess_pgen.offset = 2.0
crab_pgen.guess_pulse = guess_pgen.gen_pulse()
crab_pgen.scaling = 0.1

init_amps = np.zeros([n_ts, n_ctrls])
for j in range(dyn.num_ctrls):
    pgen = optim.pulse_generator[j]
    pgen.init_pulse()
    init_amps[:, j] = pgen.gen_pulse()

dyn.initialize_controls(init_amps)
```

### Run the pulse optimisation

```python
# Save initial amplitudes to a text file
if f_ext is not None:
    pulsefile = "ctrl_amps_initial_" + f_ext
    dyn.save_amps(pulsefile)
    print("Initial amplitudes output to file: " + pulsefile)

print("***********************************")
print("Starting pulse optimisation")
result = optim.run_optimization()

# Save final amplitudes to a text file
if f_ext is not None:
    pulsefile = "ctrl_amps_final_" + f_ext
    dyn.save_amps(pulsefile)
    print("Final amplitudes output to file: " + pulsefile)
```

### Report the results

```python
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Initial fidelity error {}".format(result.initial_fid_err))
print("Final fidelity error {}".format(result.fid_err))
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
        label="u{}".format(j),
    )
ax2.legend(loc=8, ncol=n_ctrls)
plt.tight_layout()
plt.show()
```

### Versions

```python
about()
```

<!-- #raw -->
References:

3.  Doria, P., Calarco, T. & Montangero, S. 
    Optimal Control Technique for Many-Body Quantum Dynamics. 
    Phys. Rev. Lett. 106, 1–4 (2011).

4.  Caneva, T., Calarco, T. & Montangero, S. 
    Chopped random-basis quantum optimization. 
    Phys. Rev. A - At. Mol. Opt. Phys. 84, (2011).
<!-- #endraw -->

<!-- #raw -->

<!-- #endraw -->
