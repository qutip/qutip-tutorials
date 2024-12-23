---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: qutip-dev310
    language: python
    name: python3
---

## Waveguide in Lindblad limit with input-output HEOM
Authors: Neill Lambert (nwlambert@gmail.com) and Mauro Cirio (mauro.cirio@gmail.com)

In this example we reproduce the example from https://arxiv.org/abs/2408.12221 

Some of the functions used to define the waveguide correlations and parameters are taken from
https://github.com/mcirio/io-HEOM-Markovian/  and adapted under the MIT licence for use here.

Even though this example is in the Markovian limit, we can use the QuTiP HEOM solver to manage the input-output 
formalism for us.  We avoid reproducing a lot of the derivation here, as we largely follow exactly what is the paper above.

However, intuitively, the system is coupled to a waveguide with linear dispersion, and a flat spectral density, under the rotating-wave
approximation. Thus from the point of view of the system, a qubit, its normal interaction with the bath just causes standard Markovian Lindblad decay.
We full capture this by creating a Liouvillian to describe that decay.  The system evolution does not evolve 'normal' HEOM.

We then introduce input to the waveguide in the form of a Gaussian pulse propagating towards the qubit 'from the left' at $t=0$, which is defined with some mean position $x_{in}$ and momentum $p_{in}$, the latter of which is given so that it has energy close to the qubit resonance. We can define this input using the input-output HEOM.  

We then introduce 'output' observables which are defined as the mean number of photons across modes in a spatial interval $\delta_x$ at position $X_{out}$ at time $t_{out}$.  Again, this can be defined in terms of input-output heom.




```python
from functools import partial

import matplotlib.animation as animation
import numpy as np
import qutip as qt
from IPython.display import HTML
from matplotlib import pyplot as plt
import qutip
from qutip import basis, destroy, expect, qeye, sigmaz
from qutip.solver.heom import HEOMSolver, InputOutputBath

from copy import copy

from matplotlib.colors import LinearSegmentedColormap

from scipy.special import erfi

%matplotlib inline

```

```python
"""
Adapted from https://github.com/mcirio/io-HEOM-Markovian/ under the
MIT licence:

MIT License

Copyright (c) 2024 mauro cirio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def Omega_in_pm(Gamma, c, sigma_in, p_in, x_in, pm, t):
    # input time-dependent "frequency"
    tau = 1 / (2 * sigma_in)
    m1 = 2 ** (-1) * (2 * np.pi) ** (-0.25) * tau ** (-0.5)
    m2 = 1j * np.sqrt(Gamma * c)

    xt = x_in + c * t
    m3 = np.exp(-(xt**2) / (4 * tau**2) - 1j * pm * p_in * xt)
    m4 = 1 - 1j * pm * erfi(xt / (2 * tau) + 1j * pm * p_in * tau)
    res1 = m1 * m2 * m3 * m4

    xmt = x_in - c * t
    m3 = np.exp(-(xmt**2) / (4 * tau**2) - 1j * pm * p_in * xmt)
    m4 = 1 + 1j * pm * erfi(xmt / (2 * tau) + 1j * pm * p_in * tau)
    res2 = m1 * m2 * m3 * m4
    # Note: This has an -i prefactor not present in the journal article
    # because of slightly different definitions within the HEOM solver
    return -1j * (res1 + res2)


# correlation functions needed to evaluation the output, number of
# photons at x_out at t_out


def cross(c, Deltax, sigma_in, p_in, x_in, x_out, t_out):
    # input time-dependent "frequency"
    tau = 1 / (2 * sigma_in)
    m1 = np.sqrt(Deltax) * 2 ** (-1) * (2 * np.pi) ** (-0.25) * tau ** (-0.5)

    Dx = x_out - x_in

    xmt = Dx - c * t_out
    m3 = np.exp(-(xmt**2) / (4 * tau**2) - 1j * p_in * xmt)
    m4 = 1 - 1j * erfi(xmt / (2 * tau) + 1j * p_in * tau)
    res2 = m1 * m3 * m4

    xt = Dx + c * t_out
    m3 = np.exp(-(xt**2) / (4 * tau**2) - 1j * p_in * xt)
    m4 = 1 + 1j * erfi(xt / (2 * tau) + 1j * p_in * tau)
    res1 = m1 * m3 * m4

    return res1 + res2
```

```python

# Define the units and system Hamiltonian

# Units
w = 2.0 * np.pi  # frequency
ll = 1.0  # length
c = ll * w  # speed of light
ws = w  # system frequency

# Bath properties:
Gamma = 0.4 * w  # Markovian decay rate
g_p = np.sqrt(Gamma * c / (2 * np.pi))  # system-bath coupling strenght

# HEOM parameters:
T = 15 / w  # max time

N_T_out = 15  # number of output times to observe
t_out_list = np.linspace(0, T, N_T_out)  # out times


# Increasing N_X_out will give smoother output plots/animations
# but increase run time.
N_X_out = 50  # number of output points in space
x_in = -4.5 * ll  # space of initial wave-packet
x_out_list = np.linspace(
    -2 * abs(x_in), 2 * abs(x_in), N_X_out
)  # out-space discretization
Deltax_out = abs(x_out_list[1] - x_out_list[0])  # out space resolution


N_T = 100  # number of points in time for each run
t_list = np.linspace(0, T, N_T)  # dynamical times

# Initial pulse properties
P = 10 / ll  # max momentum
# input pulse
p_in = P / 10.0  # momentum of initial wave-packet
sigma_in = 1 / 2.0 * P / 10.0  # standard deviation of the initial wave-packet
detuning = (
    c * p_in / ws
)  # intuitive detuning value between the initial wave-packet and the system

```

```python
# system setup, Markovian decay

Hsys = 0.5 * ws * sigmaz()
rho0 = basis(2, 1) * basis(2, 1).dag()

L0 = qt.liouvillian(Hsys, [np.sqrt(2 * Gamma) * destroy(2).dag()])

obs = destroy(2) * destroy(2).dag()  # system observable
```

```python
# make input functions just a function of time
Omega_in_p_t = partial(Omega_in_pm, Gamma, c, sigma_in, p_in, x_in, 1)
Omega_in_m_t = partial(Omega_in_pm, Gamma, c, sigma_in, p_in, x_in, -1)
```

```python
# we assume the bath is initiall in vacuum and at t=0 two
# ''input'' operators are applied.
# Here we need two because we have a RWA bath,
# which breaks some symmetry of the normal approach.

ck_in_p = [Omega_in_p_t]
ck_in_m = [Omega_in_m_t]
```

```python
# Note: differing from https://github.com/mcirio/io-HEOM-Markovian/  here
# we avoid using the fact the output correlations are delta functions
# to employ discrete kicks.
# We instead approximate the delta functions as Gaussians.
# Mathematicians may not like this.

def delta_fun_approx(x, xs):
    sigma = 1e-2
    return (np.exp(-(((x - xs) / sigma) ** 2) / 2) /
            (sigma * np.sqrt(2 * np.pi)))


options = {
    "nsteps": 15000,
    "store_states": True,
    "progress_bar": False,
    "store_ados": True,  # docs are wrong
    "max_step": 1e-3,
}

NC = 4

x_out_expect = []

for x_out in x_out_list:
    x_out_t = []

    for t_out in t_out_list:

        tout_p = t_out + x_out / c
        tout_m = t_out - x_out / c

        def omega_out(t):
            return np.sqrt(Gamma * np.abs(Deltax_out) / c) * (
                delta_fun_approx(t, tout_p) + delta_fun_approx(t, tout_m)
            )

        def fun_1_R(t):
            return omega_out(t)

        def fun_2_L(t):
            return omega_out(t)

        ck_out_1_R = [fun_1_R]
        ck_out_2_L = [fun_2_L]

        bathp = InputOutputBath(destroy(2), ck_input=ck_in_p, tag="in1")
        bathm = InputOutputBath(destroy(2).dag(), ck_input=ck_in_m, tag="in2")
        bath1 = InputOutputBath(destroy(2), ck_output_fn_R=ck_out_1_R,
                                tag="out1")
        bath2 = InputOutputBath(destroy(2).dag(), ck_output_fn_L=ck_out_2_L,
                                tag="out2")

        HEOMMats = HEOMSolver(L0, [bathp, bathm, bath1, bath2], NC,
                              options=options)

        t_list = np.linspace(0, t_out, 100)
        resultMats = HEOMMats.run(rho0, t_list)
        ft = cross(c, Deltax_out, sigma_in, p_in, x_in, x_out, t_out)

        # 00,00
        zzzz = np.abs(ft) ** 2 * (resultMats.states[-1]).tr()
        # 11,11
        labels = resultMats.ado_states[-1].filter(
            level=4, tags=["in1", "in2", "out1", "out2"]
        )
        states = []
        for label in labels:
            states.append(resultMats.ado_states[-1].extract(label))

        oooo = np.sum([expect(state, qeye(2)) for state in states])

        # 00,11
        labels = resultMats.ado_states[-1].filter(level=2, tags=["out1",
                                                                 "out2"])
        states = []
        for label in labels:
            states.append(resultMats.ado_states[-1].extract(label))

        zzoo = np.sum([expect(state, qeye(2)) for state in states])

        # 10,01
        labels = resultMats.ado_states[-1].filter(level=2, tags=["in1",
                                                                 "out2"])
        states = []
        for label in labels:

            states.append(resultMats.ado_states[-1].extract(label))

        ozzo = ft * np.sum([expect(state, qeye(2)) for state in states])

        # 01,10
        labels = resultMats.ado_states[-1].filter(level=2, tags=["in2",
                                                                 "out1"])
        states = []
        for label in labels:

            states.append(resultMats.ado_states[-1].extract(label))

        zooz = np.conj(ft) * np.sum([expect(state, qeye(2)) for state in
                                     states])

        x_out_t.append(zzzz + oooo - zzoo - ozzo - zooz)

    x_out_expect.append(x_out_t)
```

```python
Delta_x = x_out_list[1] - x_out_list[0]

x_out_list_rescaled = [x / abs(x_in) for x in x_out_list]
x_out_expect_T = np.real(np.array(x_out_expect)).T.tolist()
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
for n_HEOM, data_HEOM in enumerate(x_out_expect_T):

    offset_data_HEOM = [x / Delta_x for x in data_HEOM]
    ax.plot(x_out_list_rescaled, offset_data_HEOM)


fig.set_tight_layout(True)

ax.tick_params(axis="both", which="major")
```

```python
# Animate the pulse propagation

cmap_name = "my_gradient"
gradient_colors = ["indigo", "lightblue"]
n_bins = 15
cmap = LinearSegmentedColormap.from_list(cmap_name, gradient_colors, N=n_bins)
x_out_list_rescaled = [x / abs(x_in) for x in x_out_list]
x_out_expect_T = np.real(np.array(x_out_expect)).T.tolist()
data_list = x_out_expect_T
n_data = 1.0 * len(data_list)

artist_list = list()
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
for n_HEOM, data_HEOM in enumerate(x_out_expect_T):

    n_HEOM = copy(1.0 * n_HEOM)
    offset_data_HEOM = copy([x / Delta_x for x in data_HEOM])

    (artist,) = ax.plot(
        x_out_list_rescaled, offset_data_HEOM, color="black", linewidth=5
    )
    artist2 = ax.fill_between(
        x_out_list_rescaled,
        [0 for _ in x_out_list],
        offset_data_HEOM,
        zorder=n_data - n_HEOM,
        color=cmap(float(n_HEOM / (n_data - 1))),
        alpha=1,
    )
    artist_list.append([artist2, artist])

fig.set_tight_layout(True)

ax.tick_params(axis="both", which="major")


output = animation.ArtistAnimation(
    fig, artist_list, interval=100, blit=True, repeat_delay=10000
)


# output.save("pulse.gif")
```

```python
HTML(output.to_jshtml())
```

```python
# Construct the conditional system state from ADOs, system conditioned on input
# This is not dependent on output so we just use the last saved run from above.

result_2 = []

obser = destroy(2) * destroy(2).dag()
for t in range(len(t_list)):
    labels = resultMats.ado_states[t].filter(level=2, tags=["in1", "in2"])
    states = []
    for label in labels:
        states.append(resultMats.ado_states[t].extract(label))

    result_2.append(
        expect(resultMats.states[t], obser)
        - np.sum([expect(state, obser) for state in states])
    )
```

```python
plt.figure()
plt.plot(
    [t / abs(x_in / c) for t in t_list],
    expect(resultMats.states, obser),
    "r",
    alpha=1,
    label=r"heom",
)
plt.plot(
    [t / abs(x_in / c) for t in t_list],
    result_2,
    "b--",
    alpha=1,
    label=r"heom conditional on input",
)

# plt.plot(tlist, result_mc.runs_expect[0][7],':',label=r'mcsolve 1 run')

plt.xlabel("Time (|x|/c)", fontsize=18)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=18)
plt.legend()
# plt.savefig("mcsolve.pdf")
plt.show()
```

```python
qutip.about()
```
