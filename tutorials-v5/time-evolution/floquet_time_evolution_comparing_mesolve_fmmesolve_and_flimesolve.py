# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 01:15:17 2023

@author: Fenton
"""
import numpy as np
import matplotlib.pyplot as plt

from qutip import (Qobj, about, basis, flimesolve, fmmesolve, fsesolve,
                   mesolve, plot_expectation_values, sigmax, sigmaz, destroy)


def noise_spectrum(omega):
    return (omega > 0) * gamma * omega / (4 * np.pi)


# define constants
delta = 0.2 * 2 * np.pi
eps0 = 2 * np.pi
A = 2.5 * 2 * np.pi
omega = 2 * np.pi

# Non driving hamiltoninan
H0 = -delta / 2.0 * sigmax() - eps0 / 2.0 * sigmaz()

# Driving Hamiltonian
H1 = [A / 2.0 * sigmaz(), "sin(w*t)"]
args = {"w": omega}

# combined hamiltonian
H = [H0, H1]

# initial state
psi0 = basis(2, 0)

# simulation time list
T = 2 * np.pi / omega
periods = 50
Nt = 2**6
tlist = np.linspace(0, periods * T, periods*Nt)

# Noise Spectral Density
gamma = 0.5

# Coupling operator and noise spectrum
c_ops = sigmax()
spectra_cb = [noise_spectrum]

# Solve using FMMEsolve
fmme_result = fmmesolve(
    H,
    psi0,
    tlist,
    c_ops=[c_ops],
    spectra_cb=spectra_cb,
    T=T,
    args=args,
)

# Solve using FLiMEsolve
flime_result = flimesolve(
    H,
    psi0,
    tlist,
    c_ops=[c_ops],
    c_op_rates=[np.sqrt(gamma)],
    T=T,
    args=args,
)

# Solve using MEsolve
me_result = mesolve(
    H,
    psi0,
    tlist,
    c_ops=[np.sqrt(gamma)*c_ops],
    args=args,
    )

flimestates = np.stack([i.full() for i in flime_result.states])
fmmestates = np.stack([i.full() for i in fmme_result.states])
mestates = np.stack([i.full() for i in me_result.states])

me_fmme_diff = mestates-fmmestates
me_flime_diff = mestates-flimestates

fig, ax = plt.subplots(4, 4)

ax[0, 0].plot(tlist/T, mestates[:, 0, 0], color='lightcoral')
ax[0, 1].plot(tlist/T, mestates[:, 0, 1], color='gold')
ax[0, 2].plot(tlist/T, mestates[:, 1, 0], color='palegreen')
ax[0, 3].plot(tlist/T, mestates[:, 1, 1], color='paleturquoise')
ax[0, 0].set_ylabel('amplitude')
ax[0, 0].legend(['ME[0,0]'])
ax[0, 1].legend(['ME[0,1]'])
ax[0, 2].legend(['ME[1,0]'])
ax[0, 3].legend(['ME[1,1]'])

ax[1, 0].plot(tlist/T, fmmestates[:, 0, 0], color='tomato')
ax[1, 1].plot(tlist/T, fmmestates[:, 0, 1], color='goldenrod')
ax[1, 2].plot(tlist/T, fmmestates[:, 1, 0], color='limegreen')
ax[1, 3].plot(tlist/T, fmmestates[:, 1, 1], color='cornflowerblue')
ax[1, 0].set_ylabel('amplitude')
ax[1, 0].legend(['FMME[0,0]'])
ax[1, 1].legend(['FMME[0,1]'])
ax[1, 2].legend(['FMME[1,0]'])
ax[1, 3].legend(['FMME[1,1]'])

ax[2, 0].plot(tlist/T, flimestates[:, 0, 0], color='darkred')
ax[2, 1].plot(tlist/T, flimestates[:, 0, 1], color='darkgoldenrod')
ax[2, 2].plot(tlist/T, flimestates[:, 1, 0], color='forestgreen')
ax[2, 3].plot(tlist/T, flimestates[:, 1, 1], color='mediumblue')
ax[2, 0].set_ylabel('amplitude')
ax[2, 0].legend(['FLiME[0,0]'])
ax[2, 1].legend(['FLiME[0,1]'])
ax[2, 2].legend(['FLiME[1,0]'])
ax[2, 3].legend(['FLiME[1,1]'])

ax[3, 0].plot(tlist/T, me_fmme_diff[:, 0, 0],
              color='darkolivegreen', alpha=0.5)
ax[3, 0].plot(tlist/T, me_flime_diff[:, 0, 0],
              color='rebeccapurple', linestyle='--')
ax[3, 1].plot(tlist/T, me_fmme_diff[:, 0, 1],
              color='darkolivegreen', alpha=0.5)
ax[3, 1].plot(tlist/T, me_flime_diff[:, 0, 1],
              color='rebeccapurple', linestyle='--')
ax[3, 2].plot(tlist/T, me_fmme_diff[:, 1, 0],
              color='darkolivegreen', alpha=0.5)
ax[3, 2].plot(tlist/T, me_flime_diff[:, 1, 0],
              color='rebeccapurple', linestyle='--')
ax[3, 3].plot(tlist/T, me_fmme_diff[:, 1, 1],
              color='darkolivegreen', alpha=0.5)
ax[3, 3].plot(tlist/T, me_flime_diff[:, 1, 1],
              color='rebeccapurple', linestyle='--')
ax[3, 0].set_ylabel('Percent Deviation')
ax[3, 0].set_xlabel('time (in periods $\\tau$ of the Hamiltonian)')
ax[3, 1].set_xlabel('time (in periods $\\tau$ of the Hamiltonian)')
ax[3, 2].set_xlabel('time (in periods $\\tau$ of the Hamiltonian)')
ax[3, 3].set_xlabel('time (in periods $\\tau$ of the Hamiltonian)')
ax[3, 0].legend(['ME-FMME [0,0]',
                'ME-FLiME [0,0]'])
ax[3, 1].legend(['ME-FMME [0,1]',
                'ME-FLiME [0,1]'])
ax[3, 2].legend(['ME-FMME [1,0]',
                'ME-FLiME [1,0]'])
ax[3, 3].legend(['ME-FMME [1,1]',
                'ME-FLiME [1,1]'])

fig.suptitle("Evolution of driven two level system - "
             "absolute percent deviation of "
             "fmmesolve and flimesolve from mesolve")
