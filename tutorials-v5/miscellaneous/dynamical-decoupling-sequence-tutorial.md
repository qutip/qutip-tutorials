---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Engineering Dynamic Decoupling Sequence Quantum Simulations with Trapped Ions 

Authors: Tarun Kumar Allamsetty (Duke University), Emma Stavropoulos (Duke University)
Reference: Morong, W., et al. (2023). Engineering Dynamic Decoupling Sequence Quantum Simulations with Trapped Ions. PRX Quantum, 4(1), 010334. https://doi.org/10.1103/PRXQuantum.4.010334

This notebook demonstrates how dynamic decoupling sequences can engineer desired Hamiltonians, specifically an Ising spin model with two ions.
The notebook is divided into three sections:
1. **Target Hamiltonian:** Defining the ideal two-spin Ising model.
2. **Noisy Hamiltonian:** Examining the impact of experimental Stark shift noise.
3. **Dynamic Decoupling Sequence:** Implementing a sequence to suppress noise.



## Section 1: Target Hamiltonian

The target is the XX Ising model, defined as:
$$
H_t = \sum_{j<j'} J_{j,j'} \sigma_j^x \sigma_{j'}^x
$$ 

For two spins, the Hamiltonian is $H = J \sigma_1^x \sigma_{2}^x$. We analyze the time evolution starting from the spin-up state $|11\rangle_z$.

Since $H$ is time-independent, the Schrödinger equation is solved by exponentiating the Hamiltonian:
$$
|\psi(t)\rangle = e^{-i H t} |11\rangle
$$
Using a Taylor series expansion and the property $H^2 = I$:
$$
\begin{aligned}
|\psi(t)\rangle &= e^{-i J \sigma_1^x \sigma_{2}^x t} |11\rangle_z \\
&= \cos(J t) |11\rangle_z - i \sin(J t) \sigma_1^x \sigma_2^x |11\rangle_z \\
&= \cos(J t) |11\rangle_z - i \sin(J t) |00\rangle_z
\end{aligned}
$$

The analytical expectation value of the spin in the $z$-direction, $\langle \sigma^z_1 \rangle$, is:
$$
\langle \sigma^z_1 \rangle = \cos^2(J t) - \sin^2(J t) = \cos(2 J t) 
$$

The simulation results below show this sinusoidal oscillation.


```python
import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmax, sigmay, sigmaz, tensor, identity, mesolve, basis, about
```

```python
J = 2 * np.pi * 400
sx1 = tensor(sigmax(), identity(2))
sx2 = tensor(identity(2), sigmax())
sy1 = tensor(sigmay(), identity(2))
sy2 = tensor(identity(2), sigmay())
sz1 = tensor(sigmaz(), identity(2))
sz2 = tensor(identity(2), sigmaz())
H_xx = J * (sx1 * sx2)

times = np.linspace(0, 0.005, 400)
psi0 = tensor(basis(2,1), basis(2,1))
result_xx = mesolve(H_xx, psi0, times, [], [sz1, sz2])

mean_spin = (result_xx.expect[0] + result_xx.expect[1]) / 2

plt.figure(figsize=(8, 5))
plt.plot(times, mean_spin, label=r'$\langle \sigma_z \rangle$', color='teal')
plt.xlabel('Time')
plt.ylabel('Expectation Value')
plt.title('Dynamics of $\sigma_z$')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

```

# Section 2: Noisy Hamiltonian

Trapped-ion implementations are susceptible to AC Stark shift noise. We examine its impact on the dynamics:

$$
H_{noisy} = \sum_{j<j'} J_{j,j'} \sigma_j^x \sigma_{j'}^x + \epsilon(t) \sum_j  \sigma_j^z
$$ 

For two ions, we model the noise as a periodic signal:
$H_{noisy} = J \sigma_1^x \sigma_{2}^x + \epsilon(t) (\sigma_1^z + \sigma_2^z)$, where $\epsilon(t)$ varies slowly compared to the coupling strength $J$.

Note that $H^2 \propto I$ and the Hamiltonian terms no longer commute. Analytic solutions are difficult for this system, so we use QuTiP's time-dependent solver.

The simulation results show exponential decay in the oscillations. This occurs because the noise term rotates the spins in a different direction than the target interaction.



```python
def epsilon_t(t):
    return 10*J * np.sin(30 * t) 

# Combine into time-dependent list format
H_noise_op = sz1 + sz2
H_xx_noisy = [H_xx, [H_noise_op, epsilon_t]]

psi0 = tensor(basis(2,1), basis(2,1))
# Run the solver
result = mesolve(H_xx_noisy, psi0, times, [], [sz1, sz2])
mean_spin_stark = (result.expect[0] + result.expect[1]) / 2 

plt.figure(figsize=(8, 5))
plt.plot(times, mean_spin_stark, label=r'$\langle \sigma_z \rangle$ with Stark Noise')
plt.xlabel('Time')
plt.ylabel('Expectation Value')
plt.title('Time Evolution with AC Stark Noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

# Section 3: Dynamic Decoupling Sequence

For analysis, assume $H_{noisy} = J \sigma_1^x \sigma_{2}^x + \epsilon (\sigma_1^z + \sigma_2^z)$. The Stark shift noise is assumed constant as it is slowly varying.

We analyze the simplified sequence shown in Figure 2 of the reference:
$$
U_1 = R_{-y}(\pi) e^{-i\mathcal{H}_{noisy} t_1} R_x(\pi) e^{-i\mathcal{H}_{noisy} t_1} 
$$

Average Hamiltonian Theory (AHT) gives the effective Hamiltonian:

$ \bar{H} = \frac{1}{T} \sum_i  \bar{H}_i t_i $; $ \bar{H}_i = (P_{i-1} \dots P_1)^{-1} H_i P_1 \dots P_{i-1} $

For the two-pulse case:
1. $\bar{H}_1 = H_{noisy} = J \sigma_1^x \sigma_{2}^x + \epsilon (\sigma_1^z + \sigma_2^z)$
2. $\bar{H}_2 = R_{x}(-\pi)H_{noisy}R_{x}(\pi) = J \sigma_1^x \sigma_{2}^x - \epsilon (\sigma_1^z + \sigma_2^z)$

The average Hamiltonian cancels the noise term. This effectively inverts the noise's impact in the toggling frame, leading to its cancellation over time. This requires the noise to be slowly varying relative to the pulse sequence.


```python
import numpy as np
import matplotlib.pyplot as plt
from qutip import mesolve, Options

opts = Options(store_final_state=True)
Rx_pi = (-1j * (np.pi / 2) * (sx1 + sx2)).expm()
Ry_minus_pi = (-1j * (-np.pi / 2) * (sy1 + sy2)).expm()

num_cycles = 30
total_intervals = num_cycles * 2  # 2 halves per cycle

time_chunks = np.array_split(times, total_intervals)

current_state = psi0
mean_spin_all = [] 

for cycle in range(num_cycles):
    
    # ==== FIRST HALF ====
    chunk_idx_1 = cycle * 2
    times_first_half = time_chunks[chunk_idx_1]
    
    res_1 = mesolve(H_xx_noisy, current_state, times_first_half, [], [sz1, sz2], options=opts)
    mean_spin_1 = (res_1.expect[0] + res_1.expect[1]) / 2 
    mean_spin_all.append(mean_spin_1)
    
    current_state = Rx_pi * res_1.final_state
    
    # ==== SECOND HALF ====
    chunk_idx_2 = cycle * 2 + 1
    times_second_half = time_chunks[chunk_idx_2]

    # Note the inversion of the measurement operator to account for the Rx(pi) pulse.
    res_2 = mesolve(H_xx_noisy, current_state, times_second_half, [], [-sz1, -sz2], options=opts)
    mean_spin_2 = (res_2.expect[0] + res_2.expect[1]) / 2
    
    mean_spin_all.append(mean_spin_2)
    
    current_state = Ry_minus_pi * res_2.final_state

mean_spin_dd = np.concatenate(mean_spin_all)

plt.figure(figsize=(10, 5))
plt.plot(times, mean_spin_dd, label=f'DD Sequence ({num_cycles} Cycles)', color='black')

# Mark the pulse timings
for time_chunk in time_chunks[1:]:
    plt.axvline(time_chunk[0], color='gray', linestyle=':', alpha=0.4)

plt.xlabel('Time')
plt.ylabel(r'Mean $\langle \sigma_z \rangle$')
plt.title(f'Time Evolution with AC Stark Noise ({num_cycles} Repetitions)')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.show()
```

# Section 4: Conclusion

Dynamical decoupling suppresses decoherence but also slows the oscillation frequency. This trade-off is discussed in the referenced paper.

```python
plt.figure(figsize=(10, 5))
plt.plot(times, mean_spin, label=r'Ideal $H_{XX}$', color='#1f77b4', linestyle='--', alpha=0.8)
plt.plot(times, mean_spin_stark, label='Uncorrected Stark Noise', color='gray', alpha=0.4, linewidth=1.5)
plt.plot(times, mean_spin_dd, label=f'DD Sequence ({num_cycles} Cycles)', color='#d62728', linewidth=2)

for time_chunk in time_chunks[1:]:
    plt.axvline(time_chunk[0], color='black', linestyle=':', alpha=0.2, label='_nolegend_')

plt.xlabel('Time ($t/J$)') # Standardized units for Duke lab notebooks
plt.ylabel(r'Magnetization $\langle \sigma_z \rangle$')
plt.title('Error Suppression via Dynamical Decoupling Sequence')
plt.legend(loc='upper right', frameon=True, fontsize='small')
plt.grid(True, which='both', linestyle='--', alpha=0.2)
plt.tight_layout()
plt.show()
```

```python
about()
```
