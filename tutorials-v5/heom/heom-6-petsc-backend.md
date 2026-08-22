---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# PETSc Backend for HEOM

## New examples added that were not in the paper
*(Note: The examples below use models and benchmark scripts specifically designed for testing the new PETSc distributed computing capabilities and are not directly associated with the examples in the original QuTiP-BoFiN paper.)*

### Introduction

In this tutorial, we will discuss how to use the PETSc backend for the Hierarchical Equations of Motion (HEOM) solver in QuTiP. The HEOM method is a powerful non-perturbative approach to simulate the dynamics of open quantum systems. However, as the hierarchy depth or the system size increases, the number of Auxiliary Density Operators (ADOs) grows rapidly, leading to a massive system of coupled ordinary differential equations. 

To overcome memory and computational bottlenecks, the `petsc` backend allows for:
1. **Distributed Matrix Assembly:** Constructing the large hierarchy Liouvillian across multiple MPI ranks.
2. **Parallel ODE Integration:** Time evolution using scalable implicit and explicit solvers provided by the PETSc TS (Time Stepping) module.
3. **Parallel Steady-State Solving:** Directly finding the steady state of the HEOM system using iterative linear solvers (KSP) and preconditioners from PETSc.

In order to use the PETSc backend, you must have `petsc4py` (and optionally `mpi4py` to run distributed workloads) installed in your environment.

```python
import time
import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

def print0(*args):
    """Helper to print only on the root MPI rank."""
    if rank == 0:
        print(*args, flush=True)
```

### Example 1: Time Evolution of a Spin Chain (Ising Model)

We consider a short Ising spin chain coupled to Drude-Lorentz baths. The Hamiltonian for an $N$-spin chain with transverse magnetic field $g$ and nearest-neighbor interaction $J$ is:

$$ H = \sum_{i=0}^{N-1} g \sigma_z^{(i)} - \sum_{i=0}^{N-2} J \sigma_x^{(i)} \sigma_x^{(i+1)} $$

We couple some of the spins to independent Drude-Lorentz baths.

#### Setup System
We define a 2-spin system where the first spin is coupled to a bath.

```python
N = 2
g0 = 1.0
J0 = 1.4    

# Operators for individual qubits
g = g0 * np.ones(N)
J = J0 * np.ones(N)
sx_list, sy_list, sz_list = [], [], []

for i in range(N):
    op_list = [qt.qeye(2)] * N
    op_list[i] = qt.sigmax()
    sx_list.append(qt.tensor(op_list))
    
    op_list[i] = qt.sigmay()
    sy_list.append(qt.tensor(op_list))
    
    op_list[i] = qt.sigmaz()
    sz_list.append(qt.tensor(op_list))
```

Next, we construct the Hamiltonian and the HEOM baths. We use the `DrudeLorentzBath` which encapsulates the spectral density properties and generates the correct multi-exponential correlation function expansions.

```python
H = 0.
for i in range(N):
    H += g[i] * sz_list[i]

for n in range(N - 1):
    H += -J[n] * sx_list[n] * sx_list[n + 1]

# Attach a bath to the first site (site 0)
baths = [
    DrudeLorentzBath(
        sz_list[0],
        lam=0.01,
        gamma=1.0,
        T=1.0,
        Nk=4,
    )
]

# Initial state: both spins in ground state
psi0 = qt.tensor([qt.basis(2, 0)] * N)
rho0 = qt.ket2dm(psi0)
```

#### Running the Distributed PETSc Backend

To utilize the PETSc backend, we specify `backend="petsc"` when initializing the `HEOMSolver`. We also pass an `options` dictionary tailored to the PETSc `TS` integrator. For stiff ODEs like those often produced by HEOM, implicit solvers like BDF (`ts_type: "bdf"`) are highly recommended.

```python
MAX_DEPTH = 6
tlist = np.linspace(0, 10, 50)

print0(f"Initializing PETSc HEOMSolver (MPI Size: {size})...")
solver_petsc = HEOMSolver(
    H, baths, max_depth=MAX_DEPTH, backend="petsc", 
    options={
        "store_ados": False, 
        "progress_bar": None,
        "store_states": True,
        "ts_type": "bdf",       # Use implicit solver for stiff ODEs
        "ts_adapt": "basic",    # Enable adaptive step sizing
        "atol": 1e-8,
        "rtol": 1e-8,
        "max_steps": 100000
    }
)

if comm is not None: comm.Barrier()
print0(f"Total ADOs: {len(solver_petsc.ados.labels)}")

print0("Running time evolution...")
res_petsc = solver_petsc.run(rho0, tlist, e_ops=[sz_list[-1]])

if rank == 0:
    ts = solver_petsc._integrator.ts
    print0("\nPETSc Internal Stats:")
    print0(f"   TS Type          : {ts.getType()}")
    print0(f"   TS Steps         : {ts.getStepNumber()}")
    print0(f"   Final dt         : {ts.getTimeStep():.2e}")
```

### Example 2: Finding the Steady State

In many physical scenarios, we are interested in the long-time non-equilibrium steady state of the system rather than the transient dynamics. Evolving the system to $t \rightarrow \infty$ using an ODE solver can be slow. 

Instead, we can solve for the state directly by finding the kernel of the HEOM superoperator. The PETSc backend implements a scalable steady-state solver utilizing the Krylov Subspace (KSP) solvers in PETSc.

We use the same Hamiltonian and bath setup, but we don't need to specify time-evolution options.

```python
MAX_DEPTH = 10  # We can push to higher depth

solver_steady = HEOMSolver(
    H, baths, max_depth=MAX_DEPTH, backend="petsc"
)

if comm is not None: comm.Barrier()

print0("Solving for Steady State...")
# KSP (Krylov Subspace) options can be provided directly
steady_rho, steady_ados = solver_steady.steady_state(
    ksp_type="gmres", 
    pc_type="none",
    ksp_rtol=1e-6, 
    ksp_atol=1e-8
)

if rank == 0:
    print0(f"Trace of steady state: {steady_rho.tr():.6f}")
```

### HEOM PETSc Simulations on Computing Clusters via MPI

How to run petsc examples will in practice will depend on your HPC infrastructure. For testing we used the batch script below on the Fugaku supercomputer.

```bash
#!/bin/bash
#PJM -L "node=4"
#PJM -L "rscgrp=small"
#PJM -g "project_name"
#PJM -L "elapse=00:60:00"
#PJM --mpi "max-proc-per-node=48"
#PJM -x PJM_LLIO_GFSCACHE=/project_volume
#PJM -j
#PJM -o benchmark_ising_petsc_diagnostic.out

set -euo pipefail

source "$HOME/miniforge3-a64fx/etc/profile.d/conda.sh"
conda activate qutip-petsc


export PETSC_DIR="$HOME/other-code/petsc"
export PETSC_ARCH="arch-fugaku-complex-1.2.43-debug"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

unset LD_PRELOAD
export XOS_MMM_L_HPAGE_TYPE=none

# Program configuration.

PYTHON="$(command -v python)"
BENCHMARK="$PWD/benchmark_ising_petsc_fugaku.py"

# Max of 48 ranks per node, 4 nodes, 192 ranks in total
NPROCS=192 
MAX_DEPTH=6

mpiexec -n "$NPROCS" "$PYTHON" -X faulthandler -u "$BENCHMARK"
```

### References

[1] [Tanimura, J. Chem. Phys. (2020)](https://arxiv.org/abs/2006.05501).  
[2] [Lambert *et al.*, Phys. Rev. Research (2023)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.013181).  
[3] [Abhyankar *et al.*, arXiv:1806.01437 [math.NA]](https://arxiv.org/abs/1806.01437).
<!-- #endregion -->
