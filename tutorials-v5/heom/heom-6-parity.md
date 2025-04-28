---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: qutip-dev
    language: python
    name: python3
---

<!-- #region -->
# Reproducing the Kondo Peak, a tutorial on the HEOM parity Feature

Authors: [Gerardo Suarez](https://gsuarezr.github.io) & Neill Lambert, 2025

### Introduction


In this tutorial we will find the Kondo peak for a system of two spins 
connected to two fermionic leads. The Hamiltonian of this setup is given 
by:

$H_{T}=H_{S}+H_{E}+H_{I}$

Where $H_{S}$ is the system Hamiltonian



$H_{S}=  \sum_{i=1,2} 
\omega_{0} d_{i}^{\dagger} d_{i} + U d_{1}^{\dagger} d_{1}
d_{2}^{\dagger} d_{2} $


$H_{E}$ is the environment Hamiltonian

$H_{E}= \sum_{\alpha=L,R} \sum_{k} \epsilon_{\alpha,k} 
c^{\dagger}_{\alpha,k}c_{\alpha,k}$

and $H_{I}$ is the interaction Hamiltonian

$H_{I}=H_{I}^{(L,1)} +H_{I}^{(R,2)}$

with

$H_{I}^{(\alpha,i)}= \sum_{k} g_{\alpha,k} (c_{\alpha,k}^{\dagger} 
d_{i}+c_{\alpha,k} d_{i}^{\dagger})$


The interaction between the electronic system and Fermionic leads is 
characterized by a Lorentzian spectral density

$J(\omega)=\frac{1}{2 \pi} \frac{\Gamma_{\alpha} W_{f}^{2}}
{(\omega-\mu_{\alpha})^{2}+W_{f}^{2}}$

The example in this tutorial is Figure 2a in [Cirio et al](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033011)
, and the procedure is described in the supplemental material of [Li et al](https://arxiv.org/pdf/1207.6359) 

### Setting up the simulation

We start by importing the necessary packages
<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor,qeye,spre,operator_to_vector,expect,qeye,fdestroy
from qutip.solver.heom import HEOMSolver
from qutip.core import LorentzianEnvironment
import  scipy.sparse as sp
from scipy.sparse.linalg import lgmres
import qutip

%matplotlib inline
```

We define the system, bath and simulation parameters. Then we construct,
our fermionic environment using the `LorentzianEnvironment`  class from `QuTiP`

```python
# Simulation Parameters
# Ncc=2 is chosen for speed for more accurate results use Nc=5
Ncc = 2
Nk1 = 2 # We choose 2 Pade exponents for the first simulation
Nk2= 4 # 4 for the second one

N = 2 # 2 fermionic sites
d1 = fdestroy(N,0) # first site
d2 = fdestroy(N,1) # second site

# Bath paramters: both system and bath parameters are taken from Cirio et al.
mu = 0.  #chemical potential
Gamma  = 1  #coupling strength
W = 2.5 #bath width

#system params:
#coulomb repulsion
U = 3 * np.pi * Gamma
#impurity energy
w0 = - U / 2.

T = 0.2 * Gamma # Temperature

# Hamiltonian of the system
H = w0 *(d1.dag() * d1 + d2.dag() * d2) + U * d1.dag() * d1 * d2.dag() * d2

# Environment
env = LorentzianEnvironment(W=W,gamma=2*Gamma,T=T,mu=mu)
```

In our example both leads will be identical environments, we now use the
`approximate` method, to obtain `ExponentialFermionicEnviroment` 
representations of our environment which we will use as leads

```python
envL = env.approximate("pade",Nk=Nk1,tag="L") #left lead
envR = env.approximate("pade",Nk=Nk1,tag="R") #right lead
```

### Simulation

We now proceed to setup the heom solver, and find the solution for the steady
state

```python
HEOMPade = HEOMSolver(H, [(envL,d1),(envR,d2)], Ncc)  
rhoss, fullss= HEOMPade.steady_state()
expect(rhoss, d1.dag()*d1)
```

<!-- #region -->
We now use the steady state to construct the density of states, using the
Generator of the dynamics one may obtain the density the evolution of the 
correlator as proposed shown in 
[Li et al](https://arxiv.org/pdf/1207.6359.pdf):

$\langle d_{1}^{\dagger}(\tau) d_{1}(0) \rangle = 
Tr[d_{1}^{\dagger}(0) e^{\mathcal{L}\tau} \big(d_{1}(0) \rho_{ss}\big)]$

where we used the Quantum regression theorem. If we  now define the correlation
functions of the system

$C_{S}^{(+)}(t)=\langle d_{1}^{\dagger}(\tau) d_{1}(0) \rangle$ 

$C_{S}^{(-)}(t)=\langle d_{1}(\tau) d_{1}^{\dagger}(0)  \rangle$ 


And the retarded and advanced single particle Green functions

$G_{R}(t)=-i \theta(t) \left[ C_{S}^{(-)}(t) + C_{S}^{(+)}(-t)\right]$

$G_{A}(t)=i \theta(-t) \left[ C_{S}^{(-)}(t) + C_{S}^{(+)}(-t)\right]$

We can then write the spectral density of the system as 

$A(\omega)=\frac{i}{2 \pi} \int_{-\infty}^{\infty} dt 
e^{i \omega t} (G_{R}(t)-G_{A}(t))$

which we can rewrite in terms of the system correlation functions by 
using $C^{(+)}(t)=\overline{C}^{(+)}(-t)$ as

$A(\omega)=\frac{1}{\pi} \Re\left(\int_{0}^{\infty} dt 
e^{i \omega t} C_{S}^{(-)}(t)+ C_{S}^{(+)}(t) e^{-i \omega t}\right)$

To obtain this consider 

$C_{S}^{(\pm)}(w) = \int_{0}^{\infty} dt C_{S}^{(\pm)}(t) e^{ \mp i \omega t}
=\int_{0}^{\infty} dt \langle d_{1}^{(\pm)}(\tau) d_{1}^{\mp}(0) \rangle e^{\mp i \omega t}
=\int_{0}^{\infty} dt \langle \langle d_{1}^{(\pm)} | 
e^{ (L \mp i \omega) t} |(d_{1}^{\mp})_{ss} \rangle  \rangle
= - \langle \langle d_{1}^{(\pm)} | 
(L \mp i \omega)^{-1}|(d_{1}^{\mp})_{ss}\rangle  $

where we used $d_{1}^{(+)} = d_{1}^{\dagger}$ and $d_{1}^{(-)} = d_{1}$. 
Finally, we can write

$C_{S}^{(\pm)}(w) = \langle \langle d_{1}^{(\pm)} | X \rangle \rangle $

where the vector $| X \rangle \rangle$ is obtained by  solving the linear 
problem

$(\mathcal{L} \mp i \omega) X = (d_{1}^{(\mp)})_{ss} $


below there are some auxiliary functions that compute this quantities 
from the steady state and the HEOM generator  ($\mathcal{L}$)
(the right hand side of HEOMSolver). 
<!-- #endregion -->

```python

def prepare_matrices(result,fullss):
    """Prepares constant matrices to be used
    at each w"""
    L=sp.csr_matrix(result.rhs(0).full())
    sup_dim = result._sup_shape # size of vectorized Liouvillian
    ado_number  = result._n_ados # number of ADOS
    rhoss = fullss._ado_state.ravel() # flattened steady state 
    ado_identity = sp.identity(ado_number, format='csr')
    #d1 in the system+ADO space
    d1_big = sp.kron(ado_identity, sp.csr_matrix(spre(d1).full())) 
    d1ss = np.array(d1_big @ rhoss, dtype=complex)
    #d1dag in the system+ADO space
    d1dag_big = sp.kron(ado_identity, sp.csr_matrix(spre(d1.dag()).full()))
    d1dagss = np.array(d1dag_big @  rhoss, dtype=complex)
    # identity on system and Full space
    Is = sp.csr_matrix(np.eye(int(np.sqrt(sup_dim))).ravel().T)
    I = sp.identity(len(rhoss))
    return  Is,I,d1dagss,d1ss,L, d1dag_big, d1_big, sup_dim


def density_of_states(wlist,result,fullss):
    r"""
    Calculates $C_{S}^{(\pm)}(w)$
    Returns $2 \Re(C_{S}^{(-)}(w)+\overline{C}_{S}^{(+)}(w)))$
    """
    ddagd = []
    dddag = []
    Is,I,d1dagss,d1ss,L, d1dag, d1,sup_dim=prepare_matrices(result,fullss)
    for w in wlist:
        # Linear Problem for C_{s}^{(+)}
        x, _= lgmres((L-1.0j*w*I), d1ss,atol=1e-8)
        Cp1 = d1dag  @ x # inner product on ADOS
        Cp =  (Is @ Cp1[:sup_dim]) # inner product on system
        ddagd.append(Cp)
        # Linear Problem for C_{s}^{(-)}
        x, _= lgmres((L+1.0j*w*I),  d1dagss,atol=1e-8)
        Cm1 = d1 @ x # inner product on ADOS
        Cm =  (Is @ Cm1[:sup_dim]) # inner product on system
        dddag.append(Cm)
        
    return -2*(np.array(ddagd).flatten()+np.array(dddag).flatten()).real

```

We now proceed with the calculation

```python
wlist = np.linspace(-10,15,500)

ddos=density_of_states(wlist,HEOMPade,fullss)
```

Let us take a look at the density of states, we expect a peak around $w=0$
as in [Cirio et al](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033011)

```python
plt.plot(wlist,ddos,label=r"$N_k = 2$",linewidth=4)

plt.legend(fontsize=10)

plt.xlim(-10,15)
plt.yticks([0.,1,2],[0,1,2])
plt.xlabel(r"$\omega/\Gamma$",fontsize=20,labelpad=-10)
plt.ylabel(r"$2\pi \Gamma A(\omega)$ ",fontsize=20)

         
plt.show()
```

Why didn't we get the peak?  The answer is that we did the evolution without 
taking the parity of the state  into account,
 to clarify why this is wrong consider that the generator is now acting on

$Tr[d^{\dagger}(0) e^{\mathcal{L}\tau} \big(d(0) \rho_{ss}\big)]$

rather than the usual 

$ e^{\mathcal{L}\tau} \rho$

as the state of the system has even parity, when we apply creation
or annihilation operators on it we make it odd, and as such we need to evolve 
it with an odd parity generator (for more details on parity see
[this paper](https://arxiv.org/abs/2108.09094))

The `HEOMSolver` makes this easy for us. To take the odd parity of the operator
into account we just set the odd_parity argument to True on the HEOMSolver. 
We now repeat the calculations

```python
HEOMPadeOdd = HEOMSolver(H, [(envL,d1),(envR,d2)], Ncc,odd_parity=True)  
ddosOdd=density_of_states(wlist,HEOMPadeOdd,fullss)
```

```python
plt.plot(wlist,ddosOdd,label=r"$N_k = 2$ Odd Parity",linewidth=4)
plt.plot(wlist,ddos,"--",label=r"$N_k = 2$",linewidth=4)

plt.legend(fontsize=12)

plt.xlim(-10,15)
plt.yticks([0.,1,2],[0,1,2])
plt.xlabel(r"$\omega/\Gamma$",fontsize=20,labelpad=-10)
plt.ylabel(r"$2\pi \Gamma A(\omega)$ ",fontsize=20)

         
plt.show()
```

From [Cirio et al](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.5.033011)
we now that $N_{k}=2$ is not enough for convergence, we know repeat the 
calculation with $N_{k}=4$

```python
# On qutip master branch this is approx_by_pade instead of approximate
envL = env.approximate("pade",Nk=Nk2,tag="L") #left lead
envR = env.approximate("pade",Nk=Nk2,tag="R") #right lead
```

Again we start from the steady state

```python
HEOMPade4 = HEOMSolver(H, [(envL,d1),(envR,d2)], Ncc)  
HEOMPade4Odd = HEOMSolver(H, [(envL,d1),(envR,d2)], Ncc, odd_parity=True)  
rhoss4, fullss= HEOMPade4.steady_state()
expect(rhoss, d1.dag()*d1)
```

And repeat for odd parity

```python
ddos4Odd=density_of_states(wlist,HEOMPade4Odd,fullss)
```

Finally we visualize everything

```python
plt.plot(wlist,ddos4Odd,label=r"$N_k = 4$ Odd Parity",linewidth=4)
plt.plot(wlist,ddosOdd,"--",label=r"$N_k = 2$ Odd Parity",linewidth=4)
plt.legend(fontsize=12)

plt.xlim(-10,15)
plt.yticks([0.,1,2],[0,1,2])
plt.xlabel(r"$\omega/\Gamma$",fontsize=20,labelpad=-10)
plt.ylabel(r"$2\pi \Gamma A(\omega)$ ",fontsize=20)

         
plt.show()
```

We have used `QuTiP`'s `HEOMSolver` to generate the density of states, and 
visualized the Kondo Peak.

### About

```python
qutip.about()
```

### Testing

This section can include some tests to verify that the expected outputs are
generated within the notebook. We put this section at the end of the notebook,
so it's not interfering with the user experience. Please, define the tests
using `assert`, so that the cell execution fails if a wrong output is generated.

```python
assert np.allclose(expect(rhoss, d1.dag()*d1), 0.5)
assert np.allclose(expect(rhoss4, d1.dag()*d1), 0.5)
```
