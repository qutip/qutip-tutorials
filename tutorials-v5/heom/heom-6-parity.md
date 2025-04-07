---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor,destroy,qeye,spre,spost,operator_to_vector,expect,qeye
from qutip.solver.heom import FermionicBath,LorentzianPadeBath,HEOMSolver
import scipy.sparse as sp
from scipy.sparse.linalg import lgmres
```

<!-- #region -->
# Kondo  Peak

In this tutorial we will find the kondo peak for a system of two spins connected to two fermionic reservoirs 

The Hamiltonian of this setup is given by:

$H_{T}=H_{S}+H_{f}+H_{ef}$

Where $H_{S}$ is the system Hamiltonian, which is divided into three contributions. The electron system hamiltonian ($H_{e}$), the single-mode cavity ($H_{c}$) and the light matter coupling ($H_{ef}$):

$H_{S} = H_{e} + H_{ec}$


$H_{e}=  \sum_{n=g,e}\sum_{\sigma=\uparrow,\downarrow} \epsilon_{n} \hat{n}^{n \sigma} + U_{n} \hat{n}_{n \uparrow} \hat{n}_{n \downarrow} $

$H_{ec} = \sum_{\sigma=\uparrow,\downarrow} g_{ct} (d_{g\sigma}^{\dagger} d_{e\sigma} + d_{e\sigma}^{\dagger} d_{g\sigma} )(a^{\dagger}+a)$


The other two terms in the total Hamiltonian are given by

$H_{f}= \sum_{\alpha} \sum_{k} \epsilon_{\alpha,k} c^{\dagger}_{\alpha,k}c_{\alpha,k}$

and

$H_{ef}= \sum_{k} \sum_{\alpha=L,R} \sum_{\sigma=\uparrow,\downarrow} g_{\alpha,k} (c_{\alpha,k}^{\dagger} d_{g\sigma}+c_{\alpha,k} d_{g\sigma}^{\dagger})$


The interaction between the electronic systm and fermionic leads can be fully characterized by the Lorentzian spectral density

$J_{f_{\alpha}}(\omega)=\frac{1}{2 \pi} \frac{\Gamma_{\alpha} W_{f}^{2}}{(\omega-\mu_{\alpha})^{2}+W_{f}^{2}}$


<!-- #endregion -->

We first define some utility functions to construct the Hailtonian and  the system's Liouvillian supper operator

$\mathcal{L}(A)= - i [ H , A]$

```python
def _Is(i): return [qeye(2) for j in range(0, i)]
def _sm(N, i): return tensor(_Is(i) + [destroy(2)] + _Is(N - i - 1))
def _sz(N, i): return tensor(_Is(i) + [qeye(2)-2*destroy(2).dag() * destroy(2)] + _Is(N - i - 1))
def _oprd(lst): return np.prod(np.array(lst, dtype=object))
def _f(N, n): return _oprd([_sz(N, j) for j in range(n)])*_sm(N,n)
def liouvillian(H):
    return - 1j * (spre(H) - spost(H))
```

We define the system and bath parameters and  construct our fermionic baths using the `LorentzianPadeBath`  class from `Qutip`

```python
Ncc = 5
Nk=2
#########################
### system: two fermions
N = 2
d_1 = _f(N,0)
d_2 = _f(N,1)

#bath params:
mu = 0.  #chemical potential
Gamma  = 1  #coupling strenght
W = 2.5 #bath width

#system params:
#coulomb repulsion
U = 3 * np.pi * Gamma
#impurity energy
w0 = - U / 2.

beta = 1 / (0.2 * Gamma) # Inverse Temperature

Qops = [d_1.dag(),d_1,d_2.dag(),d_2] # coupling operators 

He = w0 *(d_1.dag() * d_1 + d_2.dag() * d_2) + U * d_1.dag() * d_1 * d_2.dag() * d_2

L = liouvillian(He) 

times = np.linspace(0,10,20000)
PadeBath=LorentzianPadeBath(Q=sum(Qops),gamma=2*Gamma,w=W,mu=mu,T=1/beta,Nk=Nk)
```

Since we have two leads, we now extract the coefficients of the bath to contruct a bath for each lead

```python
coeffsplus=PadeBath._corr(2*Gamma,W,mu,1/beta,Nk,sigma=1) #absorption coefficients
coeffsminus=PadeBath._corr(2*Gamma,W,mu,1/beta,Nk,sigma=-1) #emission coefficients
```

We then construct the baths from the lead coefficients

```python
bath1 = FermionicBath(d_1, coeffsplus[0], coeffsplus[1], coeffsminus[0], coeffsminus[1], tag ="Lead 1")
bath2 = FermionicBath(d_2, coeffsplus[0], coeffsplus[1], coeffsminus[0], coeffsminus[1], tag ="Lead 2")
```

At this point we obtain the steady state of the system and obtain the expectation value of $d_{1}^{\dagger} d_{1}$ which should be around $\frac{1}{2}$ for standard parameters

```python
resultHEOMPade = HEOMSolver(L, [bath1,bath2], Ncc)  #<---- normal parity HEOM to get normal steadystate
rhoss, fullss= resultHEOMPade.steady_state()
expect(rhoss, d_1.dag()*d_1)
```

We now use the steady state to construct the density of states, using the Generator of the dynamics and the quantum regression theorem one has:

$\langle d^{\dagger}(\tau) d(0) \rangle = Tr[d^{\dagger}(0) e^{\mathcal{L}\tau} \big(d(0) \rho_{ss}\big)]$

If we define 

$C_{R}(\tau)=\theta(\tau)(\langle d^{\dagger}(\tau) d(0) \rangle+\langle  d(0) d^{\dagger}(-\tau) \rangle)$

$C_{A}(\tau)=-\theta(\tau)(\langle d^{\dagger}(\tau) d(0) \rangle+\langle  d(0) d^{\dagger}(-\tau) \rangle)$

One can then write the density of states as 

$A(\omega)=\frac{1}{2 \pi} \int_{-\infty}^{\infty} dt e^{i \omega t} (C_{R}(t)+C_{A}(t))$

below there are some auxiliary functions that compute this quantities from the steady state of the system and the generator (the right hand side of HEOM). 

```python

wlist = np.linspace(-15,15,100)
def prepare_matrices(result,fullss):
    M_1 = result.rhs
    sup_dim = result._sup_shape
    N_he  = result._n_ados
    rhoptemp = fullss._ado_state.reshape((N_he*sup_dim, 1))
    unit_helems = sp.identity(N_he, format='csr')
    M2=sp.csr_matrix(M_1.to_list()[0].full())
    ###################################
    d_1_big = sp.kron(unit_helems, sp.csr_matrix(spre(d_1).full()))
    rho0d1 = np.array(d_1_big @ rhoptemp, dtype=complex)
    # ######################################
    d_1_bigdag = sp.kron(unit_helems, sp.csr_matrix(spre(d_1.dag()).full()))
    rho0d1dag = np.array(d_1_bigdag @  rhoptemp, dtype=complex)
    I = tensor([qeye(n) for n in d_1.dims[0]])
    I_vec3 = sp.csr_matrix(operator_to_vector(I).full().T)
    Nfull = N_he*sup_dim
    c_I = sp.identity(Nfull)
    return  I_vec3,c_I,rho0d1dag,rho0d1,sup_dim,M2, d_1_bigdag, d_1_big
def D2(w,M2,c_I):
    return (M2-1.0j*w*c_I)

def D3(w,M2,c_I):
    return (1.0j*w*c_I+M2)

def density_of_states(wlist,result,fullss):
    ddagd = []
    dddag = []
    I_vec3,c_I,rho0d1dag,rho0d1,sup_dim,M2, d_1_bigdag, d_1_big=prepare_matrices(result,fullss)
    for idx in range(len(wlist)):
        w=wlist[idx]
        x2, _= lgmres(D2(w,M2,c_I), rho0d1,atol=1e-8)
        Cw21 = d_1_bigdag  @ x2
        Cw22 =  (I_vec3 @ Cw21[:sup_dim])
        ddagd.append(Cw22)
        
        x3, _= lgmres(D3(w,M2,c_I),  rho0d1dag,atol=1e-8)
        Cw31 = d_1_big @ x3
        Cw32 =  (I_vec3 @ Cw31[:sup_dim])
        dddag.append(Cw32)
        
    return -2*(np.array(ddagd).flatten()+np.array(dddag).flatten())
```

```python
ddos=density_of_states(wlist,resultHEOMPade,fullss)
```

### We can now visualize our density of states

```python

plt.rcParams["figure.figsize"] = (12,10)
plt.plot(wlist,np.real(ddos),linestyle='--',label=r"HEOM Padé $N_k = 2$",linewidth=4)
plt.legend(fontsize=20,loc = (0.23,0.7))

plt.xlim(-10,15)
plt.yticks([0.,1,2],[0,1,2])
plt.xlabel(r"$\omega/\Gamma$",fontsize=28,labelpad=-10)
plt.ylabel(r"$2\pi \Gamma A(\omega)$ ",fontsize=28)

         
plt.show()
```

Notice we don't see any peak as the RHS used for evolution without taking the parity into account, to clarify why this is wrong consider that the generator is now acting as

$Tr[d^{\dagger}(0) e^{\mathcal{L}\tau} \big(d(0) \rho_{ss}\big)]$

rather than the usual 

$ e^{\mathcal{L}\tau} \rho$

as the state of the system has even parity, when we apply creation or anihilation operatos on it we make it odd, and as such we need to evolve it with an odd parity generator

To take the parity into account we just set the odd_parity argument to True on the HEOMSolver, we now repeat the calculations

```python

resultHEOMPade2 = HEOMSolver(L, [bath1,bath2], Ncc, odd_parity=True) #<------ use ODD parity 
ddos=density_of_states(wlist,resultHEOMPade2,fullss) # recalculate density of states with odd parity
```

```python


plt.rcParams["figure.figsize"] = (12,10)
plt.plot(wlist,np.real(ddos),linestyle='--',label=r"HEOM Padé $N_k = 2$",linewidth=4)
plt.legend(fontsize=20,loc = (0.23,0.7))

plt.xlim(-10,15)
plt.yticks([0.,1,2],[0,1,2])
plt.xlabel(r"$\omega/\Gamma$",fontsize=28,labelpad=-10)
plt.ylabel(r"$2\pi \Gamma A(\omega)$ ",fontsize=28)

         
plt.show()
```
