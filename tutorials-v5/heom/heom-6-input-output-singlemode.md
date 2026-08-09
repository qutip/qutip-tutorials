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

<!-- #region -->
# Input-Output HEOM example: single mode limit
Authors: Neill Lambert (nwlambert@gmail.com) and Mauro Cirio (mauro.cirio@gmail.com)


## Introduction
Here we demonstrate how to use the input-output HEOM model (https://arxiv.org/abs/2408.12221) in the extreme limit of a bath consisting of a single damped mode (which we term the single harmonic oscillator bath).
It is not a limit where the HEOM is particularly useful per se, but it allows us to compare the numerical solution of the equivalent damped Rabi model,
and permits extremely simple bath correlation functions.

<!-- #endregion -->

```python
import numpy as np
from matplotlib import pyplot as plt
from qutip import about, basis, destroy, expect, mesolve, qeye, sigmax, sigmaz, tensor
from qutip.solver.heom import BathExponent, BosonicBath, HEOMSolver, InputOutputBath
```

<!-- #region -->
## SHO Bath
To be precise, our system is a single qubit coupled to a bath whose spectral density is the (unphysical) lorentzian

$J(\omega) = \frac{\lambda^2}{(\omega^2-\omega_0^2) + \Gamma^2}$

This is unphysical from the continuum bath perspective as it has support on negative frequencies. Nevertheless, the standard HEOM derivation applies, and is known to then reproduce the physics of the locally-damped Rabi model.


The two-time correlation functions of this bath, at zero-temperature, are

$C(t) = \langle X(t)X(0) \rangle = \lambda^2 e^{-i\omega_0 t-\Gamma |t|}$.

In standard HEOM, we can manually input these parameters, alongside the system Hamiltonian and coupling operator:

<!-- #endregion -->

```python
# correlation functions of single damped HO


def sho_params(lam, Gamma, Om):
    """Calculation of the real and imaginary expansions of the
    damped sho
    """
    factor = 1 / 2
    ckAR = [
        (factor * lam**2),
        (factor * lam**2),
    ]

    vkAR = [
        -1.0j * Om + Gamma,
        1.0j * Om + Gamma,
    ]

    ckAI = [
        -factor * lam**2 * 1.0j,
        factor * lam**2 * 1.0j,
    ]

    vkAI = [
        -(-1.0j * Om - Gamma),
        -(1.0j * Om - Gamma),
    ]

    return ckAR, vkAR, ckAI, vkAI
```

```python
# Defining the system Hamiltonian


Del = 2 * np.pi * 1.0
Hsys = 0.5 * Del * sigmaz()
```

```python
# Initial state of the system.
rho0 = basis(2, 0) * basis(2, 0).dag()
```

```python
# System-bath coupling (underdamed spectral density)
Q = sigmax()  # coupling operator

# Bath properties:
Gamma = 0.1 * Del  # cut off frequency
lam = 0.1 * Del  # coupling strength
Om = 0.2 * Del  # resonance frequency

ckAR, vkAR, ckAI, vkAI = sho_params(lam=lam, Gamma=Gamma, Om=Om)

# HEOM parameters:
# Number of levels of the hierarchy to retain:
NC = 12

# Times to solve for:
tlist = np.linspace(0, 20 / Del, 100)

options = {
    "store_ados": True,
}


bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI, tag="SHO")
```

<!-- #region -->
## Input bath

At this point we could run the standard HEOM solver and compare system properties to that of a normal damped Rabi model (see below). 
To consider input and output properties however, we have to add some additional auxiliary input output baths.

First, we will consider 'input' consisting of creating one photon in the bath at t=0.  This requires us to define some additional correlation functions between the operators applied to create that photon, and the operators we use to define the bath coupling operator $X(t)$.

To create a photon at $t=0$ at frequency $\omega_0$  one would normally apply operators to the initial bath state that act from the left and right like $b^{\dagger}_{\omega_0} \rho(t=0) b_{\omega_0}$.  Note that our justification of this operator $b_{\omega_0}$ and the resulting correlation functions in the following is not very rigorous, one can more formally define them as sum over 'bath' modes, but we avoid that level of rigor here.

In the input-output HEOM formalism this means we need to define two (or more) correlation functions:

$ G^{(1)}_{\alpha}(t) = \mathrm{Tr}[X(t)^{\alpha}b_{\omega_0}^{\dagger^L}(t=0)] $
and
$ G^{(2)}_{\alpha}(t) = \mathrm{Tr}[X(t)^{\alpha}b_{\omega_0}^{R}(t=0)] $


where $\alpha \in L/R$, indicates whether the super-operator form of the operator acts from the left (L) or right (R).

Note that for input, because the operators act on the underlying equilibrium state $\rho(t=0)$ before the coupling operator $X$, the role of $\alpha$ in $X$ is inconsequential. In other words, the correlation functions do not depend on $\alpha$.  This is not the case for the output, as we will see below.

Since here we assume that the underlying equilibrium state for the bath is the vacuum state (upon which the input is created), we can evaluate these two functions simply as

$ G^{(1)}_{\alpha}(t) = \lambda e^{-i\omega_0 t - \Gamma |t|}$
and
$ G^{(2)}_{\alpha}(t) =  \lambda e^{i\omega_0 t - \Gamma |t|} $

In general, these correlations can be any time-dependent function.  It just so happens that in this simple case they have this simple exponential form. Unlike a normal HEOM bath, we give these in their full functional form, and not in terms of the parameters of an exponential decomposition;  in fact such a decomposition is generally not needed for neither input or output fields, though it can be employed for output fields.
<!-- #endregion -->

```python
def input1(t):
    return lam * np.exp(-1.0j * Om * t - Gamma * t)


def input2(t):
    return lam * np.exp(1.0j * Om * t - Gamma * t)


ck_in = [input1, input2]
```

```python
# Solver options:
bath_input = InputOutputBath(Q, ck_input=ck_in, tag="input")

SHO_model = HEOMSolver(Hsys, [bath, bath_input], NC, options=options)

resultSHO = SHO_model.run(rho0, tlist)
```

## Conditioned state results

To obtain information about the state conditioned on the input we must do some processing on the auxiliary density operators defined by the input.  In fact, input-output requires an extension of the regular HEOM so that, essentially, just looking at the normal 'system' state returned by the HEOM ignores the input entirely.  On the other hand, one can define quantities like

$ \rho_s(t)_{input} = \mathrm{Tr_B}[V(t)b_{\omega_0}^{R}b_{\omega_0}^{\dagger^L}\rho_T(t=0)].$

which one can interpret as conditional states of the system given certain operators on the bath.

This conditional state is given by a function of the unconditioned state and the input ados:

$ \rho_s(t)_{input} = \rho^{(0,0,0,0)} (t) - \rho^{(0,0,1,1)}(t)$

where $\rho^{(0,0,0,0)} (t)$ is the system given no input to the bath, and $\rho^{(0,0,1,1)}(t)$ is the ADOs associated with the input.


To obtain the ADOs, we use the result object ``ado_states`` property, and use the ``filter`` method the select the ADOs we need. We want those tagged as ``input`` baths (there are two, we need to mention the tag twice) and at what 'level' we want them at, in this we want (0,0,1,1), which has a total level of 2 (level can be understood as ''total number of excitations''' in the heom indices).

```python
# construct the conditional system state from ADOs

result_input = []

for t in range(len(tlist)):
    label = resultSHO.ado_states[t].filter(level=2, tags=["input", "input"])

    state = resultSHO.ado_states[t].extract(label[0])

    result_input.append(expect(resultSHO.states[t], sigmaz()) - expect(state, sigmaz()))
```

### Benchmark against expected results 

Lets check this case by simulating the equivalent single-mode Rabi model using Mesolve():

```python
options = {
    "store_states": True,
    "progress_bar": "enhanced",
}
Nbos = 12
a = qeye(2) & destroy(Nbos)
H = Hsys & qeye(Nbos)

H = H + lam * tensor(Q, qeye(Nbos)) * (a + a.dag()) + Om * a.dag() * a
# no photons:
resultME0 = mesolve(
    H,
    rho0 & (basis(Nbos, 0) * basis(Nbos, 0).dag()),
    tlist,
    [np.sqrt(Gamma * 2) * a],
    options=options,
)
# one photon initial condition:
resultME1 = mesolve(
    H,
    rho0 & (basis(Nbos, 1) * basis(Nbos, 1).dag()),
    tlist,
    [np.sqrt(Gamma * 2) * a],
    options=options,
)
```

```python
plt.figure()
plt.plot(
    tlist,
    expect(resultME0.states, sigmaz() & qeye(Nbos)),
    "-",
    label=r"mesolve no photons ",
)
plt.plot(
    tlist,
    expect(resultME1.states, sigmaz() & qeye(Nbos)),
    "-",
    label=r"mesolve one photon",
)
plt.plot(
    tlist, expect(resultSHO.states, sigmaz()), "r--", alpha=1, label=r"heom no photons"
)
plt.plot(tlist, result_input, "b--", alpha=1, label=r"heom one photons input")

plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=18)
plt.legend()
plt.show()
```

## Output bath with exponential decomposition (spectral decomposition)
Moving onto the output case, we assume we want to measure the occupation of the bath at some specific frequency, i.e., $\text{Tr}[b_{\omega_0}^{\dagger}(t_\text{out})b_{\omega_0}(t_\text{out})\rho(t)]$. In order to compute this quantity, we need to consider the additional correlation functions between the environmental coupling operator and the output operators. 

We find there are two approaches one can take, with benefits and drawbacks.  The first approach is to upgrade the output time to be the same as the "dynamical" one, i.e., the one tracking the HEOM dynamics: $t_\text{out}=t$. In this case, the correlations are

$G_{1(out)}^L(t)= \langle b_{\omega_0}^{\dagger}(t) X^L(0)\rangle  = 0$

$G_{1(out)}^R(t)=\langle b_{\omega_0}^{\dagger}(t)X^R(0)\rangle = \lambda e^{i\omega_0 t - \Gamma |t|}$

$G_{2(out)}^L = \langle b_{\omega_0}(t) X^L(0)\rangle= \lambda e^{-i\omega_0 t - \Gamma|t|}$

$G_{2(out)}^R= \langle b_{\omega_0}(t) X^R(0)\rangle = 0$

Note that the left/right symmetry is broken, as the bath operator $X$ is applied first, so that we need to keep track of all the 4 different correlation functions above.  Secondly, while trivial in this case, usually a spectral decomposition for these correlations would be needed. However, this is not always necessary as it is also possible to treat input and output on equal footing, as we will show in a later section.

First however, continuing with the spectral form, we define this bath with the new variables in an input output bath object:

```python
ck_output_L = [lam]  # field 2
ck_output_R = [lam]  # field 1
vk_output_L = [1.0j * Om + Gamma]  # field 2
vk_output_R = [-1.0j * Om + Gamma]  # field 1
```

```python
# Solver options:

# Number of levels of the hierarchy to retain:
NC = 12

options = {
    "nsteps": 15000,
    "store_states": True,
    "progress_bar": "enhanced",
    "store_ados": True,
}

bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)

ck_in_1 = [input1]
ck_in_2 = [input2]

bath_input_1 = InputOutputBath(Q, ck_input=ck_in_1, tag="input1")
bath_input_2 = InputOutputBath(Q, ck_input=ck_in_2, tag="input2")

bath_output_1R = InputOutputBath(
    Q, ck_output_R=ck_output_R, vk_output_R=vk_output_R, tag="output1"
)  # only right terms needed here, see above
bath_output_2L = InputOutputBath(
    Q, ck_output_L=ck_output_L, vk_output_L=vk_output_L, tag="output2"
)  # only left terms needed here, see above

SHO_model = HEOMSolver(
    Hsys,
    [bath, bath_output_1R, bath_output_2L, bath_input_1, bath_input_2],
    NC,
    options=options,
)

resultSHO = SHO_model.run(rho0, tlist)
```

Now, if we only need the output not conditioned on input, we can simply evaluate the quantity $\rho^{(0,0,1,1,0,0)}$

```python
result_output = []

for t in range(len(tlist)):
    labels = resultSHO.ado_states[t].filter(
        level=2, types=[BathExponent.types.Output_R, BathExponent.types.Output_L]
    )
    states = []
    for label in labels:
        states.append(resultSHO.ado_states[t].extract(label))

    result_output.append(-np.sum([state.tr() for state in states]))
```

```python
plt.figure()
plt.plot(tlist, expect(resultME0.states, a.dag() * a), "-", label=r"mesolve0 ")
plt.plot(tlist, result_output, "b--", alpha=1, label=r"heom1")
plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle a^{\dagger}a\rangle$", fontsize=18)
plt.legend()

plt.show()
```

## Correlated input-output functions


We finally arrived to the most general input-output case. To get the output conditioned on the input, we need some additional information:  the cross correlations between input and output operators (of the free bath)!  We can evaluate these as before, keeping in mind we essentially have two input and two output ''operators'' to process.  Recall we are working with, for input,  $b^{\dagger^L}_{\omega_0}(0)$, and $b^R_{\omega_0}(0)$, while for output $b^{\dagger^L}_{\omega_0}(t)$ and $b^L_{\omega_0}(t)$.  We index these functions with $(ij,kl)$, with $ij$ for the two output operators, and $kl$ the presence of the two input operators, omit the $\omega_0$ subscript, and indicate they are calculated for the free bath (without presence of system), with the subscript $f$:

$ G_{io}(t)^{(11,11)} = \langle b^{\dagger^L}(t)b^L(t)b^{\dagger}_L(0)b_R(0)\rangle_f = e^{-2  \Gamma |t|}$

$ G_{io}(t)^{(11,00)} = \langle b^{\dagger^L}(t)b^L(t)\rangle_f = 0$

$ G_{io}(t)^{(10,10)} = \langle b^{\dagger^L}(t)b^{\dagger^L}(0)\rangle_f = 0$

$ G_{io}(t)^{(10,01)} = \langle b^{\dagger^L}(t)b^R(0)\rangle_f = e^{1.0j \omega_0 t - \Gamma |t|}$

$ G_{io}(t)^{(01,10)} = \langle b^{L}(t)b^{\dagger^L}(0)\rangle_f = e^{-1.0j \omega_0 t - \Gamma |t|}$

$ G_{io}(t)^{(01,01)} = \langle b^{\dagger^L}(t)b^R(0)\rangle_f = 0$

$ G_{io}(t)^{(00,11)} = \langle b^{\dagger^L}(0)b^R(0)\rangle_f = 1$

The expression for the conditional output, in terms of ADOs and these functions (keeping just nonzero terms), is

$ \langle b^{\dagger^L}(t)b^L(t)b^{\dagger}_L(0)b_R(0)\rangle =  G_{io}(t)^{(11,11)} \rho^{(00,00,00)} + \rho^{(00,11,11)} $
$-G_{io}(t)^{(10,01)} \rho^{(00,01,10)} -G_{io}(t)^{(01,10)}\rho^{(00,10,01)} - G_{io}(t)^{(00,11)} \rho^{(00,11,00)}$

Here we group the ADO indices so the first two are for the 'regular HEOM' exponents, the middle two for the output exponents, and the last two for the input exponents. 



```python
result_output = []
for t in range(len(tlist)):
    label = resultSHO.ado_states[t].filter(level=2, tags=["output2", "input1"])

    s0110 = (
        np.exp(1.0j * Om * tlist[t] - Gamma * tlist[t])
        * resultSHO.ado_states[t].extract(label[0]).tr()
    )

    label = resultSHO.ado_states[t].filter(level=2, tags=["output1", "input2"])

    s1001 = (
        np.exp(-1.0j * Om * tlist[t] - Gamma * tlist[t])
        * resultSHO.ado_states[t].extract(label[0]).tr()
    )

    label = resultSHO.ado_states[t].filter(level=2, tags=["output1", "output2"])

    s1100 = resultSHO.ado_states[t].extract(label[0]).tr()

    label = resultSHO.ado_states[t].filter(
        level=4, tags=["output1", "output2", "input1", "input2"]
    )

    s1111 = resultSHO.ado_states[t].extract(label[0]).tr()

    result_output.append(
        resultSHO.states[t].tr() * np.exp(-2.0 * Gamma * tlist[t])
        - s0110
        - s1001
        - s1100
        + s1111
    )
```

```python
plt.figure()
plt.plot(tlist, expect(resultME1.states, a.dag() * a), "-", label=r"mesolve1 ")
plt.plot(
    tlist, result_output, "b--", alpha=1, label=r"heom1 dynamic fields/spectral decomp"
)
plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle a^{\dagger}a\rangle$", fontsize=18)
plt.legend()
plt.show()
```

## Output bath with time-dependent functions

Importantly, as mentioned several times, we can also capture the output in the same way we capture the input;  with time-dependent functions modeling an observable evaluated at a specific time $t_{out}$.  This possibility is to be expected from the symmetry of input and output in this formalism; they are both simply defined as the application of an operator at some particular time, it just so happens that input is applied at $t=0$.

The advantage is clear in more complex cases, when capturing the correlations for the output with exponential decomposition can be difficult.  In this case, with a SHO bath which naturally decomposes into just a few exponents, it is rather the opposite, and adds some unnecessary computational complexity, but serves as a useful testbed.

<!-- #region -->
In this case, we need to slightly modify the definitions of the correlation functions we need for the output;  previously we used 


$G_{1(out)}^L(t)= \langle b_{\omega_0}^{\dagger}(t) X^L(0)\rangle  = 0$

$G_{1(out)}^R(t)=\langle b_{\omega_0}^{\dagger}(t)X^R(0)\rangle = \lambda e^{i\omega_0 t - \Gamma |t|}$

$G_{2(out)}^L = \langle b_{\omega_0}(t) X^L(0)\rangle = \lambda e^{-i\omega_0 t- \Gamma |t|}$

$G_{2(out)}^R= \langle b_{\omega_0}(t) X^R(0)\rangle = 0$



<!-- #endregion -->

<!-- #region -->
Now we need to define


$G_{1(out)}^L(t)= \langle b_{\omega_0}^{\dagger}(t_{out}) X^L(t)\rangle = 0$

$G_{1(out)}^R(t)=\langle b_{\omega_0}^{\dagger}(t_{out})X^R(t)\rangle = \lambda e^{i\omega_0 (t_{out}-t) - \Gamma(t)(t_{out}-t)}$

$G_{2(out)}^L = \langle b_{\omega_0}(t_{out}) X^L(t)\rangle = \lambda e^{-i\omega_0 (t_{out}-t) - \Gamma(t)(t_{out}-t)}$

$G_{2(out)}^R= \langle b_{\omega_0}(t_{out}) X^R(t)\rangle = 0$

<!-- #endregion -->

```python
# Number of levels of the hierarchy to retain:
NC = 12

options = {
    "nsteps": 15000,
    "store_states": True,
    "progress_bar": False,
    "store_ados": True,
}

result_output = []


def g1R(t, args):
    return lam * np.exp(1.0j * Om * (args["tout"] - t) - Gamma * (args["tout"] - t))


def g2L(t, args):
    return lam * np.exp(-1.0j * Om * (args["tout"] - t) - Gamma * (args["tout"] - t))


bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)

ck_in_1 = [input1]
ck_in_2 = [input2]

bath_input_1 = InputOutputBath(Q, ck_input=ck_in_1, tag="input1")
bath_input_2 = InputOutputBath(Q, ck_input=ck_in_2, tag="input2")

bath_output_1R = InputOutputBath(Q, ck_output_fn_R=[g1R], tag="output1")
bath_output_2L = InputOutputBath(Q, ck_output_fn_L=[g2L], tag="output2")
args = {"tout": 1}
SHO_model = HEOMSolver(
    Hsys,
    [bath, bath_output_1R, bath_output_2L, bath_input_1, bath_input_2],
    NC,
    options=options,
    args=args,
)

for tout in tlist:

    t_list = np.linspace(0, tout, 2)
    args = {"tout": tout}
    resultSHO = SHO_model.run(rho0, t_list, args=args)

    labels = resultSHO.ado_states[-1].filter(level=2, tags=["output2", "input1"])

    for label in labels:
        s0110 = (
            np.exp(1.0j * Om * tout - Gamma * tout)
            * resultSHO.ado_states[-1].extract(label).tr()
        )

    labels = resultSHO.ado_states[-1].filter(level=2, tags=["output1", "input2"])

    for label in labels:
        s1001 = (
            np.exp(-1.0j * Om * tout - Gamma * tout)
            * resultSHO.ado_states[-1].extract(label).tr()
        )

    labels = resultSHO.ado_states[-1].filter(level=2, tags=["output1", "output2"])
    for label in labels:
        s1100 = resultSHO.ado_states[-1].extract(label).tr()

    labels = resultSHO.ado_states[-1].filter(
        level=4, tags=["output1", "output2", "input1", "input2"]
    )

    for label in labels:
        s1111 = resultSHO.ado_states[-1].extract(label).tr()

    result_output.append(
        resultSHO.states[-1].tr() * np.exp(-2.0 * Gamma * tout)
        - s0110
        - s1001
        - s1100
        + s1111
    )
```

```python
plt.figure()
plt.plot(tlist, expect(resultME1.states, a.dag() * a), "-", label=r"mesolve1 ")
plt.plot(
    tlist, np.real(result_output), "b--", alpha=1, label=r"heom1 static fields/TD func"
)
plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle a^{\dagger}a\rangle$", fontsize=18)
plt.legend()

plt.show()
```

```python
about()
```

```python
assert np.allclose(result_output, expect(resultME1.states, a.dag() * a), atol=1e-5)
```

```python

```
