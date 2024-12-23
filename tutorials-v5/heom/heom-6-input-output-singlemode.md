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
import qutip
from matplotlib import pyplot as plt
from qutip import basis, destroy, expect, mesolve, qeye, sigmax, sigmaz, tensor
from qutip.solver.heom import (BathExponent, BosonicBath, HEOMSolver,
                               InputOutputBath)

%matplotlib inline
```

<!-- #region -->
## SHO Bath
To be precise, our system is a single qubit coupled to a bath whose spectral density is the (unphysical) lorentzian

$J(\omega) = \frac{\lambda^2}{(\omega^2-\omega_0^2) + \Gamma^2}$

This is unphysical from the continuum bath perspective as it has support on negative frequencies. Nevertheless, the standard HEOM derivation applies, and is known to then reproduce the physics of the locally-damped Rabi model.


The two-time correlation functions of this bath, at zero-temperature, are

$C(t) = <X(t)X(0)> = \lambda^2 e^{i\omega_0 t-\Gamma |t|}$.

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

## Input bath

At this point we could run the standard HEOM solver and compare system properties to that of a normal damped Rabi model (see below). 
To consider input and output properties however, we have to add some additional auxiliary input output baths.

First, we will consider 'input' consisting of creating one photon in the bath at t=0.  This requires us to define some additional correlation functions between the operators applied to create that photon, and the operators we use to define the bath coupling operator $X(t)$.

To create a photon at $t=0$ at frequency $\omega_0$  one would normally apply operators to the initial bath state that act from the left and right like $b^{\dagger}_{\omega_0} \rho(t=0) b_{\omega_0}$.  Note that our justification of this operator $b_{\omega_0}$ and the resulting correlation functions in the following is not very rigorous, one can more formally define them as sum over 'bath' modes, but we avoid that level of rigor here.

In the input-output HEOM formalism this means we need to define two (or more) functions:

$$ G^{(1)}_{\alpha}(t) = \mathrm{Tr}[X(t)^{\alpha}b_{\omega_0}^{\dagger^L}(t=0)] $$
and
$$ G^{(2)}_{\alpha}(t) = \mathrm{Tr}[X(t)^{\alpha}b_{\omega_0}^{R}(t=0)] $$

where also $\alpha \in L/R$, and $L/R$ indicates whether the super-operator form of the operator acts from the left or right.

Note that for input, becauase the oprators act before $X$, the role of $\alpha$ in $X$ is inconsequential. Thus for each alpha, the function is the same.  This is not the case for output, as we will see below.  

Recalling that we assume here the bath is initially in its vacuum state, we can evaluate these two functions simply as

$$ G^{(1)}_{\alpha}(t) = \lambda e^{-i\omega_0 t - \Gamma |t|}$$
and
$$ G^{(2)}_{\alpha}(t) =  \lambda e^{i\omega_0 t - \Gamma |t|} $$

Unlike a normal HEOM bath we give these in a functional form, not as a parameters to an exponential decomposition;  in fact such a decomposition is generally not needed for input or output fields, though can be employed for output fields. Generally these functions can be any time-dependent function.  It just so happens that in this simple case they have this simple exponential form.

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

To obtain information about the state conditioned on the input we must do some processing on the auxiliary density operators defined by the input.  Essentially, just looking at the normal 'system' state returned by the HEOM ignores the input entirely.  Looking at the ADOS on the other hand, defines quantities like  

$$ \rho_s(t)_{input} = \mathrm{Tr_B}[V(t)b_{\omega_0}^{R}b_{\omega_0}^{\dagger^L}\rho_T(t=0)].$$

which one can interpret as conditional states of the system given certain operators on the bath.

This conditional state is given by a function of the unconditioned state and the input ados:

$$ \rho_s(t)_{input} = \rho^{(0,0,0,0)} (t) - \rho^{(0,0,1,1)}(t)$$

where $\rho^{(0,0,0,0)} (t)$ is the system given no input to the bath, and $\rho^{(0,0,1,1)}(t)$ is the ADOs associated with the input.


To obtain the ADOs, we use the result object ``ado_states`` property, and use the ``filter`` method the select the ADOs we need. We want those tagged as ``input`` baths (there are two, we need to mention the tag twice) and at what 'level' we want them at, in this we want (0,0,1,1), which has a total level of 2 (level can be understood as ''total number of excitations''' in the heom indicies).

```python
# construct the conditional system state from ADOs

result_input = []

for t in range(len(tlist)):
    label = resultSHO.ado_states[t].filter(level=2, tags=["input",
                                                          "input"])

    state = resultSHO.ado_states[t].extract(label[0])

    result_input.append(expect(resultSHO.states[t], sigmaz()) -
                        expect(state, sigmaz()))
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
    tlist, expect(resultSHO.states, sigmaz()), "r--", alpha=1,
    label=r"heom no photons"
)
plt.plot(tlist, result_input, "b--", alpha=1,
         label=r"heom one photons input")

plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle \sigma_z \rangle$", fontsize=18)
plt.legend()
plt.show()
```

## Output bath with exponential decomposition (spectral decomposition)
Moving onto conditional output, we assume we want to measure the occupation of the bath:

This also requires defining some additional correlation functions with the output observables and the free-bath evolution. We find there are two approaches one can take, with benefits and drawbacks.  

Since we are defining the time of observation occuring 'at the end' of when we solve the HEOM, i.e., output, the first option we have is to define functions like $<b_{\omega_0}^{\dagger}(t)b_{\omega_0}(t)>$ as the observables, which requires we define

$$G_{1(out)}^L(t)= <b_{\omega_0}^{\dagger}(t) X^L(0)> = 0$$

$$G_{1(out)}^R(t)<b_{\omega_0}^{\dagger}(t)X^R(0)> = \lambda e^{i\omega_0 t - \Gamma |t|}$$

$$G_{2(out)}^L = <b_{\omega_0}(t) X^L(0)> = \lambda e^{-i\omega_0 t - \Gamma|t|}$$

$$G_{2(out)}^R= <b_{\omega_0}(t) X^R(0)> = 0$$

Note that the left/right symmetry is broken, as the bath operator $X$ is applied first, and thus we split these output funcitons into 4 terms.  Secondly, here we can perform a spectral decomposition into exponents (which is convenient in this case, as it is how they appear naturally).  However, it turns out this is not always neccessary, and we can treat intput and output on equal footing, which we will show in a moment.

First however, continuing with the spectral form, we define this bath with the new variables in an intput output bath object:

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

Now, if we only need the output, not conditioned on input, it is still available by suitably processing the ADOs.  In this case we need just the ADOs assoicated with the output, $\rho^{(0,0,0,0,1,1)}$

```python
result_output = []

for t in range(len(tlist)):
    labels = resultSHO.ado_states[t].filter(
        level=2, types=[BathExponent.types.Output_R,
                        BathExponent.types.Output_L]
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


To get the output conditioned on the input, we need some additional information;  correlations between input and output operators (of the free bath)!  We can valuate these as before, keeping in mind we essentially have two input and two output ''operators'' to process.  Recall we are working with, for input,  $b^{\dagger^L}_{\omega_0}(0)$, and $b^R_{\omega_0}(0)$, while for output $b^{\dagger^L}_{\omega_0}(t)$ and $b^L_{\omega_0}(t)$.  We index these functions with $(ij,kl)$, with $ij$ for the two output operators, and $kl$ the presence of the two input operators, omit the $\omega_0$ subscript, and indicate they are calculated for the free bath (without prescence of system), with the subscript $f$:
$$ G_{io}(t)^{(11,11)} = <b^{\dagger^L}(t)b^L(t)b^{\dagger}_L(0)b_R(0)>_f = e^{-2  \Gamma |t|}$$
$$ G_{io}(t)^{(11,00)} = <b^{\dagger^L}(t)b^L(t)>_f = 0$$
$$ G_{io}(t)^{(10,10)} = <b^{\dagger^L}(t)b^{\dagger^L}(0)>_f = 0$$
$$ G_{io}(t)^{(10,01)} = <b^{\dagger^L}(t)b^R(0)>_f = e^{1.0j \omega_0 t - \Gamma |t|}$$
$$ G_{io}(t)^{(01,10)} = <b^{L}(t)b^{\dagger^L}(0)>_f = e^{-1.0j \omega_0 t - \Gamma |t|}$$
$$ G_{io}(t)^{(01,01)} = <b^{\dagger^L}(t)b^R(0)>_f = 0$$
$$ G_{io}(t)^{(00,11)} = <b^{\dagger^L}(0)b^R(0)>_f = 1$$

The expression for the conditional output, in terms of ADOs and these functions (keeping just nonzero terms), is

$$ <b^{\dagger^L}(t)b^L(t)b^{\dagger}_L(0)b_R(0)> =  G_{io}(t)^{(11,11)} \rho^{(00,00,00)} + \rho^{(00,11,11)} $$ 
$$-G_{io}(t)^{(10,01)} \rho^{(00,01,10)} -G_{io}(t)^{(01,10)}\rho^{(00,10,01)} - G_{io}(t)^{(00,11)} \rho^{(00,11,00)}$$

Here we grop the ADO incices so the first two are for the 'normal heom' exponents, the middle two for the output exponents, and the last two for the input exponents. Note that in this function the correlations $G$ multiply ADOs with  which they are adjoint.



```python
result_output = []
for t in range(len(tlist)):
    label = resultSHO.ado_states[t].filter(level=2, tags=["output2",
                                                          "input1"])

    s0110 = (
        np.exp(1.0j * Om * tlist[t] - Gamma * tlist[t])
        * resultSHO.ado_states[t].extract(label[0]).tr()
    )

    label = resultSHO.ado_states[t].filter(level=2, tags=["output1",
                                                          "input2"])

    s1001 = (
        np.exp(-1.0j * Om * tlist[t] - Gamma * tlist[t])
        * resultSHO.ado_states[t].extract(label[0]).tr()
    )

    label = resultSHO.ado_states[t].filter(level=2, tags=["output1",
                                                          "output2"])

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
    tlist, result_output, "b--", alpha=1,
    label=r"heom1 dynamic fields/spectral decomp"
)
plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle a^{\dagger}a\rangle$", fontsize=18)
plt.legend()
plt.show()
```

## Output bath with time-dependent functions

Importantly, as mentioned several times, we can also capture the output in the same way we capture the input;  with time-dependent functions that are predefined for a particular obserbable at a time $t_{out}$.  This could be expected from the symmetry of input and output in this formalism; they are simply defined in terms of apply some operators at some particular time, it just so happens that input is applied at $t=0$.

The advantage is clear in more complex cases, when capturing the correlations for the output with exponential decomposition can be difficult.  In this case, with a SHO bath which naturally decomposes into just a few exponents, it is rather the oppposite, and adds some unneccessary computational complexity, but serves as a useful testbed.

<!-- #region -->
In this case, we need to slightly modify the definitions of the correlation functions we need for the output;  previously we used 


$$G_{1(out)}^L(t)= <b_{\omega_0}^{\dagger}(t) X^L(0)> = 0$$

$$G_{1(out)}^R(t)<b_{\omega_0}^{\dagger}(t)X^R(0)> = \lambda e^{i\omega_0 t - \Gamma |t|}$$

$$G_{2(out)}^L = <b_{\omega_0}(t) X^L(0)> = \lambda e^{-i\omega_0 t- \Gamma |t|}$$

$$G_{2(out)}^R= <b_{\omega_0}(t) X^R(0)> = 0$$



<!-- #endregion -->

<!-- #region -->
Now we need to define


$$G_{1(out)}^L(t)= <b_{\omega_0}^{\dagger}(t_{out}) X^L(t)> = 0$$

$$G_{1(out)}^R(t)<b_{\omega_0}^{\dagger}(t_{out})X^R(t)> = \lambda e^{i\omega_0 (t_{out}-t) - \Gamma(t)(t_{out}-t)}$$

$$G_{2(out)}^L = <b_{\omega_0}(t_{out}) X^L(t)> = \lambda e^{-i\omega_0 (t_{out}-t) - \Gamma(t)(t_{out}-t)}$$

$$G_{2(out)}^R= <b_{\omega_0}(t_{out}) X^R(t)> = 0$$

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
for tout in tlist:

    def g1R(t):
        return lam * np.exp(1.0j * Om * (tout - t) - Gamma * (tout - t))

    def g2L(t):
        return lam * np.exp(-1.0j * Om * (tout - t) - Gamma * (tout - t))

    bath = BosonicBath(Q, ckAR, vkAR, ckAI, vkAI)

    ck_in_1 = [input1]
    ck_in_2 = [input2]

    bath_input_1 = InputOutputBath(Q, ck_input=ck_in_1, tag="input1")
    bath_input_2 = InputOutputBath(Q, ck_input=ck_in_2, tag="input2")

    bath_output_1R = InputOutputBath(Q, ck_output_fn_R=[g1R], tag="output1")
    bath_output_2L = InputOutputBath(Q, ck_output_fn_L=[g2L], tag="output2")

    SHO_model = HEOMSolver(
        Hsys,
        [bath, bath_output_1R, bath_output_2L, bath_input_1, bath_input_2],
        NC,
        options=options,
    )

    t_list = np.linspace(0, tout, 100)
    resultSHO = SHO_model.run(rho0, t_list)

    labels = resultSHO.ado_states[-1].filter(level=2, tags=["output2",
                                                            "input1"])

    for label in labels:
        s0110 = (
            np.exp(1.0j * Om * tout - Gamma * tout)
            * resultSHO.ado_states[-1].extract(label).tr()
        )

    labels = resultSHO.ado_states[-1].filter(level=2, tags=["output1",
                                                            "input2"])

    for label in labels:
        s1001 = (
            np.exp(-1.0j * Om * tout - Gamma * tout)
            * resultSHO.ado_states[-1].extract(label).tr()
        )

    labels = resultSHO.ado_states[-1].filter(level=2, tags=["output1",
                                                            "output2"])
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
    tlist, np.real(result_output), "b--", alpha=1,
    label=r"heom1 static fields/TD func"
)
plt.xlabel("Time", fontsize=18)
plt.ylabel(r"$\langle a^{\dagger}a\rangle$", fontsize=18)
plt.legend()

plt.show()
```

```python
qutip.about()
```

```python
assert np.allclose(result_output,
                   expect(resultME1.states, a.dag() * a),
                   atol=1e-5)
```
