---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# QuTiP example: Bloch sphere animation


J.R. Johansson and P.D. Nation

For more information about QuTiP see [http://qutip.org](http://qutip.org)


Animation with qutip and matplotlib: decaying qubit visualized in a Bloch sphere.
(Animation with matplotlib does not work yet in python3)

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from qutip import Bloch, about, basis, mesolve, sigmam, sigmax, sigmay, sigmaz
from qutip.ipynbtools import plot_animation

%matplotlib inline
```

```python
def qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist):
    # operators and the hamiltonian
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    H = w * (np.cos(theta) * sz + np.sin(theta) * sx)
    # collapse operators
    c_op_list = []
    n_th = 0.5  # temperature
    rate = gamma1 * (n_th + 1)
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm)
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sm.dag())
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(np.sqrt(rate) * sz)

    # evolve and calculate expectation values
    output = mesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])
    return output
```

```python
w = 1.0 * 2 * np.pi  # qubit angular frequency
theta = 0.2 * np.pi  # qubit angle from sigma_z axis (toward sigma_x axis)
gamma1 = 0.5  # qubit relaxation rate
gamma2 = 0.2  # qubit dephasing rate
# initial state
a = 1.0
psi0 = (a * basis(2, 0) + (1 - a) * basis(2, 1)) / \
        (np.sqrt(a**2 + (1 - a) ** 2))
tlist = np.linspace(0, 4, 150)
```

```python
result = qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist)
```

```python
def plot_setup(result):

    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection="3d", elev=30, azim=-40)

    return fig, axes
```

```python
sphere = None


def plot_result(result, n, fig=None, axes=None):

    global sphere

    if fig is None or axes is None:
        fig, axes = plot_setup(result)

    if not sphere:
        sphere = Bloch(axes=axes)
        sphere.vector_color = ["r"]

    sphere.clear()
    sphere.add_vectors([result.expect[0][n],
                        result.expect[1][n],
                        result.expect[2][n]])
    sphere.add_points(
        [
            result.expect[0][: n + 1],
            result.expect[1][: n + 1],
            result.expect[2][: n + 1],
        ],
        meth="l",
    )
    sphere.make_sphere()

    return axes.artists
```

```python
# You can choose your own writer and codec here.
# Setting codec=None sets the codec to the standard
# defined in matplotlib.rcParams['animation.codec']
plot_animation(plot_setup, plot_result, result, writer="ffmpeg", codec=None)
```

## Versions

```python
about()
```
