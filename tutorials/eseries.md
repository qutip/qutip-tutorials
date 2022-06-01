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

# QuTiP example: eseries


J.R. Johansson and P.D. Nation

For more information about QuTiP see [http://qutip.org](http://qutip.org)

```python
from numpy import pi
```

```python
from qutip import *
```

## Example eseries object: $\sigma_x  \exp(i\omega t)$

```python
omega = 1.0
es1 = eseries(sigmax(), 1j * omega)
```

```python
es1
```

## Example eseries object: $\sigma_x \cos(\omega t)$

```python
omega = 1.0
es2 = eseries(0.5 * sigmax(), 1j * omega) + eseries(0.5 * sigmax(), -1j * omega)
```

```python
es2
```

## Evaluate eseries object at time $t = 0$

```python
esval(es2, 0.0)
```

## Evaluate eseries object at array of times $t = [0, \pi, 2\pi]$

```python
tlist = [0.0, 1.0 * pi, 2.0 * pi]
esval(es2, tlist)
```

## Expectation values of eseries

```python
es2
```

```python
expect(sigmax(), es2)
```

## Arithmetics with eseries

```python
es1 = eseries(sigmax(), 1j * omega)
es1
```

```python
es2 = eseries(sigmax(), -1j * omega)
es2
```

```python
es1 + es2
```

```python
es1 - es2
```

```python
es1 * es2
```

```python
(es1 + es2) * (es1 - es2)
```

## Expectation values of eseries

```python
es3 = eseries([0.5*sigmaz(), 0.5*sigmaz()], [1j, -1j]) + eseries([-0.5j*sigmax(), 
                                                                  0.5j*sigmax()], [1j, -1j])
es3
```

```python
es3.value(0.0)
```

```python
es3.value(pi/2)
```

```python
rho = fock_dm(2, 1)
es3_expect = expect(rho, es3)

es3_expect
```

```python
es3_expect.value([0.0, pi/2])
```

## Versions

```python
from qutip.ipynbtools import version_table

version_table()
```
