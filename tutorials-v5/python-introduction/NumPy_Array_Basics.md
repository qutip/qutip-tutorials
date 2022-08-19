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

```python
import numpy as np
```

# Introduction to NumPy Arrays


J.R. Johansson and P.D. Nation

For more information about QuTiP see [http://qutip.org](http://qutip.org)


## Introduction


Until now we have been using lists as a way of storing multiple elements together.  However, when doing numerical computations, lists are not very good.  For example, what if I wanted to add one to a list of numbers?  For a list `a = [1, 2, 3]` we can not write `a + 1`.


Instead we would have to do

```python
a = [1, 2, 3]
for k in range(3):
    a[k] = a[k] + 1
```

Working with lists would quickly become very complicated if we wanted to do numerical operations on many elements at the same time, or if, for example, we want to be able to construct vectors and matrices in our programs.  All of these features, and more, come with using NumPy **arrays** as our preferred data structure.


## NumPy Arrays


When dealing with numerical data in Python, nearly 100% of the time one uses arrays from the NumPy module to store and manipulate data.  NumPy arrays are very similar to Python lists, but are actually arrays in c-code that allow for very fast multi-dimensional numerical, vector, matrix, and linear algebra operations.  Using arrays with slicing, and **vectorization** leads to very fast Python code, and can replace many of the for-loops that you would have use if you coded a problem using lists. As a general rule, **minimizing the number of for-loops maximizes the performance of your code**.  To start using arrays, we can start with a simple list and use it as an argument to the array function

```python
a = np.array([1, 2, 3, 4, 5, 6])
print(a)
```

We have now created our first array of integers.  Notice how, when using print, the array looks the same as a list, however it is very much a different data structure.  We can also create an array of floats, complex numbers or even strings

```python
a = np.array([2.0, 4.0, 8.0, 16.0])
b = np.array([0, 1 + 0j, 1 + 1j, 2 - 2j])
c = np.array(["a", "b", "c", "d"])
print(a)
print(b)
print(c)
```

In general there are three different ways of creating arrays in Python:

- First create a list and then call the array function using the list as an input argument.

- Use NumPy functions that are designed to create arrays: **zeros, ones, arange, linspace**.

- Import data into Python from file.


### Arrays from Lists


We have already seen how to create arrays with simple lists, but now lets look at how to create more complicated lists that we can turn into arrays.  A short way of creating a list, say from 0 to 9 is as follows:

```python
output = [n for n in range(10)]
print(output)
```

This code is doing the exact same thing as the longer expression

```python
output = []
for n in range(10):
    output.append(n)
print(output)
```

We could turn this into an array quite easy

```python
np.array(output)
```

Or, we can save even more space and create the list inside of the array function:

```python
np.array([n for n in range(10)])
```

This can also be used to create more complicated arrays

```python
np.array([2.0 * k**0.563 for k in range(0, 10, 2)])
```

### Array Creation in NumPy (see [NumPy Documentation](http://docs.scipy.org/doc/numpy/reference/routines.array-creation.html) for more info.)


NumPy has several extremely important array creation functions that will make you life much easier. For example, creating arrays of all zeros or ones is trivial. 

```python
np.zeros(5)
```

```python
np.ones(10)
```

However, the most useful functions are [**```arange```**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html#numpy.arange) which generates evenly spaced values within a given interval in a similar way that the ```range``` function did, and [**```linspace```**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html) that makes a linear array of points from a starting to an ending value.

```python
np.arange(5)
```

```python
np.arange(0, 10, 2)
```

```python
# make an array of 20 points linearly spaced from 0 to 10
np.linspace(0, 10, 20)
```

```python
np.linspace(-5, 5, 15)  # 15 points in range from -5 to 5
```

## Differences Between Arrays and Lists


Having played with arrays a bit, it is now time to explain the main differences between NumPy arrays and Python lists.


Python lists are very general and can hold any combination of data types.  However, NumPy **arrays can only hold one type of data** (integers, floats, strings, complex).  If we try to combine different types of data, then the array function will **upcast** the data in the array such that it all has the same type

```python
np.array([1, 2, 3.14])  # [int,int,float] -> [float,float,float]
```

Upcasting between integers and floats does not cause too much trouble, but mixing strings and numbers in an array can create problems

```python
np.array([1.0, 1 + 1j, "hello"])  # array data is upcast to strings
```

If we want, we can manually change the type of the data inside the array using the ```dtype``` ("data type") keyword argument.  Frequently used dtypes are: ```int, float, complex, bool, str, object```, etc.  For example, to convert a list of integers to floats we can write

```python
np.array([1, 2, 3, 4, 5], dtype=float)
```

```python
np.arange(2, 10, 2, dtype=complex)
```

```python
np.array([k for k in range(10)], dtype=str)
```

Unlike Python lists, **we can not remove or add elements to an array once it has been created**.  Therefore, we must know the size of the array before creating it.


Because arrays hold only one type of data, mathematical functions such as multiplication and addition of arrays can be implemented in at the c-code level.  This means that these kinds of operations are very fast and memory efficient.  The mathematical operations on arrays are performed **elementwise**, which means that each element gets acted on in the same way.  This is an example of **vectorization**.  For example:

```python
a = np.array([1, 2, 3, 4])
5.0 * a  # This gets upcasted because 5.0 is a float
```

```python
5 * a**2 - 4
```

Recall that none of these operations worked on Python lists.


## Using NumPy Functions on Arrays


Remember that NumPy has a large builtin [collection of mathematical functions](http://docs.scipy.org/doc/numpy/reference/routines.math.html).  When using NumPy arrays as our data structure, these functions become even more powerful as we can apply the same function elementwise over the entire array very quickly.  Again, this is called vectorization and can speed up your code by many times.

```python
x = np.linspace(-np.pi, np.pi, 10)
np.sin(x)
```

```python
x = np.array([x**2 for x in range(4)])
np.sqrt(x)
```

```python
x = np.array([2 * n + 1 for n in range(10)])
sum(x)  # sums up all elements in the array
```

## Boolean Operations on Arrays


Like other mathematical functions, we can also use conditional statements on arrays to check whether each individual element satisfies a given expression.  For example, to find the location of array elements that are less than zero we could do

```python
a = np.array([0, -1, 2, -3, 4])
print(a < 0)
```

The result in another array of boolean (```True/False```) values indicating whether a given element is less than zero.  Or, for example, we can find all of the odd numbers in an array.

```python
a = np.arange(10)
print((np.mod(a, 2) != 0))
```

## Slicing NumPy Arrays


Just like lists, arrays can be sliced to get certain elements of the array, or to modify certain elements of the array.  For example, lets try to get every third element from a given array

```python
a = np.arange(20)
a[3::3]
```

Now lets set each of these elements equal to -1.

```python
a[3::3] = -1
print(a)
```

We can also slice the array so that it returns the original array in reverse

```python
a = np.arange(10)
a[::-1]
```

Finally, what if we want to get only those elements in the array that satisfy a certain conditional statement?  Recall that conditional statements on an array return another array of boolean values.  We can use this boolean array as an index to pick out only those elements where the boolean value is ```True```.

```python
a = np.linspace(-10, 10, 20)
print(a[a <= -5])
```

We must be careful though. Checking for multiple conditionals is not allowed `print(a[-8 < a <= -5])`.

The reason for this is the computer does not know how to take an array of many ```True/False``` values and return just a single value.


## Example: Rewriting Sieve of Eratosthenes


Here we will replace most of the for-loops used when writing the Sieve of Eratosthenes using lists will arrays.  This will make the code much easier to read and actually much faster for computing large prime numbers.  The main part of the original code is:

```python
N = 20
# generate a list from 2->N
numbers = []
for i in range(2, N + 1):  # This can be replaced by array
    numbers.append(i)
# Run Seive of Eratosthenes algorithm marking nodes with -1
for j in range(N - 1):
    if numbers[j] != -1:
        p = numbers[j]
        for k in range(j + p, N - 1, p):  # This can be replaced by array
            numbers[k] = -1
# Collect all elements not -1 (these are the primes)
primes = []
for i in range(N - 1):  # This can be replaced by array
    if numbers[i] != -1:
        primes.append(numbers[i])
print(primes)
```

Using arrays instead of lists simplifies the code:

```python
N = 20
# generate a list from 2->N
numbers = np.arange(2, N + 1)  # replaced for-loop with call to arange
# Run Seive of Eratosthenes algorithm
# by marking nodes with -1
for j in range(N - 1):
    if numbers[j] != -1:
        p = numbers[j]
        numbers[j + p: N - 1: p] = -1  # replaced for-loop by slicing array
# Collect all elements not -1 (these are the primes)
# Use conditional statement to get elements !=-1
primes = numbers[numbers != -1]
print(primes)
```

<h1 align="center">End of Tutorial</h1> 
<h3 align="center"><a href="http://qutip.org/tutorials.html">Return to QuTiP tutorials page</a></h3> 
