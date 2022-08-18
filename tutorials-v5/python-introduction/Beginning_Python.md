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

# Brief Python Language Tutorial


P.D. Nation and J.R. Johansson

For more information about QuTiP see [http://qutip.org](http://qutip.org)


## Imports
Here we import the required functions for later usage.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
```

## Using Python For Simple Calculations


As a first step, lets try to use the interactive Python command line tool **iPython** as a basic calculator.  Addition, subtraction, and multiplication, all work in the same way as you would write the equations down on paper

```python
10 + 5
```

```python
10 - 157
```

```python
4 / 3
```

```python
(50 - 4) * 10 / 5
```

However, things like raising a number to a power, $4^{4}$, are written differently.

```python
4**4
```

We could also express this in the mathematically equivalent way as $4^{4.0}$.  However, inside of the computer, the result is not treated the same as the above answer.

```python
4**4.0
```

## Integers vs. Floating Point Numbers


All information stored in a computer must be represented in a binary format consisting of zeros and ones (e.g. $461\rightarrow 111001101$).  Each zero or one is called a **bit**, and given $N$ bits, one can store all of the integers in the range $[0,2^{N-1}]$, where the $-1$ is due to the fact that the first bit is reserved for defining if a number is positive or negative   

However, given a fixed number of bits, it is impossible to store an arbitrary number exactly.  Therefore, if one is given a random number, unless the number is exactly divisible by a factor of two, the conversion between the random number and the binary bit representation ultimately leads to a loss of precision, and what is known as **roundoff error**.



When dealing with numbers inside a computer there are two distinct types of numbers to consider:


- **Integers** - (1,2,4,-586,..) Are what are called **fixed-point numbers**, where the term fixed-point means that there is a fixed number of decimal places in the number (zero for integers).  These numbers can be stored exactly in a computer.


- **Doubles/Floats** - (3.141,0.21,-0.1,..) These are **floating-point numbers** that are the binary equivalent to scientific notation $c=2.99792458\times 10^{8}$.  Doubles (also called double-precision numbers) are floating point numbers that are written using 64-bits and, in general, are only accurate to the 15th or 16th decimal place.  Floats (or single-precision numbers) use 32-bits, and are good to 6-7 decimal places.  **Serious scientific calculations always require a combination of integers and double (64-bit) numbers**.

```python
7 + 0.000000000000001
```

```python
7 + 0.0000000000000001
```

```python
0.1 + 0.2
```

This last example clearly highlights the fact that the computer does not store decimal (floating-point) numbers exactly.  The loss of precision in floating-point numbers can be characterized by the **machine precision**, $\epsilon_{\rm m}$, that is defined to be the smallest positive number such that 

$$1_{\rm c}+\epsilon_{\rm m}\neq 1_{\rm c}$$

where the subscript on $1_{\rm c}$ is meant to remind you that this is a computer number.  Therefore, for any arbitrary number $N$ is related to its floating-point equivalent $N_{\rm c}$ by

$$N_{\rm c}=N\pm \epsilon, \ \ \forall~|\epsilon|< \epsilon_{\rm m}.$$

**Take Home Message** - All double-precision decimal numbers that are not factors of two will have error in the 15th decimal place.  This can lead to errors in your numerical solutions if you are not careful.


## Making Python Smarter Using NumPy


Python itself has limited support for mathematics outside of simple arithmetic.  Therefore, we will use the functions in the NumPy module to do more impressive, and faster, calculations.  We have imported NumPy already at the top of this notebook and can use it now by referring to `np`.

We can now do more impressive calculations:

```python
np.exp(2.34)
```

```python
np.sqrt(5)
```

```python
np.sinc(0.5)
```

## Variables


If we want to be able to store the numbers and results from our calculations then we must define variables using the "=" sign:

```python
radius = 5
area = np.pi * radius**2
area
```

We see that our variables name is defined on the left of the ```=``` sign and the value its given is defined on the right.  Here we have also used the ```pi``` variable that has been predefined by NumPy.  Variables can then be used in other expressions.  

If a predefined variable is again used on the left side of ```=``` then its original value is replaced.

```python
x = 10
x = (x**2 + 25) / 10
x
```

This is different than the mathematical equation $10x=x^{2}+25$  which has the solution $x=5$.  Therefore, it is important to remember that the ```=``` sign in a computer program is **not** equivalent to the mathematical equality. 


What happens if you try to use a variable without first defining it? 

Python would give us an error that the variable is not defined.  In addition, there are several words that are reserved by the Python language and cannot be used as variables:

    and, as, assert, break, class, continue, def, del, elif, else, except, 
    exec, finally, for, from, global, if, import, in, is, lambda, not, or,
    pass, print, raise, return, try, while, with, yield
    
Other than the above reserved words, your variables can be anything that starts with a letter or the underscore character "$\_$" followed by any combination of alphanumeric characters and "$\_$".  Note that using upper or lower case letters will give you two different variables.

```python
_freq = 8
Oscillator_Energy = 10
_freq * Oscillator_Energy
```

### Some Rules About Variables


Although there are many ways to define variables in Python, it is best to try to define your variables in all the same way.  In this class, all of our variables will use only lower case characters. 


```python
speed_of_light = 2.9979 * 10**8
spring_constant = np.sqrt(2 / 5)
```

It is also good practice to use variable names that correspond to the physical quantity that the variable represents.


## Strings


Often we want to print some text along with our variables, ask the user for input, or actually use the words and letters themselves as variables (e.g. in DNA analysis).  All of these can be accomplished using **strings**.  We have already seen one string already in this class: 

```python
"Hello Class"
```

We can also use single quotes, e.g. `'Hello Class'`.

If we want to use the quote symbol in the string itself then we need to mix the two types 

```python
"How was Hwajung's birthday party?"
```

Just like we did with integers and doubles, we can assign a string to a variable, and we can even add two strings together.

```python
a = "I like "  # There is a blank space at the end of this string.
b = "chicken and HOF"
a + b
```

Notice the blank space at the end of the string in variable "a" provides spacing between "like" and "chicken".


If we want to print out stuff, including strings and integers or doubles together, then we can use the builtin ```print``` function to accomplish this

```python
temp = 23
text = "The temperature right now is"
print(text, temp)
```

Notice how the print function automatically puts a space between the the two input arguments.  The ```print``` function automatically takes any number of string, integer, double, or other variables, converts them into strings, and then prints them for the user.


## Lists


Often times we will want to group many variables together into one object.  In Python this is accomplished by using a **```list```** datatype variable.

```python
shopping_list = ["eggs", "bread", "milk", "bananas"]
```

If we want to access a single variable inside of the list, then we need to use the **index** that corresponds to the variable inside of square brackets.

```python
shopping_list[2]
```

We see that the "milk" string can be accessed using the index number $2$.  However, we can see that this variable is actually the third string in the list.  This discrepancy is due to the fact that Python (like C-code) considers the first element in a list, or other multivariable data structures, to be at index $0$.

```python
shopping_list[0]
```

This is important to remember, and will take some getting used to before it becomes natural.  If we want to access the elements of the list from back to front, we can use negative indices

```python
shopping_list[-1]
```

```python
shopping_list[-2]
```

If we are given a list variable and we want to known how many elements are inside of the list, then we can use the ```len``` function that returns an integer giving the length of the list.

```python
len(shopping_list)
```

If we want to change the length of the list by adding or removing elements, then we can use ```append``` and ```remove```, respectively.

```python
shopping_list.append("apples")
shopping_list
```

```python
shopping_list.remove("bread")
shopping_list
```

Note that lists to not have to have the same type of data in each element!  You can mix any data types you want.

```python
various_things = [1, "hello", -1.234, [-1, -2, -3]]
various_things
```

All of these elements can be accessed in the usual way

```python
various_things[0]
```

```python
various_things[-1]
```

```python
various_things[3][1]
```

## Iterating Through Lists and Python Indention Rules


One of the most important reasons for using lists is because one often wants to do the same type of manipulation on each of the elements one at a time.  Going through a list in this fashion is called **iteration** and is accomplished in Python using the ```for``` command: 

```python
items = [
    "four calling birds",
    "three french hens",
    "two turtle doves",
    "a partridge in a pear tree",
]
for thing in items:
    print(thing)
```

Here, "thing" is a variable that takes the value of each item in the list "items" and then gets sent to the ```print``` function.  We are free to call this variable anything we want.  

```python
for variable in items:
    print(variable)
```

The next important thing to notice is that after the colon ":" the print statement is indented.  This indention after a colon is required in the Python programming langage and represents a section of the code called a **block**.  If we did not intent the print function then Python would yell at us. 


Blocks are a standard part of any programming language and are used for organization and flow-control in computer code.  Anything that is indented in the above example will be run for each item in the list

```python
for variable in items:
    print("My true love gave to me", variable)
```

## Slicing Lists


If we want to grab certain elements from a list we can make use of **slicing** to conveniently access the elements.  Slicing can be used on any **sequence** such as lists, strings, and as we will see shortly, arrays. Consider our ```shopping_list``` list:

```python
shopping_list = ["eggs", "bread", "milk", "bananas", "apples"]
```

To get the first element we used a single index

```python
shopping_list[0]
```

But if we want to get the first three elements in the list we can use: 

```python
shopping_list[0:3]
```

We could also grab the last two elements using:

```python
shopping_list[-2:]
```

Or, we can get even more complex and grab all of the even number elements by using a third argument in the brackets that tells use the step size:

```python
shopping_list[0::2]
```

## Conditional Statements


We have now seen a collection of data types (integers, doubles/floats, lists, strings) but we have yet to discuss how to compare two different variables.  For example, how do we check if two different integers $a$ and $b$ are equal?  Or how do we know if $a\ge b$?  This is accomplished using **conditional statements**.  The basic operations in boolean logic are "equal" (```==```), "not equal" (```!=```), "greater than" (```>```), "greater than or equal" (```>=```), "less than" (```<```), and "less than or equal" (```<=```).  All of these conditionals operate on two variables and return a simple boolean ```True``` or ```False``` answer.  For example

```python
a = 5
b = 8
a > b
```

```python
c = 0
c <= 0, c >= 0
```

```python
a = 5
b = 6
a == b, a != b
```

It is important to point out that in Python ```1``` and ```0``` are the same as ```True``` and ```False```, respectively.

```python
t = True
f = False
t == 1, f == 0
```

We can also combine multiple conditional statements

```python
a = -1
b = 4
c = 10
d = 11
a < b < c != d
```

These operations can also be used on lists and strings:

```python
[4, 5, 6] >= [4, 5, 7]
```

```python
[4, 5, 6] <= [4, 5, 7]
```

```python
"today" == "Today"
```

### Conditional Statements and Flow Control


The main purpose of these conditional statements is to control the flow of a Python program.  The result of a conditional statement can be used to control a program using ```if/else``` and ```while``` statements.

```python
today = "friday"
if today == "friday":
    print("We have class today :(")  # this is a code block
else:
    print("No class today :)")  # this is also a code block
```

The code block below the ```if``` statement is run only if the conditional ```today=='friday'``` returns ```True```.  If the conditional is ```False``` then the code block inside the ```else``` statement is run.  We can also check multiple conditions by using the ``elif`` statement after ```if```:

```python
today = "thursday"
if today == "friday":
    print("We have class today :(")
elif today == "thursday":
    print("Our assignment is due today :(")
else:
    print("No class today :)")
```

The other important flow control expression is the **```while``` loop** that executes a block of code repeatedly until the conditional statement at the start of the loop is ```False```.

```python
n = 0
while n <= 10:  # evaluate code block until n>10
    print("The current value of n is:", n)
    n = n + 1  # increase the value of n by 1
```

When using a ```while``` loop you must make sure the conditional is not ```True``` forever.  Otherwise your program will be in an **infinite loop** that never ends.


### Example: Even and Odd Numbers


Let us determine whether a given number between [1,10] is an even or odd number.

```python
for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    if np.remainder(n, 2) == 0:
        print(n, "is even")
    else:
        print(n, "is odd")
```

Typing lists with a long sequence of integers is quite annoying.  Fortunately, Python has a builtin function called ```range``` that makes creating sequences of integers very easy.  For instance, the above example becomes

```python
for n in range(1, 11):
    if np.remainder(n, 2) == 0:
        print(n, "is even")
    else:
        print(n, "is odd")
```

Notice how the range function only counts to $10$ even though the range goes to $11$.  The endpoint is *never* part of the generated sequence when using ```range```.  If we wanted the ```range``` function to start at zero instead of one we could simply write ```range(11)```.  We can also make sequences that go in arbitrary steps:

```python
for n in range(0, 11, 2):
    print(n)
```

The ```range``` function does not return a list of integers but is something called a **generator**.  In general, the ```range``` function should only be used in combination with the ```for``` command.


### Example: Fibonacci Sequence


Let us follow the Python documentation and calculate the first ten numbers in the Fibonacci sequence:

```python
n = 10
fib = [0, 1]
for i in range(2, n):
    fib.append(fib[i - 1] + fib[i - 2])
print(fib)
```

We can also write this using a ```while``` loop if we wanted to.

```python
n = 2
fib = [0, 1]
while n < 10:
    fib.append(fib[n - 1] + fib[n - 2])
    n = n + 1
print(fib)
```

## Writing Scripts and Functions


Up until now we have been running little code snippets but have not really been doing any real programming.  Recall that Python is a scripting language.  Therefore, most of the time, we want to write **scripts** that contain a collection of constants, variables, data structures, functions, comments, etc., that perform various complicated tasks. 


###  Scripts


A Python script file is nothing but a text file containing Python code that ends with a **.py** extension.  Python scripts are also called Python **programs**.  If we open up any editor, then we are given a blank window that we can enter our Python commands in.


Before we begin to write our scripts, lets first discuss the best format for writing your scripts.

```python
# This is an example script for the P461 class
# Here we will calculate the series expansion
# for sin(x) up to an arbitrary order N.
#
# Paul Nation, 02/03/2014

N = 5  # The order of the series expansion
x = np.pi / 4.0  # The point at which we want to evaluate sine

ans = 0.0
for k in range(N + 1):
    ans = ans + (-1) ** k * x ** (1 + 2 * k) / factorial(1 + 2 * k)
print("Series approximation:", ans)
print("Error:", np.sin(x) - ans)
```

We can see that the script has four main parts: First, we have a section of **comments** that describe what the script does and when it was created.  In python all comments start with the **```#```** symbol.  Everything after this symbol is ignored by the computer.  Second, we have the section of the scripts that load the necessary functions that we need from other packages.  Third is a section where we define all of the constants that are going to be used in the script. You should also add comments here that tell us what the constants are.  Finally, your main body of code goes after these sections.


### Functions


We are finally in a position to look at one of the most important parts of any programming language **functions**.  Functions are blocks of code that accomplish a specific task. Functions usually take "input arguments", perform operations on these inputs, and then "return" one or more results. Functions can be used over and over again, and can also be "called" from the inside of other functions.  Let us rewrite our script for $sin(x)$ using a function and then describe each part.

```python
N = 5  # The order of the series expansion
x = np.pi / 4.0  # The point at which we want to evaluate sine


def sine_series(x, N):
    ans = 0.0
    for k in range(N + 1):
        ans = ans + (-1) ** k * x ** (1 + 2 * k) / factorial(1 + 2 * k)
    return ans


result = sine_series(x, N)
print("Series approximation:", result)
print("Error:", np.sin(x) - result)
```

We see see that a function is created using the keyword ```def``` which is short "define", then the name of the function followed by the input arguments in parentheses.  After the block of code called by the function, the ```return``` keyword specifies what variable(s) and/or data structure(s) are given as the output.  So a general functions call is

```python
def function_name(arg1, arg2):
    "Block of code to run"
    "..."
    return result
```

Again, everything after the colon (:) that is inside the function must be indented.  The beauty of using functions is that we can use the same code over and over, just by changing the constants near the top of our Python script.

Variables that are defined inside of a function are called **local variables** and only defined for the block of code inside of the function.  In our previous example, ```k``` was a local variable.  The input arguments and return arguments are *not* local variables.  Once a function is done running, the local variables are erased from memory.  Therefore, if you want get something out of a function, your must return the value when your done.


If we want to return more than one thing at the end of the function then we just need to separate the different items by a comma.

```python
N = 100  # Number of points to generate


def random_coordinates(N):
    x_coords = []
    y_coords = []
    for n in range(N):
        xnew, ynew = np.random.random(2)
        x_coords.append(xnew)
        y_coords.append(ynew)
    return x_coords, y_coords


xc, yc = random_coordinates(N)
plt.plot(xc, yc, "ro", markersize=8)
plt.show()
```

```python
N = 20  # Number of points to generate


def random_coordinates(N):
    x_coords = []
    y_coords = []
    for n in range(N):
        xnew, ynew = np.random.random(2)
        x_coords.append(xnew)
        y_coords.append(ynew)
    return x_coords, y_coords


def dist2d(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def max_dist(xc, yc):
    max_dist = 0.0
    num_points = len(xc)
    for ii in range(num_points):
        for jj in range(num_points):
            dist = dist2d(xc[ii], yc[ii], xc[jj], yc[jj])
            if dist > max_dist:
                max_dist = dist
                xvals = [xc[ii], xc[jj]]
                yvals = [yc[ii], yc[jj]]
    return max_dist, xvals, yvals


xc, yc = random_coordinates(N)
max_dist, pnt1, pnt2 = max_dist(xc, yc)
plt.plot(xc, yc, "ro", markersize=8)
plt.plot(pnt1, pnt2, "b-", lw=2)
plt.show()
```

Obvious this last example is more complex, and in particular, it is hard to understand what the functions.  Even for your own functions, you may often forget what your functions do unless you provide some documentation and comments in your scripts.  Here we will see how to properly document a function in Python by looking at the ```max_dist``` function:

```python
def max_dist(xc, yc):
    """
    Finds the maximum distance between any two points
    in a collection of 2D points.  The points corresponding
    to this distance are also returned.

    Parameters
    ----------
    xc : list
        List of x-coordinates
    yc : list
        List of y-coordinates

    Returns
    -------
    max_dist : float
        Maximum distance
    xvals : list
        x-coodinates of two points
    yvals : list
        y-coordinates of two points

    """
    max_dist = 0.0  # initialize max_dist
    num_points = len(xc)  # number of points in collection
    for ii in range(num_points):
        for jj in range(num_points):
            dist = dist2d(xc[ii], yc[ii], xc[jj], yc[jj])
            if dist > max_dist:
                max_dist = dist
                xvals = [xc[ii], xc[jj]]
                yvals = [yc[ii], yc[jj]]
    return max_dist, xvals, yvals
```

Everything inbetween the ```"""..."""``` is called a **docstring** and it gives a tells someone who is not familiar with a partiular functions a detailed explaination as to what the function does, what parameters it takes as inputs, and what values it returns.  It is also good practice to put some comments next to your local variables so the user knows what each of these is for. Although it seems like a lot of work at first, writing docstrings will make you a much better programmer in the future.


<h1 align="center">End of Tutorial</h1> 
<h3 align="center"><a href="http://qutip.org/tutorials.html">Return to QuTiP tutorials page</a></h3> 
