import numpy as np
import math

# function 1
def ackley(x):
    # Define the constants a, b, and c for the Ackley function
    a = 20
    b = 0.2
    c = 2 * math.pi
    # Get the number of dimensions (n) from the length of the input array x
    n = len(x)
    # Calculate the first part of the Ackley function
    sum1 = np.sum(x ** 2)
    # Calculate the second part of the Ackley function
    sum2 = np.sum(np.cos(c * x))
    # Calculate the individual terms for the Ackley function
    term1 = -a * math.exp(-b * math.sqrt(sum1 / n))
    term2 = -math.exp(sum2 / n)
    # Combine the terms and add constants to get the final result
    result = term1 + term2 + a + math.exp(1)
    
    return result

# function 2
def brown(x):
    # Extract the elements of the input vector x except the last one (xi)
    xi = x[:-1]
    # Extract the elements of the input vector x starting from the second element (xi+1)
    xi1 = x[1:]
    # Calculate the terms for each pair of xi and xi+1
    with np.errstate(over='ignore'):
        terms = ((xi ** 2) ** ((xi ** 2 + xi1 + 1) / (xi ** 2 + xi1)))
    # Sum up the terms and add the first element of the input vector (x0)
    result = terms.sum() + x[0]
    # Return the final result
    return result

# function 3
def dixon_price(x):
    n = len(x) 
    # Check the dimension of the input vector
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    result = (x[0] - 1) ** 2
    for i in range(1, n):
        result += (i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2
    
    return result

# function 4
def exponential(x):
    return -np.exp(np.sum(np.square(x)) * (-0.5))

# function 5
# The Griewank function has many widespread local minima, which are regularly distributed.
def griewank(x):
    n = len(x)
    sum_sq = np.sum(np.square(x))
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    result = 1 + (sum_sq / 4000) - prod_cos
    return result

# function 6
def powell_singular(x):
    n = len(x)
    if n < 4:
        raise ValueError("Dimension of the input vector must be at least 4.")
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10 * x[i + 1]) ** 2
        result += 5 * (x[i + 2] - x[i + 3]) ** 2
        result += (x[i + 1] - 2 * x[i + 2]) ** 4
        result += 10 * (x[i] - x[i + 3]) ** 4
    return result

# function 7
def rana(x):
    x1 = x[0]
    xi = x[1:]

    sqrt_abs_diff = np.sqrt(np.abs(x1 - xi + 1))
    sqrt_abs_sum = np.sqrt(np.abs(x1 + xi + 1))

    term1 = xi * np.sin(sqrt_abs_diff) * np.cos(sqrt_abs_sum)
    term2 = (x1 + 1) * np.sin(sqrt_abs_sum) * np.cos(sqrt_abs_diff)

    return np.sum(term1 + term2)

# function 8
def rastrigin(x):
    A = 10
    n = len(x)
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    sum_term = np.sum(x**2 - A * np.cos(2 * np.pi * x))
    result = A * n + sum_term
    
    return result

# function 9
def rosenbrock(x):
    n = len(x)
    
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    
    result = 0
    
    for i in range(n - 1):
        result += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    
    return result

# function 10 
def schwefel(x):
    n = len(x)
    
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    
    result = 418.9829 * n
    
    for i in range(n):
        result -= x[i] * np.sin(np.sqrt(abs(x[i])))
    
    return result

# function 11
def sphere(x):
    n = len(x)
    
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    
    result = np.sum(x**2)
    
    return result



