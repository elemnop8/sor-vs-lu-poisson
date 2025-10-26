""" This module implements numerical methods to solve the Poisson problem using 
    the finite difference method. It includes functionalities to set up the 
    right-hand side vector, map between equation numbers and grid points, compute 
    numerical errors, and visualize the error behavior.
    Author: M. Nguyen, E. Tarielashvili.
    pylint Version 3.1.0
    pylint score: 10/10
"""
import numpy as np
import matplotlib.pyplot as plt

def rhs(n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If n < 2.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    #stepsize
    h = 1 / n
    h2 = h**2 #square of the step size for scaling

    # Number of unknowns
    capn = (n-1)**2

    # Create the right-hand side vector b
    b = np.zeros(capn)

    # iterate over all interior grid points
    index = 0
    for k in range(1,n):
        for j in range(1,n):
            x1 = j/n
            x2 = k/n
            b[index] = f(x1, x2) * h2
            index += 1

    return b

def idx(nx, n):
    """ Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    j, k = nx
    return (j - 1) * (n - 1) + k

def inv_idx(m, n):
    """ Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.
    
    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    j = (m - 1) // (n - 1) + 1
    k = (m - 1) % (n - 1) + 1

    return [j, k]

def compute_error(n, hat_u, u):
    """Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.

    Parameters
    ----------
    n : int
        Number of intersections in each dimension
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.

    Returns
    -------
    float
        Maximal absolute error at the discretization points
    """
    # Step size h
    h = 1 / n

    # Maximum error
    max_error = 0.0

    # Index for hat_u
    index = 0

    # Loop over all inner points
    for k in range(1, n):  # Loop over x2 direction
        for j in range(1, n):  # Loop over x1 direction
            # Coordinates of the current point
            x1 = j * h
            x2 = k * h

            # Exact solution u(x) at the current point
            exact_value = u(x1, x2)

            # Numerical solution at the current point
            numerical_value = hat_u[index]

            # Compute the absolute error
            abs_error = abs(numerical_value - exact_value)

            # Update the maximum error
            max_error = max(max_error, abs_error)

            # Increment the index
            index += 1

    return max_error

def compute_error_plot(max_n, hat_u, u):
    """ Plots the error of the numerical solution in dependence of N

    Parameters
    ----------
    max_n : int
        Maximum value of n to consider.
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.
    """
    ns = range(2, max_n + 1)
    capn_values = [(n - 1)**2 for n in ns]
    error = []

    for n in capn_values:
        error.append(compute_error(n, hat_u, u))

    plt.figure(figsize=(10, 6))
    plt.loglog(capn_values, error, label="Fehler", marker='o')
    plt.xlabel("$N = (n-1)^2$", fontsize=15)
    plt.ylabel("absolute Fehler", fontsize=15)
    plt.title("Fehler der Approximation", fontsize=16, pad=20)
    plt.legend(fontsize=13)
    plt.grid()
    plt.show()
