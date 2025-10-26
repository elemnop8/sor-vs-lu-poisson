""" Module for solving linear system Ax = b 
    Author: M. Nguyen, E. Tarielashvili.
    pylint Version 3.1.0
    pylint score: 9.68/10
"""
import numpy as np
from scipy.linalg import lu
from numpy.linalg import norm

def solve_lu(p, l, u, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u.

    Parameters
    ----------
    p : numpy.ndarray
        permutation matrix of LU-decomposition
    l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
        upper triangular matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """
    # Apply permutation matrix to b: b' = P @ b
    pb = p @ b

    # L * y = Pb via forward substitution
    n = l.shape[0]
    y = np.zeros_like(pb, dtype=float)
    for i in range(n):
        y[i] = pb[i] - np.dot(l[i, :i], y[:i])

    # U * x = y via backward substitution
    x = np.zeros_like(y, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(u[i, i + 1:], x[i + 1:])) / u[i, i]

    return x

def solve_sor(matrix, rhs_vector, x0, params=None, omega=1.5):
    """ Solves the linear system Ax = b via the successive over relaxation method.

    Parameters
    ----------
    A : numpy.ndarray
        System matrix of the linear system.
    b : numpy.ndarray (of shape (N,) )
        Right-hand-side of the linear system.
    x0 : numpy.ndarray (of shape (N,) )
        Initial guess of the solution.

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    omega : float, optional
        relaxation parameter

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinity norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """
    # Convert sparse matrices to dense format if necessary
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.toarray()
    if not isinstance(rhs_vector, np.ndarray):
        rhs_vector = rhs_vector.toarray().flatten()

    # Extract termination parameters from the params dictionary
    eps = params.get('eps', 1e-8)  # Tolerance for residual norm
    max_iter = params.get('max_iter', 1000)  # Maximum number of iterations
    var_x = params.get('var_x', 1e-4)  # Tolerance for change in solution

    if params is None:
        params = {'eps': 1e-8, 'max_iter': 1000, 'var_x': 1e-4}


    # Ensure that at least one termination condition is active
    if eps <= 0 and max_iter <= 0 and var_x <= 0:
        raise ValueError("At least one termination condition must be active.")

    # Initialization
    n = len(rhs_vector)  # Number of variables in the system
    x = x0.copy()  # Current guess for the solution
    residual_norms = []  # List to store residual norms
    iterates = [x.copy()]  # List to store solution iterates

    for _ in range(max_iter):
        x_old = x.copy()

        # Perform SOR iteration
        for i in range(n):
            sigma = 0
            if i > 0:
                sigma += np.dot(matrix[i, :i], x[:i])
            if i < n - 1:
                sigma += np.dot(matrix[i, i+1:], x_old[i+1:])
            x[i] = (1 - omega) * x_old[i] + (omega / matrix[i, i]) * (rhs_vector[i] - sigma)

        # Compute residual
        r = rhs_vector - matrix @ x
        residual_norm = norm(r, ord=np.inf)
        residual_norms.append(residual_norm)
        iterates.append(x.copy())

        # Check termination conditions
        if residual_norm < eps:
            return "eps", iterates, residual_norms
        if norm(x - x_old, ord=np.inf) < var_x:
            return "var_x", iterates, residual_norms

    return "max_iter", iterates, residual_norms

def main():
    """
    Main function to demonstrate the use of LU decomposition and SOR methods
    for solving linear systems.
    """
    # Define a sample linear system Ax = b
    matrix = np.array([[4, 1, 0],
              [1, 4, 1],
              [0, 1, 4]], dtype=float)
    b = np.array([6, 10, 6], dtype=float)

    # Define initial guess for x0 for SOR
    x0 = np.zeros_like(b)

    # LU decomposition (example)
    p, l, u = lu(matrix)  # LU decomposition with permutation (use SciPy's LU decomposition)

    # Solve using LU decomposition
    x_lu = solve_lu(p, l, u, b)
    print("Solution from LU decomposition:")
    print(x_lu)

    # Solve using Successive Over-Relaxation (SOR)
    params = {'eps': 1e-8, 'max_iter': 1000, 'var_x': 1e-4}
    omega = 1.5
    reason, iterates, residual_norms = solve_sor(matrix, b, x0, params=params, omega=omega)

    print(f"\nSolution from SOR (Terminated with reason: {reason}):")
    print(iterates[-1])  # The last iterate should be the final solution
    print("\nResidual norms during iterations:")
    print(residual_norms)

if __name__ == "__main__":
    main()
