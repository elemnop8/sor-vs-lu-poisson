""" This module contains functions for plotting the errors and runtime comparisons 
    for different numerical solvers, such as LU-decomposition and SOR with both fixed and 
    h-dependent epsilon values.
    Author: M. Nguyen, E. Tarielashvili.
    pylint Version 3.1.0
    pylint score: 10/10
"""

import numpy as np
import matplotlib.pyplot as plt
from hilfsmittel import save_plot

def plot_error_for_eps_variants(numbers, errors, epsilon_values):
    """
    Plots the error in the infinity norm for different ε(k) variants.

    Parameters
    ----------
    Ns : list of int
        Number of inner points for each grid size.
    errors : dict
        Dictionary where keys are ε(k) values and values are lists of errors
        corresponding to the grid sizes.
    epsilon_values : list of int
        List of k values for ε(k) = h^k.

    Returns
    -------
    None.
    """
    # Define 5 distinct linestyles and markers
    # Dashed-dot-dashed for the 5th
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
    # Circle, square, diamond, cross, triangle
    markers = ['s', 'o', 'D', 'x', '^']

    plt.figure(figsize=(10, 6))

    for i, k in enumerate(epsilon_values):
        linestyle = linestyles[i % len(linestyles)] # Cycle through linestyles
        marker = markers[i % len(markers)] # Cycle through markers
        plt.loglog(numbers, errors[k], linestyle=linestyle,
                   marker=marker, label=f"ε = $h^{{{k}}}$", alpha=0.8)

    # Add reference lines for theoretical convergence rates
    for p, k in enumerate(epsilon_values):
        if k in errors and errors[k]:
            # reference lines
            e0 = errors[k][0]  # Use the first error value as reference
            h_values = 1 / ((numbers ** 0.5) +1)
            label= f"$\\mathcal{{O}}\\left(\\frac{{1}}{{N^{{{epsilon_values[p]}}}}}\\right)$"
            if p in (1, 2):
                ref_line = e0 * h_values ** epsilon_values[p]
                plt.loglog(numbers, ref_line, linestyle='--', alpha=0.7,
                           label=label)

    plt.xlabel("Number of Inner Points N= $(n-1)^2$ ", fontsize=15)
    plt.ylabel("Maximal Error (Infinity Norm)", fontsize=15)
    plt.title("Error vs Number of Inner Points for Different ε(k)", fontsize=16, pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    save_plot("plot_error_for_eps_variants.png")
    plt.show()

def plot_fixed_and_h_dependent_errors(numbers, errors_fixed, errors_h, errors_lu):
    """
    Plots the error in the infinity norm for fixed epsilon and h-dependent epsilon.

    Parameters
    ----------
    Ns : list of int
        Number of inner points for each grid size.
    errors_fixed : list of float
        Errors for the fixed epsilon across grid sizes.
    errors_h : list of float
        Errors for the h-dependent epsilon across grid sizes.
    errors_lu : list of float
        Errors for lu-decomposition across grid sizes.

    Returns
    -------
    None.
    """
    compare_1_n=1/np.array(numbers)
    plt.figure(figsize=(10, 6))
    plt.loglog(numbers, compare_1_n, linestyle=":", color="gray", label="1/N")

    # Plot fixed epsilon errors
    plt.loglog(numbers, errors_fixed, color="pink",linewidth=3,label="Fixed ε = 1e-8")

    # Plot h-dependent epsilon errors
    plt.loglog(numbers, errors_h, linestyle=":",
               marker="o",color="blue", label=f"h-dependent ε")

    # Plot LU errors
    plt.loglog(numbers, errors_lu, linestyle="--",marker="x",color="purple", label="error LU")

    plt.xlabel("Number of Inner Points N= $(n-1)^2$ ", fontsize=15)
    plt.ylabel("Maximal Error", fontsize=15)
    plt.title("Error Comparison: Fixed ε vs h-dependent ε", fontsize=16, pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    save_plot("plot_fixed_and_h_dependent_errors.png")
    plt.show()

def plot_runtime_comparison(numbers, lu_times, sor_times_fixed, sor_times_h):
    """
    Plot comparing the measured times

    Parameters
    ----------
    Ns : list of int
        Number of inner points for each grid size.
    lu_times : list of float
        list of measured times solving lu-decomposition
    sor_times_fixed : list of float
        list of measured times solving via sor with fixed epsilon
    sor_times_h : list of float
        list of measured times solving via sor with h-dependend epsilon

    Returns
    -------
    None.
    """
    plt.figure(figsize=(10, 6))

    # Plot the runtime comparison for LU and SOR solvers
    plt.loglog(numbers, lu_times, label='LU Solver',
               linestyle='--', marker='o', color='blue')
    plt.loglog(numbers, sor_times_fixed, label='SOR Solver (fixed ε)',
               linestyle='-', marker='s', color='green')
    plt.loglog(numbers, sor_times_h, label='SOR Solver (h-dependent ε)',
               linestyle=':', marker='^', color='red')

    # Labeling the axes and setting the title
    plt.xlabel("Number of Inner Points N= $(n-1)^2$", fontsize=15)
    plt.ylabel("Measured time in milliseconds ms", fontsize=15)
    plt.title("Comparing the runtime between SOR and LU-decomposition", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    save_plot("runtime_comparison.png")
    plt.show()

def main():
    """
    Main function that demonstrates the usage of plotting
    functions for error analysis and runtime comparisons.
    """
    # example for N
    numbers = np.array([10, 20, 40, 80, 160, 320])

    # example for errors
    errors = {
        1: [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125],
        2: [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625],
        3: [0.3, 0.15, 0.075, 0.0375, 0.01875, 0.009375],
        4: [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    }

    epsilon_values = [1, 2, 3, 4]

    # example errors for fixed an different epsilons
    errors_fixed = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    errors_h = [0.15, 0.075, 0.0375, 0.01875, 0.009375, 0.0046875]

    # example measured times
    lu_times = [10, 9, 7, 5, 3, 2]
    sor_times_fixed = [12, 11, 9, 7, 5, 4]
    sor_times_h = [14, 13, 10, 8, 6, 5]

    # Plot for errors with differens epsilons
    plot_error_for_eps_variants(numbers, errors, epsilon_values)

    # Plot for errors
    plot_fixed_and_h_dependent_errors(numbers, errors_fixed, errors_h, errors_fixed)

    # Plot runtime
    plot_runtime_comparison(numbers, lu_times, sor_times_fixed, sor_times_h)

if __name__ == "__main__":
    main()
