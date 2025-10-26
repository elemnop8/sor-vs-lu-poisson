# Iterative vs. Direct Solvers for Linear Systems ⚙️  

This project was created as part of *Project Practicum I* at the **Humboldt-Universität zu Berlin (2025)**.  
It investigates **iterative** and **direct** methods for solving large linear systems, focusing on the **Successive Over-Relaxation (SOR)** method and comparing it with **LU decomposition**.

---

## 📘 Contents
- **Report:** `Handout_SOR_vs_LU.pdf`  
- **Code:**
  - `poisson_problem_2d.py` – setup of the Poisson problem  
  - `block_matrix_2d.py` – construction of sparse block matrices  
  - `linear_solvers.py` – LU decomposition and iterative solver implementations  
  - `experiments_it.py` – numerical experiments with SOR and LU  
  - `hilfsfunktionen_plot.py` – plotting utilities  

---

## ⚙️ Methods
- Finite Difference Discretization of the 2D Poisson Equation  
- Construction of Sparse Matrices and Linear Systems  
- **LU Decomposition** (direct method)  
- **Successive Over-Relaxation (SOR)** method  
- Convergence and Error Analysis  
- Spectral Radius Computation  

---

## 📊 Results
The experiments show:
- The **SOR method** converges faster for optimal relaxation parameter \( \omega_{\text{opt}} \approx 1.8{-}1.9 \).  
- The **optimal ω increases** with finer grid sizes \( n \) and approaches 2.  
- For small systems (\( N \leq 10^4 \)), **LU decomposition** remains faster.  
- An **\( h^2 \)** convergence rate was verified for the finite difference discretization.  
- Using an **\( h^4 \)**-dependent stopping criterion \( \varepsilon = h^4 \) yields high accuracy and stable results.  

---

## 🧠 Requirements
Python ≥ 3.9  
Packages: `numpy`, `scipy`, `matplotlib`
