"""
Example 1: FEM assembly-based implementation (Algorithms IV, V, and VI)

This script demonstrates the general assembly framework for the 1D second-order
elliptic equation using:
- Algorithm IV: Stiffness matrix assembly
- Algorithm V: Load vector assembly  
- Algorithm VI: Dirichlet boundary conditions

Problem: -d/dx(e^x * du/dx) = -e^x[cos(x) - 2sin(x) - x cos(x) - x sin(x)]
Domain: 0 <= x <= 1
Boundary conditions: u(0) = 0, u(1) = cos(1)
Analytical solution: u(x) = x cos(x)
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.helper import construct_P_matrix_linear, construct_T_matrix_linear
from utils.stiff_matrix_generation import generate_stiff_matrix_by_assembly
from utils.load_vector_generation import generate_load_vector_by_assembly
from utils.boundary_condition import apply_dirichlet_boundary_conditions


def analytical_solution(x):
    """
    Compute analytical solution for Example 1
    Analytical solution: u(x) = x cos(x)
    """
    u = x * np.cos(x)
    return u


def compute_error(u_fem, u_analytical):
    """Compute L2 and L∞ errors"""
    l2_error = np.sqrt(np.mean((u_fem - u_analytical)**2))
    linf_error = np.max(np.abs(u_fem - u_analytical))
    return l2_error, linf_error


def solve_1d_elliptic_fem_assembly(a, b, N, c_func, f_func, ga, gb, verbose=True):
    """
    Solve the 1D second-order elliptic equation using Finite Element Method
    with Algorithm IV and V (local assembly approach)
    
    Problem: -d/dx(c(x) * du/dx) = f(x) for a < x < b
    Boundary conditions: u(a) = ga, u(b) = gb
    
    Parameters:
    a, b (float): Domain boundaries
    N (int): Number of elements
    c_func (callable): Coefficient function c(x)
    f_func (callable): Source function f(x)
    ga, gb (float): Boundary values
    verbose (bool): If True, print step-by-step information
    
    Returns:
    tuple: (x_nodes, u_fem, P, T, A, b_vector)
    """
    # Step 1: Construct P and T matrices for linear finite elements
    if verbose:
        print("Step 1: Constructing P and T matrices...")
    P = construct_P_matrix_linear(a, b, N)
    T = construct_T_matrix_linear(N)
    
    if verbose:
        print(f"  P matrix shape: {P.shape} (node coordinates)")
        print(f"  T matrix shape: {T.shape} (element connectivity)")
        print(f"  Number of nodes: {P.shape[1]}")
        print(f"  Number of elements: {T.shape[1]}")
    
    # Extract node coordinates
    x_nodes = P[0, :]  # All nodes including boundaries (N+1 total nodes)
    
    # Step 2: Generate stiffness matrix A using Algorithm IV
    if verbose:
        print("\nStep 2: Generating stiffness matrix using Algorithm IV...")
    A = generate_stiff_matrix_by_assembly(P, T, c_func)
    if verbose:
        print(f"  Stiffness matrix A shape: {A.shape}")
        print(f"  Condition number: {np.linalg.cond(A):.2e}")
    
    # Step 3: Generate load vector b using Algorithm V
    if verbose:
        print("\nStep 3: Generating load vector using Algorithm V...")
    b_vector = generate_load_vector_by_assembly(P, T, f_func)
    if verbose:
        print(f"  Load vector b shape: {b_vector.shape}")
    
    # Step 4: Apply Dirichlet boundary conditions using Algorithm VI
    if verbose:
        print("\nStep 4: Applying Dirichlet boundary conditions using Algorithm VI...")
    # boundary_nodes: shape (2, nbn)
    # boundary_nodes[0, k] = condition_type (1 for Dirichlet)
    # boundary_nodes[1, k] = node_index (0-indexed)
    boundary_nodes = np.array([[1, 1],      # Condition types: both Dirichlet
                               [0, N]])     # Node indices: 0 (left) and N (right)
    
    # Define boundary value function
    def g_func(x):
        if abs(x - a) < 1e-10:
            return ga
        elif abs(x - b) < 1e-10:
            return gb
        else:
            return 0.0  # Should not be called for interior nodes
    
    A_mod, b_mod = apply_dirichlet_boundary_conditions(A, b_vector, P, boundary_nodes, g_func)
    if verbose:
        print(f"  Applied boundary conditions at nodes: 0 and {N}")
    
    # Step 5: Solve the linear system
    if verbose:
        print("\nStep 5: Solving linear system Au = b...")
    try:
        u_fem = np.linalg.solve(A_mod, b_mod.flatten())
        if verbose:
            print("  System solved successfully")
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"  Warning: Linear system is singular or ill-conditioned: {e}")
            print(f"  Matrix condition number: {np.linalg.cond(A_mod):.2e}")
        # Use least squares solution as fallback
        u_fem = np.linalg.lstsq(A_mod, b_mod.flatten(), rcond=None)[0]
        if verbose:
            print("  Using least squares solution")
    
    return x_nodes, u_fem, P, T, A, b_vector


def main():
    """
    Main function to test the assembly-based FEM implementation
    """
    print("=" * 70)
    print("Example 1: 1D Second-Order Elliptic Equation - FEM Assembly")
    print("Algorithms IV, V, and VI")
    print("=" * 70)
    
    # Problem setup
    a, b = 0.0, 1.0  # Domain [0, 1]
    N = 10  # Number of elements
    ga, gb = 0.0, np.cos(1.0)  # Boundary conditions: u(0) = 0, u(1) = cos(1)
    
    # Define coefficient and source functions
    def c_func(x):
        """Coefficient function c(x) = e^x"""
        return np.exp(x)
    
    def f_func(x):
        """Source function f(x) = -e^x[cos(x) - 2sin(x) - x cos(x) - x sin(x)]"""
        return -np.exp(x) * (np.cos(x) - 2*np.sin(x) - x*np.cos(x) - x*np.sin(x))
    
    print(f"\nProblem: -d/dx(e^x * du/dx) = -e^x[cos(x) - 2sin(x) - x cos(x) - x sin(x)]")
    print(f"Domain: [{a}, {b}]")
    print(f"Boundary conditions: u({a}) = {ga}, u({b}) = {gb:.6f}")
    print(f"c(x) = e^x")
    print(f"f(x) = -e^x[cos(x) - 2sin(x) - x cos(x) - x sin(x)]")
    print(f"Analytical solution: u(x) = x cos(x)")
    print(f"Number of elements: {N}")
    print(f"Mesh size: h = {(b-a)/N:.4f}")
    print()
    
    # Solve using FEM assembly method
    x_nodes, u_fem, P, T, A, b_vector = solve_1d_elliptic_fem_assembly(
        a, b, N, c_func, f_func, ga, gb
    )
    
    # Compute analytical solution
    print("\n" + "=" * 70)
    print("COMPUTING ANALYTICAL SOLUTION")
    print("=" * 70)
    u_analytical = analytical_solution(x_nodes)
    
    # Compute errors
    l2_error, linf_error = compute_error(u_fem, u_analytical)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"L2 Error: {l2_error:.6e}")
    print(f"L∞ Error: {linf_error:.6e}")
    print()
    
    # Display solution at selected points
    print("Solution comparison at selected points:")
    print("-" * 70)
    print(f"{'x':<10} {'FEM':<15} {'Analytical':<15} {'Error':<15}")
    print("-" * 70)
    
    for i in range(0, len(x_nodes), max(1, len(x_nodes)//10)):
        x = x_nodes[i]
        u_fem_val = u_fem[i]
        u_anal_val = u_analytical[i]
        error = abs(u_fem_val - u_anal_val)
        print(f"{x:<10.3f} {u_fem_val:<15.6f} {u_anal_val:<15.6f} {error:<15.6e}")
    
    # Display matrix information
    print("\n" + "=" * 70)
    print("MATRIX AND VECTOR INFORMATION")
    print("=" * 70)
    print(f"P matrix (first 5 columns):")
    print(P[:, :min(5, P.shape[1])])
    print()
    
    print(f"T matrix (first 5 columns):")
    print(T[:, :min(5, T.shape[1])])
    print()
    
    print("Stiffness matrix A (first 5x5 block):")
    print(A[:min(5, A.shape[0]), :min(5, A.shape[1])])
    print()
    
    print("Load vector b:")
    print(b_vector.flatten())
    
    # Compute maximum absolute errors for different mesh sizes
    print("\n" + "=" * 70)
    print("MAXIMUM NUMERICAL ERRORS AT ALL MESH NODES")
    print("=" * 70)
    
    # Mesh sizes from the table: h = 1/4, 1/8, 1/16, 1/32, 1/64, 1/128
    h_values_table = [1/4, 1/8, 1/16, 1/32, 1/64, 1/128]
    max_errors = []
    
    print(f"\n{'h':<15} {'Maximum absolute error at all nodes':<40}")
    print("-" * 70)
    
    for h in h_values_table:
        # Calculate number of elements N from h
        N_h = int((b - a) / h)
        if N_h < 1:
            continue
        
        # Solve for this mesh size (suppress verbose output)
        x_h, u_fem_h, _, _, _, _ = solve_1d_elliptic_fem_assembly(
            a, b, N_h, c_func, f_func, ga, gb, verbose=False
        )
        u_anal_h = analytical_solution(x_h)
        
        # Compute maximum absolute error
        max_error = np.max(np.abs(u_fem_h - u_anal_h))
        max_errors.append(max_error)
        
        # Format error in scientific notation
        error_str = f"{max_error:.4e}"
        print(f"{h:<15.6f} {error_str:<40}")
    
    print("-" * 70)
    
    # Verify convergence rate (should be approximately O(h²))
    if len(max_errors) >= 2:
        print("\nConvergence rate check (error ratio when h is halved):")
        for i in range(1, len(max_errors)):
            ratio = max_errors[i-1] / max_errors[i]
            print(f"  Error({h_values_table[i-1]}) / Error({h_values_table[i]}) = {ratio:.4f} (expected ~4.0 for O(h²))")
    
    # Plot results
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Solution comparison
    plt.subplot(2, 3, 1)
    plt.plot(x_nodes, u_fem, 'bo-', label='FEM Solution (Assembly)', markersize=4)
    plt.plot(x_nodes, u_analytical, 'r--', label='Analytical Solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    plt.subplot(2, 3, 2)
    error = np.abs(u_fem - u_analytical)
    plt.plot(x_nodes, error, 'go-', markersize=4)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Convergence study
    plt.subplot(2, 3, 3)
    N_values = [5, 10, 20, 40, 80]
    l2_errors = []
    linf_errors = []
    
    for n in N_values:
        x_n, u_fem_n, _, _, _, _ = solve_1d_elliptic_fem_assembly(
            a, b, n, c_func, f_func, ga, gb, verbose=False
        )
        u_anal_n = analytical_solution(x_n)
        l2_err, linf_err = compute_error(u_fem_n, u_anal_n)
        l2_errors.append(l2_err)
        linf_errors.append(linf_err)
    
    h_values = [(b-a)/n for n in N_values]
    plt.loglog(h_values, l2_errors, 'bo-', label='L2 Error')
    plt.loglog(h_values, linf_errors, 'ro-', label='L∞ Error')
    plt.loglog(h_values, [h**2 for h in h_values], 'k--', label='O(h²)')
    plt.xlabel('Mesh size h')
    plt.ylabel('Error')
    plt.title('Convergence Study')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Stiffness matrix visualization
    plt.subplot(2, 3, 4)
    plt.imshow(A, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Stiffness Matrix A')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    
    # Plot 5: P matrix visualization
    plt.subplot(2, 3, 5)
    plt.plot(P[0, :], np.zeros_like(P[0, :]), 'bo', markersize=8, label='Nodes')
    plt.xlabel('x coordinate')
    plt.ylabel('y (zero)')
    plt.title('Node Coordinates (P matrix)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: T matrix visualization (element connectivity)
    plt.subplot(2, 3, 6)
    for n in range(T.shape[1]):
        node1 = T[0, n]
        node2 = T[1, n]
        x1 = P[0, node1]
        x2 = P[0, node2]
        plt.plot([x1, x2], [0, 0], 'g-', linewidth=2, alpha=0.6)
        plt.plot([x1, x2], [0, 0], 'go', markersize=6)
    plt.xlabel('x coordinate')
    plt.ylabel('y (zero)')
    plt.title('Element Connectivity (T matrix)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('FEM_assembly_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'FEM_assembly_results.png'")
    plt.show()
    
    return x_nodes, u_fem, u_analytical, l2_error, linf_error, P, T


if __name__ == "__main__":
    # Run the test
    x, u_fem, u_anal, l2_err, linf_err, P, T = main()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Final L2 error: {l2_err:.6e}")
    print(f"Final L∞ error: {linf_err:.6e}")
    print(f"P matrix shape: {P.shape}")
    print(f"T matrix shape: {T.shape}")

