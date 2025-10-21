import numpy as np
import matplotlib.pyplot as plt
from utils.stiff_matrix_generation import generate_stiff_matrix_by_linear
from utils.load_vector_generation import generate_load_vector_by_linear
from utils.helper import compute_integral

def solve_1d_elliptic_fem(a, b, N, c_func, f_func, ga, gb):
    """
    Solve the 1D second-order elliptic equation using Finite Element Method
    
    Problem: -d/dx(c(x) * du/dx) = f(x) for a < x < b
    Boundary conditions: u(a) = ga, u(b) = gb
    
    Parameters:
    a, b (float): Domain boundaries
    N (int): Number of internal nodes
    c_func (callable): Coefficient function c(x)
    f_func (callable): Source function f(x)
    ga, gb (float): Boundary values
    
    Returns:
    tuple: (x_nodes, u_fem, u_analytical, error)
    """
    # Mesh setup
    h = (b - a) / N  # Mesh size
    x_nodes = np.linspace(a, b, N + 1)  # All nodes including boundaries (N+1 total nodes)
    
    # Generate stiffness matrix A
    A = generate_stiff_matrix_by_linear(N, h, c_func)
    
    # Generate load vector b
    b_vector = generate_load_vector_by_linear(N, h, f_func)
    
    # Apply boundary conditions
    # For Dirichlet BC: u(a) = ga, u(b) = gb
    # We need to modify the system Au = b to account for boundary conditions
    
    # Create the full solution vector including boundary nodes
    u_full = np.zeros(N + 1)  # N+1 total nodes
    u_full[0] = ga  # u(a) = ga
    u_full[-1] = gb  # u(b) = gb
    
    # For Dirichlet boundary conditions, we need to modify the system
    # The modified system is: A_mod * u = b_mod
    # where A_mod has the boundary rows/columns zeroed out except diagonal = 1
    # and b_mod has the boundary values set
    
    A_mod = A.copy()
    b_mod = b_vector.copy()
    
    # Set boundary conditions in the matrix and RHS
    # First boundary: u[0] = ga
    A_mod[0, :] = 0
    A_mod[0, 0] = 1
    b_mod[0] = ga
    
    # Last boundary: u[N] = gb  
    A_mod[N, :] = 0
    A_mod[N, N] = 1
    b_mod[N] = gb
    
    # Solve the modified linear system
    try:
        u_full = np.linalg.solve(A_mod, b_mod.flatten())
    except np.linalg.LinAlgError as e:
        print(f"Warning: Linear system is singular or ill-conditioned: {e}")
        print(f"Matrix condition number: {np.linalg.cond(A_mod):.2e}")
        # Use least squares solution as fallback
        u_full = np.linalg.lstsq(A_mod, b_mod.flatten(), rcond=None)[0]
    
    return x_nodes, u_full

def analytical_solution(x):
    """
    Compute analytical solution for comparison
    For the example: c(x) = 1, f(x) = 2
    """
    # For our specific example: c(x) = 1, f(x) = 2
    # The equation becomes: -d^2u/dx^2 = 2
    # General solution: u(x) = -x^2 + C1*x + C2
    # Boundary conditions: u(0) = 0, u(1) = 1
    # u(0) = 0: 0 = -0 + 0 + C2 → C2 = 0
    # u(1) = 1: 1 = -1 + C1 + 0 → C1 = 2
    # This gives: u(x) = -x^2 + 2x
    
    u = -x**2 + 2*x
    return u

def compute_error(u_fem, u_analytical):
    """Compute L2 and L∞ errors"""
    l2_error = np.sqrt(np.mean((u_fem - u_analytical)**2))
    linf_error = np.max(np.abs(u_fem - u_analytical))
    return l2_error, linf_error

def main():
    """
    Main function to solve the 1D elliptic equation with numerical example
    """
    print("=" * 60)
    print("1D Second-Order Elliptic Equation - Finite Element Method")
    print("=" * 60)
    
    # Problem setup
    a, b = 0.0, 1.0  # Domain [0, 1]
    N = 10  # Number of internal nodes
    ga, gb = 0.0, 1.0  # Boundary conditions
    
    # Define coefficient and source functions
    def c_func(x):
        """Coefficient function c(x) = 1"""
        return np.ones_like(x)
    
    def f_func(x):
        """Source function f(x) = 2"""
        return 2 * np.ones_like(x)
    
    print(f"Problem: -d/dx(c(x) * du/dx) = f(x)")
    print(f"Domain: [{a}, {b}]")
    print(f"Boundary conditions: u({a}) = {ga}, u({b}) = {gb}")
    print(f"c(x) = 1, f(x) = 2")
    print(f"Number of internal nodes: {N}")
    print(f"Mesh size: h = {(b-a)/N:.4f}")
    print()
    
    # Solve using FEM
    print("Solving using Finite Element Method...")
    x_nodes, u_fem = solve_1d_elliptic_fem(a, b, N, c_func, f_func, ga, gb)
    
    # Compute analytical solution
    print("Computing analytical solution...")
    u_analytical = analytical_solution(x_nodes)
    
    # Compute errors
    l2_error, linf_error = compute_error(u_fem, u_analytical)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"L2 Error: {l2_error:.6e}")
    print(f"L∞ Error: {linf_error:.6e}")
    print()
    
    # Display solution at selected points
    print("Solution comparison at selected points:")
    print("-" * 50)
    print(f"{'x':<8} {'FEM':<12} {'Analytical':<12} {'Error':<12}")
    print("-" * 50)
    
    for i in range(0, len(x_nodes), max(1, len(x_nodes)//10)):
        x = x_nodes[i]
        u_fem_val = u_fem[i]
        u_anal_val = u_analytical[i]
        error = abs(u_fem_val - u_anal_val)
        print(f"{x:<8.3f} {u_fem_val:<12.6f} {u_anal_val:<12.6f} {error:<12.6e}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Solution comparison
    plt.subplot(2, 2, 1)
    plt.plot(x_nodes, u_fem, 'bo-', label='FEM Solution', markersize=4)
    plt.plot(x_nodes, u_analytical, 'r--', label='Analytical Solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Solution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    error = np.abs(u_fem - u_analytical)
    plt.plot(x_nodes, error, 'go-', markersize=4)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Convergence study
    plt.subplot(2, 2, 3)
    N_values = [5, 10, 20, 40, 80]
    l2_errors = []
    linf_errors = []
    
    for n in N_values:
        x_n, u_fem_n = solve_1d_elliptic_fem(a, b, n, c_func, f_func, ga, gb)
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
    
    # Plot 4: Matrix visualization
    plt.subplot(2, 2, 4)
    A = generate_stiff_matrix_by_linear(N, (b-a)/N, c_func)
    plt.imshow(A, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Stiffness Matrix A')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    
    plt.tight_layout()
    plt.savefig('FEM_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print matrix and vector information
    print("\n" + "=" * 60)
    print("MATRIX AND VECTOR INFORMATION")
    print("=" * 60)
    A = generate_stiff_matrix_by_linear(N, (b-a)/N, c_func)
    b_vector = generate_load_vector_by_linear(N, (b-a)/N, f_func)
    print(f"Stiffness matrix A shape: {A.shape}")
    print(f"Load vector b shape: {b_vector.shape}")
    print(f"Condition number of A: {np.linalg.cond(A):.2e}")
    print()
    
    print("Stiffness matrix A (first 5x5 block):")
    print(A[:min(5, A.shape[0]), :min(5, A.shape[1])])
    print()
    
    print("Load vector b:")
    print(b_vector.flatten())
    
    return x_nodes, u_fem, u_analytical, l2_error, linf_error

if __name__ == "__main__":
    # Run the main example
    x, u_fem, u_anal, l2_err, linf_err = main()
    
    print(f"\nExample completed successfully!")
    print(f"Final L2 error: {l2_err:.6e}")
    print(f"Final L∞ error: {linf_err:.6e}")
