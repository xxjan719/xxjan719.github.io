import numpy as np
try:
    from .helper import compute_integral
except:
    from helper import compute_integral

def generate_stiff_matrix_by_linear(node_sum, mesh_size, c_func=None):
    """
    Generate stiffness matrix for 1D finite element method
    
    For the equation -d/dx(c(x) * du/dx) = f(x), the stiffness matrix is:
    A[i,j] = ∫[0,1] c(x) * φ'_i(x) * φ'_j(x) dx
    
    Parameters:
    node_sum (int): Number of internal nodes (N)
    mesh_size (float): Mesh size (h)
    c_func (callable): Function c(x) for computing integrals. If None, uses c(x) = 1
    
    Returns:
    numpy.ndarray: Stiffness matrix A of size (N+1) x (N+1)
    """
    N = node_sum
    h = mesh_size
    
    # Default c(x) = 1 if no function provided
    if c_func is None:
        c_func = lambda x: 1.0
    
    # Initialize the matrix
    A = np.zeros((N + 1, N + 1))
    
    # For 1D linear elements, the stiffness matrix has the form:
    # A[i,i] = (1/h) * ∫[x_i, x_{i+1}] c(x) dx + (1/h) * ∫[x_{i-1}, x_i] c(x) dx
    # A[i,i+1] = A[i+1,i] = -(1/h) * ∫[x_i, x_{i+1}] c(x) dx
    
    # Process each element [x_i, x_{i+1}] for i = 0, 1, ..., N-1
    for i in range(N):  # i = 0 to N-1 (N elements)
        x_i = i * h
        x_i_plus_1 = (i + 1) * h
        
        # Compute integral of c(x) over element [x_i, x_{i+1}]
        c_integral = compute_integral(c_func, x_i, x_i_plus_1)
        element_contribution = c_integral / h
        
        # Add contribution to diagonal terms
        A[i, i] += element_contribution
        A[i + 1, i + 1] += element_contribution
        
        # Add contribution to off-diagonal terms
        A[i, i + 1] -= element_contribution
        A[i + 1, i] -= element_contribution
    
    # Scale the matrix by 1/h for the 1D Poisson equation
    A = A / h
    
    return A


def linear_basis_derivative_local(alpha, h):
    """
    Compute the derivative of local linear basis function
    
    For element [x_n, x_{n+1}], the local basis functions are:
    - α=1 (first local basis): ψ_{n1}(x) = (x_{n+1} - x) / h, derivative = -1/h
    - α=2 (second local basis): ψ_{n2}(x) = (x - x_n) / h, derivative = 1/h
    
    Parameters:
    alpha (int): Local basis function index (1 or 2, 1-indexed)
    h (float): Element size
    
    Returns:
    float: Constant derivative value
    """
    if alpha == 1:
        return -1.0 / h
    elif alpha == 2:
        return 1.0 / h
    else:
        raise ValueError(f"alpha must be 1 or 2 for linear elements, got {alpha}")


def generate_stiff_matrix_by_assembly(P, T, c_func=None, quadrature_points=None, quadrature_weights=None):
    """
    Generate stiffness matrix using Algorithm IV (local assembly)
    
    Algorithm IV: Compute and assemble integrals into A
    FOR n = 1, ..., N:
        FOR α = 1, ..., Nlb:
            FOR β = 1, ..., Nlb:
                Compute r = ∫_{x_n}^{x_{n+1}} c ψ'_{nα} ψ'_{nβ} dx
                Add r to A(Tb(β, n), Tb(α, n))
    
    Parameters:
    P (numpy.ndarray): Node coordinates matrix of shape (2, Nb)
                      P[:, j] contains coordinates of node j
    T (numpy.ndarray): Element connectivity matrix of shape (2, N)
                      T[:, n] contains global node indices for element n
    c_func (callable): Coefficient function c(x). If None, uses c(x) = 1
    quadrature_points (array, optional): Gauss quadrature points for numerical integration
    quadrature_weights (array, optional): Gauss quadrature weights
    
    Returns:
    numpy.ndarray: Stiffness matrix A of size (Nb, Nb)
    """
    # Extract dimensions
    Nb = P.shape[1]  # Number of global basis functions (nodes)
    N = T.shape[1]   # Number of elements
    Nlb = 2          # Number of local basis functions for linear elements
    
    # Default c(x) = 1 if no function provided
    if c_func is None:
        c_func = lambda x: 1.0
    
    # Initialize the matrix (sparse would be better, but using dense for simplicity)
    A = np.zeros((Nb, Nb))
    
    # Algorithm IV: Compute the integrals and assemble them into A
    for n in range(N):  # FOR n = 1, ..., N (0-indexed: 0, ..., N-1)
        # Get global node indices for element n
        i_global_1 = T[0, n]  # First node of element n
        i_global_2 = T[1, n]  # Second node of element n
        
        # Get coordinates of element nodes
        x_n = P[0, i_global_1]        # Left node coordinate
        x_n_plus_1 = P[0, i_global_2]  # Right node coordinate
        h = x_n_plus_1 - x_n           # Element size
        
        # For linear elements, the derivatives are constant
        # ψ'_{n1} = -1/h, ψ'_{n2} = 1/h
        
        # Compute derivatives for local basis functions
        psi_prime_1 = linear_basis_derivative_local(1, h)  # -1/h
        psi_prime_2 = linear_basis_derivative_local(2, h)  # 1/h
        
        # Compute integrals for all combinations of α and β
        for alpha in range(1, Nlb + 1):  # FOR α = 1, ..., Nlb (1-indexed)
            for beta in range(1, Nlb + 1):  # FOR β = 1, ..., Nlb (1-indexed)
                # Get derivatives
                if alpha == 1:
                    psi_prime_alpha = psi_prime_1
                else:
                    psi_prime_alpha = psi_prime_2
                
                if beta == 1:
                    psi_prime_beta = psi_prime_1
                else:
                    psi_prime_beta = psi_prime_2
                
                # Compute r = ∫_{x_n}^{x_{n+1}} c(x) * ψ'_{nα} * ψ'_{nβ} dx
                # Since derivatives are constant, this simplifies to:
                # r = ψ'_{nα} * ψ'_{nβ} * ∫_{x_n}^{x_{n+1}} c(x) dx
                integrand = lambda x: c_func(x)
                c_integral = compute_integral(integrand, x_n, x_n_plus_1)
                r = psi_prime_alpha * psi_prime_beta * c_integral
                
                # Map local indices to global indices
                # Tb(β, n) is the global index for local basis β in element n
                # Tb(α, n) is the global index for local basis α in element n
                i_global_beta = T[beta - 1, n]  # Convert to 0-indexed
                i_global_alpha = T[alpha - 1, n]  # Convert to 0-indexed
                
                # Add r to A(Tb(β, n), Tb(α, n))
                A[i_global_beta, i_global_alpha] += r
    
    return A

