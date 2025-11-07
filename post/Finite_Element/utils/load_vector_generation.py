import numpy as np
try:
    from .helper import compute_integral
except ImportError:
    from helper import compute_integral

def generate_load_vector_by_linear(node_sum, mesh_size, f_func=None):
    """
    Generate load vector for 1D finite element method
    
    For the equation -d/dx(c(x) * du/dx) = f(x), the load vector is:
    b[i] = ∫[0,1] f(x) * φ_i(x) dx
    
    Parameters:
    node_sum (int): Number of internal nodes (N)
    mesh_size (float): Mesh size (h)
    f_func (callable): Function f(x) for computing integrals. If None, uses f(x) = 1
    
    Returns:
    numpy.ndarray: Load vector b of size (N+1) x 1
    """
    N = node_sum
    h = mesh_size
    
    # Default f(x) = 1 if no function provided
    if f_func is None:
        f_func = lambda x: 1.0
    
    # Initialize the load vector
    b = np.zeros((N + 1, 1))
    
    # For 1D linear elements, the load vector is computed as:
    # b[i] = ∫[x_{i-1}, x_i] f(x) * φ_i(x) dx + ∫[x_i, x_{i+1}] f(x) * φ_i(x) dx
    # where φ_i(x) is the linear basis function at node i
    
    # Process each element [x_i, x_{i+1}] for i = 0, 1, ..., N-1
    for i in range(N):  # i = 0 to N-1 (N elements)
        x_i = i * h
        x_i_plus_1 = (i + 1) * h
        
        # For element [x_i, x_{i+1}], the basis functions are:
        # φ_i(x) = (x_{i+1} - x) / h
        # φ_{i+1}(x) = (x - x_i) / h
        
        # Contribution to b[i] from element [x_i, x_{i+1}]
        integrand_i = lambda x: f_func(x) * (x_i_plus_1 - x) / h
        b[i] += compute_integral(integrand_i, x_i, x_i_plus_1)
        
        # Contribution to b[i+1] from element [x_i, x_{i+1}]
        integrand_i_plus_1 = lambda x: f_func(x) * (x - x_i) / h
        b[i + 1] += compute_integral(integrand_i_plus_1, x_i, x_i_plus_1)
    
    return b

def linear_basis_function_local(alpha, x, x_n, x_n_plus_1, h):
    """
    Compute the local linear basis function value
    
    For element [x_n, x_{n+1}]:
    - α=1: ψ_{n1}(x) = (x_{n+1} - x) / h
    - α=2: ψ_{n2}(x) = (x - x_n) / h
    
    Parameters:
    alpha (int): Local basis function index (1 or 2, 1-indexed)
    x (float or array): Point(s) at which to evaluate
    x_n (float): Left node of element
    x_n_plus_1 (float): Right node of element
    h (float): Element size
    
    Returns:
    float or array: Basis function value(s)
    """
    if alpha == 1:
        return (x_n_plus_1 - x) / h
    elif alpha == 2:
        return (x - x_n) / h
    else:
        raise ValueError(f"alpha must be 1 or 2 for linear elements, got {alpha}")

def generate_load_vector_by_assembly(P, T, f_func=None, quadrature_points=None, quadrature_weights=None):
    """
    Generate load vector using Algorithm V (local assembly)
    
    Algorithm V: Compute the integrals and assemble them into b
    b = zeros(N_b, 1)
    FOR n = 1, ..., N:
        FOR β = 1, ..., N_lb:
            Compute r = ∫_{x_n}^{x_{n+1}} f * ψ_{nβ} dx
            b(T_b(β, n), 1) = b(T_b(β, n), 1) + r
    
    Parameters:
    P (numpy.ndarray): Node coordinates matrix of shape (2, Nb)
                      P[:, j] contains coordinates of node j
    T (numpy.ndarray): Element connectivity matrix of shape (2, N)
                      T[:, n] contains global node indices for element n
    f_func (callable): Source function f(x). If None, uses f(x) = 1
    quadrature_points (array, optional): Gauss quadrature points for numerical integration
    quadrature_weights (array, optional): Gauss quadrature weights
    
    Returns:
    numpy.ndarray: Load vector b of size (Nb, 1)
    """
    # Extract dimensions
    Nb = P.shape[1]  # Number of global basis functions (nodes)
    N = T.shape[1]   # Number of elements
    Nlb = 2          # Number of local basis functions for linear elements
    
    # Default f(x) = 1 if no function provided
    if f_func is None:
        f_func = lambda x: 1.0
    
    # Initialize the load vector
    b = np.zeros((Nb, 1))
    
    # Algorithm V: Compute the integrals and assemble them into b
    for n in range(N):  # FOR n = 1, ..., N (0-indexed: 0, ..., N-1)
        # Get global node indices for element n
        i_global_1 = T[0, n]  # First node of element n
        i_global_2 = T[1, n]  # Second node of element n
        
        # Get coordinates of element nodes
        x_n = P[0, i_global_1]        # Left node coordinate
        x_n_plus_1 = P[0, i_global_2]  # Right node coordinate
        h = x_n_plus_1 - x_n           # Element size
        
        # Compute integrals for all local basis functions
        for beta in range(1, Nlb + 1):  # FOR β = 1, ..., N_lb (1-indexed)
            # Compute r = ∫_{x_n}^{x_{n+1}} f(x) * ψ_{nβ}(x) dx
            # Define integrand: f(x) * ψ_{nβ}(x)
            def integrand(x):
                psi_beta = linear_basis_function_local(beta, x, x_n, x_n_plus_1, h)
                return f_func(x) * psi_beta
            
            r = compute_integral(integrand, x_n, x_n_plus_1)
            
            # Map local index to global index
            # T_b(β, n) is the global index for local basis β in element n
            i_global_beta = T[beta - 1, n]  # Convert to 0-indexed
            
            # b(T_b(β, n), 1) = b(T_b(β, n), 1) + r
            b[i_global_beta, 0] += r
    
    return b

