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

