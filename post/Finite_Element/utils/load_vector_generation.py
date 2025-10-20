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

