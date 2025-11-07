import numpy as np

def compute_integral(func, a, b, n_points=100):
    """
    Compute integral of func from a to b using trapezoidal rule
    
    Parameters:
    func (callable): Function to integrate
    a (float): Lower bound
    b (float): Upper bound  
    n_points (int): Number of points for numerical integration
    
    Returns:
    float: Approximate value of the integral
    """
    x = np.linspace(a, b, n_points)
    y = np.array([func(xi) for xi in x])
    return np.trapz(y, x)

def construct_P_matrix_linear(a, b, N):
    """
    Construct P matrix (node coordinates matrix) for linear finite elements
    
    For linear finite elements, P_b = P. The j-th column stores the coordinates
    of the j-th finite element node.
    
    Parameters:
    a (float): Left boundary of domain
    b (float): Right boundary of domain
    N (int): Number of elements
    
    Returns:
    numpy.ndarray: P matrix of shape (2, N+1) for 1D case
                  Each column j contains [x_j, x_j] where x_j is the coordinate of node j
    """
    h = (b - a) / N
    x_nodes = np.array([a + j * h for j in range(N + 1)])
    
    # For 1D linear elements, P is (2, N+1) where each column is [x_j, x_j]
    P = np.zeros((2, N + 1))
    P[0, :] = x_nodes
    P[1, :] = x_nodes
    
    return P

def construct_T_matrix_linear(N):
    """
    Construct T matrix (element connectivity matrix) for linear finite elements
    
    For linear finite elements, T_b = T. The n-th column stores the global node
    indices of the finite element nodes that constitute the n-th mesh element.
    
    Parameters:
    N (int): Number of elements
    
    Returns:
    numpy.ndarray: T matrix of shape (2, N)
                  The n-th column is [n, n+1]^T (0-indexed: [n, n+1]^T)
    """
    # For 1D linear elements, T is (2, N) where n-th column is [n, n+1]^T
    T = np.zeros((2, N), dtype=int)
    for n in range(N):
        T[0, n] = n      # First node of element n
        T[1, n] = n + 1  # Second node of element n
    
    return T



