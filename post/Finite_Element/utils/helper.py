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