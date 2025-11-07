def apply_dirichlet_boundary_conditions(A, b, P, boundary_nodes, g_func):
    """
    Apply Dirichlet boundary conditions using Algorithm VI
    
    Algorithm VI: Deal with the Dirichlet boundary conditions
    FOR k = 1, ..., nbn:
        IF boundarynodes(1, k) shows Dirichlet condition, THEN
            i = boundarynodes(2, k)
            A(i, :) = 0
            A(i, i) = 1
            b(i) = g(Pb(i))
        ENDIF
    END
    
    Parameters:
    A (numpy.ndarray): Stiffness matrix of shape (Nb, Nb)
    b (numpy.ndarray): Load vector of shape (Nb, 1)
    P (numpy.ndarray): Node coordinates matrix of shape (2, Nb)
    boundary_nodes (numpy.ndarray): Boundary nodes array of shape (2, nbn)
                                    boundary_nodes[0, k] = condition type (1 for Dirichlet)
                                    boundary_nodes[1, k] = global node index (0-indexed)
    g_func (callable): Boundary value function g(x) that returns the prescribed value
    
    Returns:
    tuple: (A_mod, b_mod) - Modified stiffness matrix and load vector
    """
    A_mod = A.copy()
    b_mod = b.copy()
    
    nbn = boundary_nodes.shape[1]  # Number of boundary nodes
    
    # Algorithm VI: Deal with the Dirichlet boundary conditions
    for k in range(nbn):  # FOR k = 1, ..., nbn (0-indexed: 0, ..., nbn-1)
        condition_type = boundary_nodes[0, k]  # boundarynodes(1, k)
        
        # IF boundarynodes(1, k) shows Dirichlet condition, THEN
        if condition_type == 1:  # 1 indicates Dirichlet condition
            i = int(boundary_nodes[1, k])  # i = boundarynodes(2, k)
            
            # A(i, :) = 0
            A_mod[i, :] = 0
            
            # A(i, i) = 1
            A_mod[i, i] = 1
            
            # b(i) = g(Pb(i))
            # Pb(i) is the coordinate of node i
            x_i = P[0, i]  # Get x-coordinate of node i
            b_mod[i, 0] = g_func(x_i)
    
    return A_mod, b_mod