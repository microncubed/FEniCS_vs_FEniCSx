import fenics as fe
import numpy as np

def fenics_poisson_test(nx,ny,fe_order,exp_order):
    '''
    A test of a Poisson equation solver in FEniCS. The method of manufactured solutions 
    is used to generate the source term for the solution u_a = np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)
    over the square simulation domain defined by the points (0,0) and (1,1). A Neumann
    boundary condition is used.
    
    Parameters
    ----------
    nx int : number of grid points in x
    ny int : number of grid points in y
    fe_order int : order of the finite element mesh
    exp_order int : order to which the analytic source term is projected onto the finite element mesh
    
    Outputs
    ----------
    u_fd np.ndarray : the finite difference solution
    u_a np.ndarray : the analytic solution
    L2 float : the L2 norm between u_a and u_fd
    '''
    
    length = 1

    # Make the grid and the function space
    mesh = fe.RectangleMesh(fe.Point(0,0),fe.Point(length,length),nx-1,ny-1)
    V = fe.FunctionSpace(mesh,'CG', fe_order)

    # The source term
    q = fe.Expression('8*pi*pi*cos(2*pi*x[0])*cos(2*pi*x[1])', degree = exp_order)
    q_int = fe.interpolate(q,V)

    # Define the equation and solve it
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    a = fe.dot(fe.grad(u),fe.grad(v))*fe.dx
    L = q_int*v*fe.dx
    u = fe.Function(V)
    fe.solve(a == L,u)

    # Get the finite difference solution in array form
    u_vv = u.compute_vertex_values()
    u_fd = np.zeros((ny,nx))
    for i in range(nx):
        for j in range(ny):
            u_fd[j,i]=u_vv[i+nx*j]

    # The analytic solution on the same grid as the finite difference solution
    x_vals = np.linspace(0, length, nx)
    y_vals = np.linspace(0, length, ny)
    X, Y = np.meshgrid(x_vals,y_vals)
    u_a = np.cos(2*np.pi*X)*np.cos(2*np.pi*Y)
    
    # Interpolate the analytic solution over the grid and work out the L2 norm
    u_aa =  fe.Expression('cos(2*pi*x[0])*cos(2*pi*x[1])', degree = exp_order)
    norm = fe.errornorm(u_aa,u,'L2')

    return u_fd, u_a, norm


