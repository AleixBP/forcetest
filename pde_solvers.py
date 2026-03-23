from __future__ import print_function, division, absolute_import
import dolfin as dl
import hippylib as hp
import numpy as np
import logging

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

def boundary_T(x, on_boundary):
    return (x[1] > 1.0 - dl.DOLFIN_EPS) and on_boundary 

def boundary_B(x, on_boundary):
    return (x[1] < dl.DOLFIN_EPS) and on_boundary 

def boundary_R(x, on_boundary):
    return (x[0] > 1.0 - dl.DOLFIN_EPS) and on_boundary 

def boundary_L(x, on_boundary):
    return (x[0] < dl.DOLFIN_EPS) and on_boundary

class HessianOperator():
    """Matrix-free reduced Hessian operator for the Newton-CG solver."""
    cgiter = 0

    def __init__(self, R, C, A, adj_A, W, bcs, use_gaussnewton=False):
        self.R = R
        self.C = C
        self.A = A
        self.adj_A = adj_A
        self.W = W
        self.bcs = bcs
        self.use_gaussnewton = use_gaussnewton

        # incremental state
        self.du = dl.Vector()
        self.A.init_vector(self.du, 0)

        # incremental adjoint
        self.dp = dl.Vector()
        self.adj_A.init_vector(self.dp, 0)

    def init_vector(self, v, dim):
        self.R.init_vector(v, dim)

    # Hessian performed on v, output as generic vector y
    def mult(self, v, y):
        self.cgiter += 1
        y.zero()
        if self.use_gaussnewton:
            self.mult_GaussNewton(v,y)
        else:
            self.mult_Newton(v,y)
            
    def mult_GaussNewton(self, v, y):
        """Gauss-Newton Hessian-vector product: H_gn * v."""
        # incremental forward
        rhs = -(self.C * v)
        for bc in self.bcs:
            bc.apply(rhs)
        dl.solve(self.A, self.du, rhs)

        # incremental adjoint
        rhs = -(self.W * self.du)
        for bc in self.bcs:
            bc.apply(rhs)
        dl.solve(self.adj_A, self.dp, rhs)

        # misfit term
        self.C.transpmult(self.dp, y)

        # regularization term
        if self.R:
            y.axpy(1., self.R * v)

    def mult_Newton(self, v, y):
        """Full Newton Hessian-vector product: H * v."""
        # incremental forward
        rhs = -(self.C * v)
        for bc in self.bcs:
            bc.apply(rhs)
        dl.solve(self.A, self.du, rhs)

        # incremental adjoint
        rhs = -(self.W * self.du)
        for bc in self.bcs:
            bc.apply(rhs)
        dl.solve(self.adj_A, self.dp, rhs)

        # misfit term
        self.C.transpmult(self.dp, y)

        # regularization term
        if self.R:
            y.axpy(1., self.R * v)


def cost(u, uw0, I1, I2, gradI2, f, gamma, noise_var, BOOL_REG=0, eps_TV=1e-7):
    """Evaluate total cost = misfit + regularization."""
    if BOOL_REG == 0:
        reg = 0.5 * gamma * dl.assemble(dl.inner(f, f) * dl.dx)
    elif BOOL_REG == 1:
        reg = 0.5 * gamma * dl.assemble((dl.inner(dl.grad(f), dl.grad(f)) + eps_TV * dl.inner(f, f)) * dl.dx)
    else:
        eps = dl.Constant(eps_TV)
        reg = 0.5 * gamma * dl.assemble(dl.sqrt(dl.inner(dl.grad(f), dl.grad(f)) + eps) * dl.dx)

    residual = dl.inner(gradI2, u - uw0) + I2 - I1
    misfit = (0.5 / noise_var) * dl.assemble(dl.inner(residual, residual) * dl.dx)
    return [reg + misfit, misfit, reg]


def setup_problem(mesh, I1, I2, gradI2, uw0, gamma, noise_var, BOOL_REG=0, eps_TV=1e-7, mu=1.0, lam=1.9762):
    """
    Build function spaces, weak forms, and BCs for the TFM problem.

    Returns:
        f, up, vq              -- mutable FE functions (updated in-place by the solver)
        Vf                     -- force function space (needed to create temporaries)
        bcs_state, bcs_adj     -- boundary conditions
        a_state, L_state       -- state bilinear / linear forms
        a_adj, L_adj           -- adjoint bilinear / linear forms
        GM_varf, GR_varf, M   -- gradient forms and mass matrix
        L_incr_adj, HM_varf, HR_varf -- Hessian sub-matrix bilinear forms
    """
    i_noise_var = 1. / noise_var

    # force space
    Vf = dl.VectorFunctionSpace(mesh, 'Lagrange', 1)
    f = dl.Function(Vf)
    f_trial = dl.TrialFunction(Vf)
    f_test = dl.TestFunction(Vf)

    # state space (Taylor-Hood: P2-P1)
    P2 = dl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = dl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W_space = dl.FunctionSpace(mesh, TH)

    up = dl.Function(W_space)
    (u_trial, p_trial) = dl.TrialFunctions(W_space)
    (u_test, p_test) = dl.TestFunctions(W_space)

    vq = dl.Function(W_space)
    (v_trial, q_trial) = dl.TrialFunctions(W_space)
    (v_test, q_test) = dl.TestFunctions(W_space)

    u, p = up.split()
    v, q = vq.split()

    # boundary conditions (homogeneous Dirichlet on all sides)
    zero_vec = dl.Constant((0., 0.))
    bcs_state = [dl.DirichletBC(W_space.sub(0), zero_vec, bc_func)
                 for bc_func in [boundary_T, boundary_B, boundary_R, boundary_L]]
    bcs_adj = [dl.DirichletBC(W_space.sub(0), zero_vec, bc_func)
               for bc_func in [boundary_T, boundary_B, boundary_R, boundary_L]]

    # constitutive law
    def epsilon(u):
        return 0.5 * (dl.grad(u) + dl.grad(u).T)

    def sigma(u, p):
        d = u.geometric_dimension()
        return -p * dl.Identity(d) + 2 * mu * epsilon(u)

    # state equation
    a_state = dl.inner(sigma(u_trial, p_trial), epsilon(u_test)) * dl.dx \
            - p_test * dl.div(u_trial) * dl.dx \
            - (p_trial * p_test / lam) * dl.dx # model
    L_state = dl.inner(f, u_test) * dl.dx # model

    # adjoint equation
    a_adj = dl.inner(sigma(v_trial, q_trial), epsilon(v_test)) * dl.dx \
          - q_test * dl.div(v_trial) * dl.dx \
          - (q_trial * q_test / lam) * dl.dx # model
    residual = dl.inner(gradI2, u - uw0) + I2 - I1 # image data
    L_adj = -i_noise_var * dl.inner(residual, dl.inner(gradI2, v_test)) * dl.dx # image data

    # gradient forms
    GM_varf = -dl.inner(f_test, v) * dl.dx # model
    if BOOL_REG == 0: # regularization
        GR_varf = gamma * dl.inner(f, f_test) * dl.dx # L2
    elif BOOL_REG == 1:
        GR_varf = gamma * (dl.inner(dl.grad(f), dl.grad(f_test)) + eps_TV * dl.inner(f, f_test)) * dl.dx # H1
    else:
        eps = dl.Constant(eps_TV)
        GR_varf = dl.Constant(gamma) * ( dl.inner(dl.grad(f), dl.grad(f_test)) / dl.sqrt(dl.inner(dl.grad(f), dl.grad(f)) + eps) ) * dl.dx

    # Hessian sub-matrix bilinear forms
    #a_incr_state = same as a_state
    #L_incr_state = same as L_state
    #a_incr_adj = same as a_adj
    L_incr_adj = i_noise_var * dl.inner(dl.inner(gradI2, u_trial), dl.inner(gradI2, u_test)) * dl.dx # image data
    HM_varf = -dl.inner(f_trial, u_test) * dl.dx # model
    if BOOL_REG == 0: # regularization
        HR_varf = dl.Constant(gamma) * dl.inner(f_trial, f_test) * dl.dx # L2
    elif BOOL_REG == 1:
        HR_varf = dl.Constant(gamma) * (dl.inner(dl.grad(f_trial), dl.grad(f_test)) + dl.Constant(eps_TV) * dl.inner(f_trial, f_test)) * dl.dx # H1
    else:
        HR_varf = dl.Constant(gamma) * ( dl.inner(dl.grad(f_trial), dl.grad(f_test)) / dl.sqrt(dl.inner(dl.grad(f), dl.grad(f)) + eps) ) * dl.dx

    # L^2 FEM mass matrix (inner product identity for preconditioning)
    M = dl.assemble(dl.inner(f_trial, f_test) * dl.dx)

    return (f, up, vq, Vf, # variables
            bcs_state, bcs_adj, # boundary conditions
            a_state, L_state, a_adj, L_adj, # weak forms of forward / adjoint PDEs
            GM_varf, GR_varf, M, # gradient forms and mass matrix
            L_incr_adj, HM_varf, HR_varf) # Hessian sub-matrix forms


def flsolver(mesh, I1, I2, gradI2, f0, uw0, gamma, noise_var, tol=1e-4, maxiter=3, c=1e-4, BOOL_REG=0, eps_TV=1e-7, mu=1.0, lam=1.9762):
    """
    PDE-constrained optimization solver for traction force microscopy.

    Solves: min_{f}  1/(2*sigma^2) * ||grad(I2) . (u - uw0) + I2 - I1||^2  +  gamma/2 * ||f||^2
            s.t.     -div(sigma(u,p)) = f,  div(u) = p/lambda

    Uses an inexact Newton-CG method with Armijo line search.

    Returns:
        mesh, u, f, I1, I2
    """
    (f, up, vq, Vf,
     bcs_state, bcs_adj,
     a_state, L_state, a_adj, L_adj,
     GM_varf, GR_varf, M,
     L_incr_adj, HM_varf, HR_varf) = setup_problem(mesh, I1, I2, gradI2, uw0, gamma, noise_var, BOOL_REG, eps_TV, mu, lam)

    # initial guess: assign f0, solve forward
    f.assign(f0)
    state_A, state_b = dl.assemble_system(a_state, L_state, bcs_state)
    dl.solve(state_A, up.vector(), state_b)

    # evaluate initial cost
    [cost_old, _, _] = cost(up.split()[0], uw0, I1, I2, gradI2, f, gamma, noise_var, BOOL_REG, eps_TV)

    ## NEWTON-CG OPTIMIZATION
    R = dl.assemble(HR_varf)
    g, f_delta = dl.Vector(), dl.Vector()
    R.init_vector(f_delta, 0)
    R.init_vector(g, 0)
    f_prev = dl.Function(Vf)

    iter = 1
    total_cg_iter = 0
    converged = False

    print("  Nit | CGit | Cost       | Misfit     | Reg        | ||Grad||   | alpha | rel ||df||")
    print("  " + "-" * 88)

    while iter < maxiter and not converged:

        # adjoint solve
        adjoint_A, adjoint_RHS = dl.assemble_system(a_adj, L_adj, bcs_adj)
        dl.solve(adjoint_A, vq.vector(), adjoint_RHS)

        # gradient
        MG = dl.assemble(GM_varf + GR_varf)
        dl.solve(M, g, MG)
        gradnorm = np.sqrt(g.inner(MG))

        # Eisenstat-Walker CG tolerance
        if iter == 1:
            gradnorm_ini = gradnorm
        tolcg = min(0.5, np.sqrt(gradnorm / gradnorm_ini))

        # assemble Hessian sub-matrices
        C = dl.assemble(HM_varf)
        W = dl.assemble(L_incr_adj)
        R = dl.assemble(HR_varf)

        # Hessian operator and preconditioner
        Hess = HessianOperator(R, C, state_A, adjoint_A, W, bcs_adj,
                               use_gaussnewton=(iter < 6))
        P = R + 1e-6 * M # regularization preconditioner with small shift for numerical stability
        Psolver = dl.PETScKrylovSolver("cg", hp.amg_method())
        Psolver.set_operator(P)

        solver = hp.CGSolverSteihaug()
        solver.set_operator(Hess)
        solver.set_preconditioner(Psolver)
        solver.parameters["rel_tolerance"] = tolcg
        solver.parameters["zero_initial_guess"] = True
        solver.parameters["print_level"] = -1

        solver.solve(f_delta, -MG)
        total_cg_iter += Hess.cgiter

        # Armijo line search
        alpha = 1.
        descent = False
        no_backtrack = 0
        f_prev.assign(f)
        while not descent and no_backtrack < 10:
            f.vector().axpy(alpha, f_delta)

            state_A, state_b = dl.assemble_system(a_state, L_state, bcs_state)
            dl.solve(state_A, up.vector(), state_b)

            [cost_new, misfit_new, reg_new] = cost(
                up.split()[0], uw0, I1, I2, gradI2, f, gamma, noise_var, BOOL_REG, eps_TV)

            if cost_new < cost_old + alpha * c * MG.inner(f_delta):
                cost_old = cost_new
                descent = True
            else:
                no_backtrack += 1
                alpha *= 0.5
                f.assign(f_prev)

        # L2 norm of the step relative to initial state or previous step (metric)
        df_vec = f.vector().copy()
        df_vec.axpy(-1.0, f_prev.vector())
        f_norm = f_prev.vector().norm("l2")
        rel_df_norm = df_vec.norm("l2") / f_norm if f_norm > 1e-10 else df_vec.norm("l2")

        print("  %3d | %4d | %1.4e | %1.4e | %1.4e | %1.4e | %4.2f | %1.4e" %
              (iter, Hess.cgiter, cost_new, misfit_new, reg_new, gradnorm, alpha, rel_df_norm))

        if gradnorm < tol and iter > 1:
            converged = True
            print("  -> Converged in %d iterations (total CG: %d)" % (iter, total_cg_iter))

        iter += 1

    if not converged:
        print("  -> Stopped in %d iterations (total CG: %d)" % (maxiter, total_cg_iter))

    return mesh, up.split()[0], f


def setup_of_problem(mesh, I1, I2, gradI2, gamma, noise_var=1.0, uw0=None):
    """
    Sets up the bilinear forms and operators for the Optical Flow problem,
    useful for extracting operators for covariance computation.
    """
    V = dl.VectorFunctionSpace(mesh, "CG", 1) # space for velocity
    
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)
    
    It = I2 - I1
    i_noise_var = 1.0 / noise_var
    
    a_data = i_noise_var * dl.inner(gradI2, u)*dl.inner(gradI2, v)*dl.dx
    a_reg = gamma * dl.inner(dl.nabla_grad(u), dl.nabla_grad(v))*dl.dx

    L_data = -i_noise_var * It*dl.inner(gradI2, v)*dl.dx
    
    if uw0 is not None:
        L_warp = i_noise_var * dl.inner(gradI2, uw0)*dl.inner(gradI2, v)*dl.dx
    else:
        L_warp = dl.Constant(0.0)*v[0]*dl.dx 
    
    return V, a_data, a_reg, L_data, L_warp

def ofsolver(mesh, I1, I2, gradI2, gamma, tol, noise_var=1.0, uw0=None):

    V, a_data, a_reg, L_data, L_warp = setup_of_problem(mesh, I1, I2, gradI2, gamma, noise_var, uw0)

    a = a_data + a_reg
    L = L_data + L_warp
    A, b = dl.assemble_system(a, L)

    u1 = dl.Function(V)
    dl.solve(A, u1.vector(), b) # could be done with an interative solver for speed at high image resolutions

    return mesh, u1, I1, I2

class OFHessianOperator():
    """Matrix-free representation of the Optical Flow Misfit Hessian for Hippylib Covariance."""
    def __init__(self, Hmisfit_matrix):
        self.Hmisfit_matrix = Hmisfit_matrix

    def init_vector(self, v, dim):
        self.Hmisfit_matrix.init_vector(v, dim)

    def mult(self, v, y):
        # Simply multiply the FEniCS Matrix against the vector
        self.Hmisfit_matrix.mult(v, y)