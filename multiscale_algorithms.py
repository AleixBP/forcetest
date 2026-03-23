import os
nthreads = "2"
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["OPENMP_NUM_THREADS"] = nthreads
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads

import math
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from utils.dolfin_numpy_utils import array_to_scalar, vector_to_arrays, arrays_to_vector
from pde_solvers import flsolver
from utils.warp import warp, outside_indices, warp_coords
from utils.pyramid import downsample_single, downscale_single
from utils.imageDerivatives import image_first_derivatives

def multiscale_algorithm(im1, im2, gamma, tol, noise_var, finest, coarsest=8, scales=None, silent=False, BOOL_REG=0, eps_TV=1e-6, solver_type='elastic', mu=1.0, lam=1.9762):

    Ny, Nx = im1.shape
    Ly = 1e0
    Lx = Ly*Nx/Ny
    #Dx = Lx/Nx
    #Dy = Ly/Ny
    u = None

    scales = range(finest, coarsest)[::-1] if scales is None else scales
    for j, i in enumerate(scales):  # loop from coarsest scale to finest scale

        downscale_factor = math.pow(2, i)
        print(downscale_factor)

        # downsampling
        im1_down = downsample_single(im1, downscale_factor)
        im2_down = downsample_single(im2, downscale_factor)
        Ny_H, Nx_H = im1_down.shape
        Dx_H = Lx / Nx_H
        Dy_H = Ly / Ny_H

        # derivatives
        dxI2, dyI2 = image_first_derivatives(im2_down)
        dxI2 *= 1. / Dx_H  # convert to image units (Lx, Ly), instead of pixels
        dyI2 *= 1. / Dy_H

        im_mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), Nx_H, Ny_H)
        mesh = im_mesh

        if j != 0:
            if not silent:
                print("Warping im2")

            # warp
            coords, x_warp, y_warp = warp_coords(im_mesh, Nx_H, Ny_H, u, Lx, Ly, coord=True, silent=silent)
            im2_down = warp(im2_down, coords)
            dxI2 = warp(dxI2, coords)
            dyI2 = warp(dyI2, coords)

            # where the warping goes outside the image domain, cancel the data-attachment term
            out_ind = outside_indices(x_warp, y_warp, Nx_H, Ny_H)
            dxI2[out_ind] = 0.
            dyI2[out_ind] = 0.
            im2_down[out_ind] = im1_down[out_ind]  # so that It = 0

        I1 = array_to_scalar(im_mesh, im1_down)
        I2 = array_to_scalar(im_mesh, im2_down)
        gradI2 = arrays_to_vector(im_mesh, dxI2, dyI2)

        # prepare solver object
        V = VectorFunctionSpace(mesh, "Lagrange", 1)  # velocity space
        F = VectorFunctionSpace(mesh, "Lagrange", 1)

        # project previous u to this scale and use it as initialization
        if u is None:
            uw0 = Function(V)  # zero velocity
            f0 = Function(F)
        else:
            uw0 = project(u, V)
            f0 = project(f, V)

        # the regularization parameters have to be scaled too
        gamma_s = gamma * downscale_factor

        # launch solver
        if solver_type == 'of':
            from pde_solvers import ofsolver
            mesh, u, _, _ = ofsolver(mesh, I1, I2, gradI2, gamma_s, tol, uw0=uw0, noise_var=noise_var)
            f = f0
        else:
            mesh, u, f = flsolver(mesh, I1, I2, gradI2, f0, uw0, gamma_s, noise_var, tol=tol, BOOL_REG=BOOL_REG, eps_TV=eps_TV, mu=mu, lam=lam)

    u = project(u, V)
    return mesh, u, f, gamma_s


if __name__ == "__main__":
    import pickle
    import time
    from skimage.io import imread

    # load images
    out_folder = 'precovar_solutions'
    file_tfm_a = 'data/8-tfm_control_cel.tif'
    file_tfm_a_rest = 'data/8-tfm_control_lisado.tif'

    #parameters
    gamma = 0.0008979900408917639
    rel_noise_std = 0.11368207454039532
    noise_std = 7.4 # rel_noise_std*65.093815625
    tol = 1e-4
    finest = 3; coarsest=8
    plotting = False
    testing = None

    img = imread(file_tfm_a).astype(np.double)
    im1 = img[0, 2, 1, ...] + img[0, 1, 1, ...]
    del img

    img_rest = imread(file_tfm_a_rest).astype(np.double)
    im2 = img_rest[55, 1, 1, ...] + img_rest[55, 0, 1, ...]
    del img_rest

    new_size = im1.shape
    Ny, Nx = im1.shape

    # choose domain size in meters (has an influence on the regularisation parameters)
    Ly = 1e0
    Lx = Ly * Nx / Ny
    print("Lx", Lx, "Ly", Ly)

    # we need to allow extrapolation when going through the mesh set
    parameters["allow_extrapolation"] = True

    vertices = [Point(1.0, 0.0),
                Point(1.0, 1.0),
                Point(0.0, 1.0),
                Point(0.0, 0.0)]

    ###################### optimisation
    t = time.time()
    fac = 1. / np.mean(im1)
    im1 *= fac
    fac = 1. / np.mean(im2)
    im2 *= fac

    noise_var = (noise_std * fac) ** 2
    mesh, u, f, gamma_s = multiscale_algorithm(im1, im2, gamma, tol, noise_var, finest, coarsest, BOOL_REG=0)

    t2 = time.time()
    rank = MPI.comm_world.rank
    if rank == 0:
        print("Finished in %.3f s" % (t2 - t))

    ## Average pixel displacement
    imlen = im1.shape[0]/math.pow(2, finest)
    print("Average pixel displacement:", imlen*norm(u, 'L2'))
    fig_tmp = plt.figure(); pl = plot(u); true_u_max = pl.get_clim(); plt.close(fig_tmp)
    print("Max pixel displacement:", imlen*true_u_max[1])
    print("Max relative displacement:", 100.*true_u_max[1])

    ## Plotting results
    if plotting:
        pl = plot(f); plt.colorbar(pl); plt.show()

    ## Plotting error
    if testing is not None:
        F = VectorFunctionSpace(mesh, "Lagrange", 1)
        meshe = Mesh(testing[0])
        VorFe = VectorFunctionSpace(meshe, "P", 1)
        fe = Function(VorFe, testing[1])
        fe = project(fe, F)
        pl = plot(fe); plt.colorbar(pl); plt.show()
        pl = plot(sqrt(inner(f-fe,f-fe))); plt.colorbar(pl); plt.show()

    ## Saving
    base_name = os.path.splitext(os.path.basename(file_tfm_a))[0]
    save_dir = os.path.join(out_folder, base_name)
    
    rank = MPI.comm_world.rank
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    MPI.comm_world.barrier()
    
    File(os.path.join(save_dir, 'u.xml')) << u
    File(os.path.join(save_dir, 'f.xml')) << f
    File(os.path.join(save_dir, 'mesh.xml')) << mesh


    if False:
        ###################### RECOMPUTE IMAGES FOR COVARIANCE
        fact_lower = math.pow(2, finest)
        im1_lower = downsample_single(im1, fact_lower)
        im2_lower = downsample_single(im2, fact_lower)
        Ny_H, Nx_H = im1_lower.shape
        Dx_H = Lx / Nx_H
        Dy_H = Ly / Ny_H

        # derivatives
        dxI2, dyI2 = image_first_derivatives(im2_lower)
        dxI2 *= 1. / Dx_H  # convert to image units (Lx, Ly), instead of pixels
        dyI2 *= 1. / Dy_H

        # keep unwarped originals
        im_mesh = mesh
        gradI2 = arrays_to_vector(im_mesh, dxI2, dyI2)
        I2 = array_to_scalar(im_mesh, im2_lower)
        I1 = array_to_scalar(im_mesh, im1_lower)

        # save unwarped originals
        File(os.path.join(save_dir, 'I1.xml')) << I1
        File(os.path.join(save_dir, 'I2.xml')) << I2
        File(os.path.join(save_dir, 'gradI2.xml')) << gradI2

        # warp final variables at MAP
        coords, x_warp, y_warp = warp_coords(mesh, Nx_H, Ny_H, u, Lx, Ly, coord=True)
        im2_lower = warp(im2_lower, coords)
        dxI2 = warp(dxI2, coords)
        dyI2 = warp(dyI2, coords)

        out_ind = outside_indices(x_warp, y_warp, Nx_H, Ny_H)
        dxI2[out_ind] = 0.
        dyI2[out_ind] = 0.
        im2_lower[out_ind] = im1_lower[out_ind]

        warped_I2 = array_to_scalar(im_mesh, im2_lower)
        warped_gradI2 = arrays_to_vector(im_mesh, dxI2, dyI2)
        
        # save warped
        File(os.path.join(save_dir, 'I2_warped.xml')) << warped_I2
        File(os.path.join(save_dir, 'gradI2_warped.xml')) << warped_gradI2

        # save metadata
        if rank == 0:
            pick_list = [Nx_H, Ny_H, gamma_s, noise_var]
            with open(os.path.join(save_dir, 'data.pkl'), 'wb') as output:
                pickle.dump(pick_list, output)


        ###################### COVARIANCE (Hessian-based posterior)
        import hippylib as hp
        from pde_solvers import setup_problem, HessianOperator
        from posterior_using_hessian_trace import prior_distro, GaussianLRPosterior

        nrank = 10000  # number of eigenvalues to approximate

        # Setup the linearised problem at the MAP estimate
        uw0 = Constant((0., 0.))  # evaluate Hessian at the converged displacement

        (f_h, up, vq, Vf,
        bcs_state, bcs_adj,
        a_state, L_state, a_adj, L_adj,
        GM_varf, GR_varf, M,
        L_incr_adj, HM_varf, HR_varf) = setup_problem(
            mesh, warped_I2, I1, warped_gradI2, uw0, gamma_s, noise_var)

        # Assign the MAP force to the problem variable
        f_h.assign(f)

        # Solve state and adjoint at the MAP
        state_A, state_b = assemble_system(a_state, L_state, bcs_state)
        solve(state_A, up.vector(), state_b)

        adjoint_A, adjoint_RHS = assemble_system(a_adj, L_adj, bcs_adj)
        solve(adjoint_A, vq.vector(), adjoint_RHS)

        # Assemble Hessian sub-matrices
        C = assemble(HM_varf)
        W = assemble(L_incr_adj)
        R = assemble(HR_varf)

        # Preconditioner (regularization block)
        P = R
        Psolver = PETScKrylovSolver("cg", hp.amg_method())
        Psolver.set_operator(P)

        # Misfit Hessian operator (matrix-free)
        Hmisfit = HessianOperator(None, C, state_A, adjoint_A, W, bcs_adj, use_gaussnewton=False)

        # Low-rank eigendecomposition via randomised double-pass
        k = nrank
        p = 10
        Omega = hp.MultiVector(f_h.vector(), k + p)
        hp.parRandom.normal(1., Omega)
        lmbda, evecs = hp.doublePassG(Hmisfit, P, Psolver, Omega, k)

        if rank == 0:
            print(f"Top {k} eigenvalues of the Misfit Hessian:\n{lmbda[:20]}")

        # Build prior and low-rank posterior
        prior = prior_distro(Vf, P, Psolver, gamma_s, mean=None)
        nu = GaussianLRPosterior(prior, lmbda, evecs)
        nu.mean = f_h.vector()

        # Pointwise variance (exact)
        post_pw_variance, pr_pw_variance, corr_pw_variance = nu.pointwise_variance(method="Exact")

        post_vari = Function(Vf, post_pw_variance)
        prio_vari = Function(Vf, pr_pw_variance)
        corr_vari = Function(Vf, corr_pw_variance)

        File(os.path.join(save_dir, 'post_vari.xml')) << post_vari
        File(os.path.join(save_dir, 'prio_vari.xml')) << prio_vari
        File(os.path.join(save_dir, 'corr_vari.xml')) << corr_vari

        if rank == 0:
            print("Covariance fields saved.")
            with open(os.path.join(save_dir, 'eigenvalues.pkl'), 'wb') as output:
                pickle.dump(lmbda, output)
