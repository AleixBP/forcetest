import os
nthreads = "2"
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["OPENMP_NUM_THREADS"] = nthreads
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import sys
import math
import numpy as np
import pickle
import time
from dolfin import *
from skimage.io import imread
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from image_generator_benchmark import generate_benchmark_images

from multiscale_algorithms import multiscale_algorithm
from utils.dolfin_numpy_utils import array_to_scalar, vector_to_arrays

def run_pipeline(noise_std_rel=0.1, ranseed=2, A=10.0, gammas=[0.01], finest=1, blackout=None, lam=1.9762, nrank=10, out_folder = 'reviewer_benchmark_results'):
    # -------------------------------------------------------------
    # 1. PARAMETERS FOR THE PIPELINE
    # -------------------------------------------------------------
    file_tfm_a_rest = 'data/8-tfm_control_lisado.tif'
    
    # Generation params
    from synthetic_forces import a_f_gaussian_dipole, a_f_gaussian_tripole
    force_func = lambda Lx, Ly, degree: a_f_gaussian_tripole(Lx, Ly, degree, A=A, width=30.0)
    
    # Reconstruction params
    tol = 1e-4  
    coarsest = 5
    np.random.seed(ranseed)
    plotting = True
    mu = 1.0
    

    # -------------------------------------------------------------
    # 2. GENERATE THE GROUND TRUTH "UNIVERSE" 
    # -------------------------------------------------------------
    print("=== STEP 1: GENERATING GROUND TRUTH DATA ===")
    img_load = imread(file_tfm_a_rest).astype(np.double)
    img_rest_raw = img_load[55, 1, 1, ...] + img_load[55, 0, 1, ...]
    
    cxy = 320
    im_rest_cropped = img_rest_raw[cxy:-cxy, cxy:-cxy]
    
    im_rest_zoomed = im_rest_cropped
    
    # Synthesize ground truth images
    im1_array, im2_array, true_u, true_f, eval_mesh = generate_benchmark_images(
        base_image_array=im_rest_zoomed,
        force_func=force_func,
        noise_std_rel=noise_std_rel,
        save_to_disk=False, 
        io_method='skimage',
        warp_mode='nearest', # Safe boundary wrap
        formulation='mixed',
        mu=mu,
        lam=lam
    )
    
    # --- ADD DEAD ZONE FOR APERTURE PROBLEM ---
    if blackout is not None:
        shif = blackout
        offsx = 10
        offsy = 0
        cx, cy = im1_array.shape[0]//2, im1_array.shape[1]//2
        im1_array[cx-shif+offsy:cx+shif+offsy, cy-shif+offsx:cy+shif+offsx] = np.mean(im1_array)
        im2_array[cx-shif+offsy:cx+shif+offsy, cy-shif+offsx:cy+shif+offsx] = np.mean(im2_array)
    # ------------------------------------------

    print(f"Generated Synthetic Images -> Size: {im1_array.shape}")
    plt.imshow(im1_array); plt.show()
    plt.imshow(im2_array); plt.show()

    # -------------------------------------------------------------
    # 3. RECONSTRUCT USING THE INVERSE SOLVER
    # -------------------------------------------------------------
    print("\n=== STEP 2: RECONSTRUCTING WITH TFM ALGORITHM ===")
    
    rank = MPI.comm_world.rank
    save_dir = os.path.join(out_folder, f"noise_{noise_std_rel}_scale_{finest}")
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    MPI.comm_world.barrier()
    
    # Normalize the images (mean=1)
    fac = 1. / np.mean(im1_array)
    im1_array *= fac
    fac = 1. / np.mean(im2_array)
    im2_array *= fac
    
    # Adjust algorithm variance to match the normalized brightness
    effective_noise_var = (noise_std_rel*np.mean(im1_array))**2 #noise_var * (fac**2)

    for gamma in gammas:    
        t_start = time.time()

        mesh_rec, u_rec, f_rec, gamma_s = multiscale_algorithm(
            im1_array,
            im2_array,
            gamma=gamma, 
            tol=tol, 
            noise_var=effective_noise_var, 
            finest=finest, 
            coarsest=coarsest,
            silent=False,
            BOOL_REG=0,
            mu=mu,
            lam=lam,
            #BOOL_REG=1,
            #eps_TV=1000.
        )
        t_end = time.time()
        
        if rank == 0:
            print(f"Algorithm convergence complete in {t_end - t_start:.2f} seconds.")

        print("gamma =", gamma)
        
        if plotting:
            fig_tmp = plt.figure(); pl = plot(true_f); clim1 = pl.get_clim(); plt.close(fig_tmp)
            fig_tmp = plt.figure(); pl = plot(f_rec); clim2 = pl.get_clim(); plt.close(fig_tmp)

        # -------------------------------------------------------------
        # 4.a EIGENVALUES
        # -------------------------------------------------------------
        print("\n=== EIGENVALUES ===")
        
        from pde_solvers import setup_problem, HessianOperator
        import hippylib as hp
        from utils.warp import warp, outside_indices, warp_coords
        from utils.pyramid import downsample_single
        from utils.imageDerivatives import image_first_derivatives
        from utils.dolfin_numpy_utils import arrays_to_vector
        
    
        # 1. Recompute the warped images directly from memory at the MAP estimate (u_rec)
        Ny, Nx = im1_array.shape
        Ly = 1.0
        Lx = Ly * Nx / Ny
        
        fact_lower = math.pow(2, finest)
        im1_lower = downsample_single(im1_array, fact_lower)
        im2_lower = downsample_single(im2_array, fact_lower)
        Ny_H, Nx_H = im1_lower.shape
        Dx_H = Lx / Nx_H
        Dy_H = Ly / Ny_H

        dxI2, dyI2 = image_first_derivatives(im2_lower)
        dxI2 *= 1. / Dx_H
        dyI2 *= 1. / Dy_H

        # Warp using u_rec directly
        coords, x_warp, y_warp = warp_coords(mesh_rec, Nx_H, Ny_H, u_rec, Lx, Ly, coord=True)
        im2_lower_warped = warp(im2_lower, coords)
        dxI2_warped = warp(dxI2, coords)
        dyI2_warped = warp(dyI2, coords)

        out_ind = outside_indices(x_warp, y_warp, Nx_H, Ny_H)
        dxI2_warped[out_ind] = 0.
        dyI2_warped[out_ind] = 0.
        im2_lower_warped[out_ind] = im1_lower[out_ind]

        # Convert to FEniCS functions dynamically
        I1_MAP = array_to_scalar(mesh_rec, im1_lower)
        I2_MAP_warped = array_to_scalar(mesh_rec, im2_lower_warped)
        gradI2_MAP_warped = arrays_to_vector(mesh_rec, dxI2_warped, dyI2_warped)
        
        uw0 = Constant((0., 0.))
        
        # 2. Setup forms
        (f_h, up, vq, Vf,
            bcs_state, bcs_adj,
            a_state, L_state, a_adj, L_adj,
            GM_varf, GR_varf, M,
            L_incr_adj, HM_varf, HR_varf) = setup_problem(
                mesh_rec, I1_MAP, I2_MAP_warped, gradI2_MAP_warped, uw0, gamma_s, effective_noise_var)
        
        f_h.assign(f_rec)

        # 3. Assemble Forward and Adjoint (Pre-Hessian computation)
        state_A, state_b = assemble_system(a_state, L_state, bcs_state)
        solve(state_A, up.vector(), state_b)
        
        adjoint_A, adjoint_RHS = assemble_system(a_adj, L_adj, bcs_adj)
        solve(adjoint_A, vq.vector(), adjoint_RHS)
        
        # 4. Assemble the Hessian matrices
        C = assemble(HM_varf)
        W = assemble(L_incr_adj)
        R = assemble(HR_varf)

        # Preconditioner
        P = R
        Psolver = PETScKrylovSolver("cg", hp.amg_method())
        Psolver.set_operator(P)
        
        # Compute the Hessian Operator (Covariance inverse) operator evaluated at the MAP 
        Hmisfit = HessianOperator(None, C, state_A, adjoint_A, W, bcs_adj, use_gaussnewton=False)
        
        print(f"Hessian matrices successfully assembled at MAP estimate.")
        print(f"Degrees of freedom for Covariance: {Vf.dim()}")
        
        # 5. Extract Eigenvalues of the Covariance/Hessian
        k = nrank
        p = 10
        Omega = hp.MultiVector(f_h.vector(), k + p)
        hp.parRandom.normal(1., Omega)
        lmbda, evecs = hp.doublePassG(Hmisfit, P, Psolver, Omega, k)

        #plt.plot(range(0, k), lmbda); plt.show()
        
        
        print(f"Top {k} eigenvalues of the Misfit Hessian:\n {lmbda}")

        # -------------------------------------------------------------
        # 4.b POST-RECONSTRUCTION COVARIANCE (HESSIAN) / UNCERTAINTY
        # -------------------------------------------------------------
        print("\n=== COVARIANCE - HESSIAN CALCULATION ===")
        from posterior_using_hessian_trace import prior_distro, GaussianLRPosterior
        mesh = mesh_rec

        prior = prior_distro(Vf, P, Psolver, gamma_s, mean=None) #prior mean is automatically zero
        nu = GaussianLRPosterior(prior, lmbda, evecs)
        nu.mean = f_h.vector()#.get_local()

        # Compute covariance
        post_pw_variance1, pr_pw_variance1, corr_pw_variance1 = nu.pointwise_variance(method="Randomized")

        Vf = VectorFunctionSpace(mesh, 'Lagrange', 1) #repeat
        post_vari_rand = Function(Vf, post_pw_variance1)
        prio_vari_rand = Function(Vf, pr_pw_variance1)
        corr_vari_rand = Function(Vf, corr_pw_variance1)
        
        File(os.path.join(out_folder, "post_vari_rand.xml")) << post_vari_rand
        File(os.path.join(out_folder, "prio_vari_rand.xml")) << prio_vari_rand
        File(os.path.join(out_folder, "corr_vari_rand.xml")) << corr_vari_rand
        File(os.path.join(out_folder, "mesh_vari_rand.xml")) << mesh
        
        # Compute covariance
        post_pw_variance, pr_pw_variance, corr_pw_variance = nu.pointwise_variance(method="Exact")

        Vf = VectorFunctionSpace(mesh, 'Lagrange', 1) #repeat
        post_vari_exact = Function(Vf, post_pw_variance)
        prio_vari_exact = Function(Vf, pr_pw_variance)
        corr_vari_exact = Function(Vf, corr_pw_variance)
        
        File(os.path.join(out_folder, "post_vari_exact.xml")) << post_vari_exact
        File(os.path.join(out_folder, "prio_vari_exact.xml")) << prio_vari_exact
        File(os.path.join(out_folder, "corr_vari_exact.xml")) << corr_vari_exact
        File(os.path.join(out_folder, "mesh_vari_exact.xml")) << mesh

    return lmbda, evecs

if __name__ == "__main__":
    _ = run_pipeline(noise_std_rel=0.15, ranseed=20, A=2., gammas=[.05], blackout=None, lam=2., finest=2, nrank=1000, out_folder="covariance_tripole")
