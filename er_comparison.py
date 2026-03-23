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

def run_pipeline(noise_std_rel=0.1, ranseed=2, A=10.0, gammas=[0.01], finest=1, blackout=None, lam=1.9762):
    # -------------------------------------------------------------
    # 1. PARAMETERS FOR THE PIPELINE
    # -------------------------------------------------------------
    file_tfm_a_rest = 'data/8-tfm_control_lisado.tif'
    out_folder = 'reviewer_benchmark_results'
    
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
    
    # Downscale for performance
    im_rest_zoomed = im_rest_cropped
    
    # Synthesize ground truth images
    im1_array, im2_array, true_u, true_f, eval_mesh = generate_benchmark_images(
        base_image_array=im_rest_zoomed,
        force_func=force_func,
        noise_std_rel=noise_std_rel,
        save_to_disk=False, 
        io_method='skimage',
        warp_mode='nearest',
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
    
    # Create the reconstruction output directory safely
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
        )
        t_end = time.time()
        
        if rank == 0:
            print(f"Algorithm convergence complete in {t_end - t_start:.2f} seconds.")

        print("gamma =", gamma)
        
        if plotting:
            fig_tmp = plt.figure(); pl = plot(true_f); clim1 = pl.get_clim(); plt.close(fig_tmp)
            fig_tmp = plt.figure(); pl = plot(f_rec); clim2 = pl.get_clim(); plt.close(fig_tmp)

    return true_f, f_rec, true_u, u_rec, mesh_rec

if __name__ == "__main__":
    #Run plotting
    _ = run_pipeline(noise_std_rel=0.15, ranseed=20, A=10., gammas=[.5], blackout=None, lam=2.)
    