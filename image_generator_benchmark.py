import os
nthreads = "2"
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["OPENMP_NUM_THREADS"] = nthreads
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
import numpy as np
from dolfin import *

from utils.warp import warp, warp_coords

def b_d(Lx, Ly, degree):
    class G(UserExpression):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def value_shape(self):
            return (2,)
        def eval(self, value, x):
            value[0]=0.
            value[1]=0.

    g=G(degree=degree)
    return g

def el_solver(mesh, f, g, mu=1.0, lam=1.9762, tol=1e-4):
    from ufl import nabla_div

    displacement_element=VectorElement("P", mesh.ufl_cell(), 1)
    force_element=VectorElement("P", mesh.ufl_cell(), 1)
    displacement_space=FunctionSpace(mesh, displacement_element)
    force_space=FunctionSpace(mesh, force_element)

    v=TestFunction(displacement_space)
    u=TrialFunction(displacement_space)
    s=Function(displacement_space)

    d = u.geometric_dimension()

    def boundary(x, on_boundary):
            return on_boundary
    bcsWfor0=[DirichletBC(displacement_space, g, boundary)]
    bcs=bcsWfor0

    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(u):
        return lam*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx # + dot(T, v)*ds

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b, 'mumps')

    return s

def mixed_el_solver(mesh, f, g, mu=1.0, lam=1.9762, degree=1):
    """
    Ground truth forward solver using the mixed displacement-pressure (u-p) 
    formulation, specifically mimicking the math in pde_solvers.py.
    """
    VE = VectorElement("P", mesh.ufl_cell(), degree)
    PE = FiniteElement("P", mesh.ufl_cell(), degree)
    W = FunctionSpace(mesh, MixedElement([VE, PE]))
    
    up = TrialFunction(W)
    vq = TestFunction(W)
    u, p = split(up)
    v, q = split(vq)
    
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)
        
    def sigma(u, p):
        d = u.geometric_dimension()
        return -p * Identity(d) + 2 * mu * epsilon(u)
        
    def boundary(x, on_boundary):
        return on_boundary
        
    bcs = [DirichletBC(W.sub(0), g, boundary)]
    
    a_state = inner(sigma(u, p), epsilon(v)) * dx \
            - q * div(u) * dx \
            - (p * q / lam) * dx
    L_state = inner(f, v) * dx
    
    sol = Function(W)
    solve(a_state == L_state, sol, bcs, solver_parameters={'linear_solver': 'mumps'})
    
    u_sol, p_sol = sol.split(deepcopy=True)
    return u_sol

def gauss_noise(im, sigma):
    """Adds Gaussian noise to an image array."""
    if sigma <= 0:
        return im
    
    row, col = im.shape
    mean = 0.
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_im = im + gauss
    return noisy_im

def generate_benchmark_images(
    base_image_path=None,
    base_image_array=None,
    force_func=None,
    noise_std_rel=0.1,
    Nx_mesh=None,
    Ny_mesh=None,
    formulation='pure',
    warp_mode='nearest',
    save_to_disk=False,
    io_method='pil', # Options: 'pil' or 'skimage'
    output_prefix="bench",
    mu=1.0, lam=1.9762,
):
    """
    Generates ground truth synthetic images for reviewer benchmarking.
    Decouples image resolution (for textures) from FEM mesh resolution (for PDE discretization).
    
    Args:
        base_image_path: Path to the experimental base image (string).
        base_image_array: Directly pass a numpy array (skips reading from disk).
        force_func: Function to generate applied forces (`a_f` from apf_*.py files).
        noise_std: Standard deviation for the Gaussian noise applied to both images.
        Nx_mesh/Ny_mesh: FEM mesh size. If None, defaults to image resolution.
        warp_mode: Interpolation mode for the warp function ('nearest', 'bilinear', etc.).
        save_to_disk: Whether to output images and VTK/PVD files to disk.
        io_method: Determines library used for reading and saving images ('pil' or 'skimage').
    """
    # 1. Load and prep the base experimental image
    if base_image_array is not None:
        image1 = base_image_array.astype(np.double)
        
        if np.max(image1) > 1.0:
            image1 = image1 / np.max(image1)
    elif base_image_path is not None:
        if io_method == 'pil':
            image_pil = Image.open(base_image_path)
            image_pil = ImageEnhance.Contrast(image_pil).enhance(1.5)
            image1 = np.array(image_pil).astype(np.double) / 255.
        elif io_method == 'skimage':
            import skimage.io as io
            from skimage import exposure
            
            image1_raw = io.imread(base_image_path, as_gray=True)
            
            if image1_raw.dtype == np.uint8:
                image1_raw = image1_raw.astype(np.double) / 255.
            elif image1_raw.max() > 1.0:
                image1_raw = image1_raw.astype(np.double) / image1_raw.max()
            else:
                image1_raw = image1_raw.astype(np.double)
                
            # Optional equivalent contrast enhancement using skimage
            image1 = exposure.adjust_gamma(image1_raw, gamma=0.8) # rough equivalent to enhance 1.5 in PIL
        else:
            raise ValueError("Invalid io_method. Choose 'pil' or 'skimage'")
    else:
        raise ValueError("Must provide either base_image_path or base_image_array.")
    
    Ny_img, Nx_img = image1.shape
    Lx, Ly = 1., 1.
    
    # 2. Setup FEM Simulation Mesh (Decoupled from image resolution!)
    Nx = Nx_mesh if Nx_mesh is not None else Nx_img
    Ny = Ny_mesh if Ny_mesh is not None else Ny_img
    mesh = RectangleMesh(Point(0., 0.), Point(Lx, Ly), Nx, Ny)
    
    degree = 1
    displacement_element = VectorElement("P", mesh.ufl_cell(), degree)
    force_element = VectorElement("P", mesh.ufl_cell(), degree)
    
    displacement_space = FunctionSpace(mesh, displacement_element)
    force_space = FunctionSpace(mesh, force_element)

    # Use provided force_func or default to synthetic_forces
    if force_func is None:
        from synthetic_forces import a_f
        f_expr = a_f(Lx, Ly, degree)
    else:
        f_expr = force_func(Lx, Ly, degree)
        
    g_expr = b_d(Lx, Ly, degree)
    
    f = interpolate(f_expr, force_space)
    g = interpolate(g_expr, displacement_space)
    
    # 3. Solve Forward Elasticity
    if formulation == 'mixed':
        u = mixed_el_solver(mesh, f, g, mu=mu, lam=lam, degree=degree)
    else:
        u = el_solver(mesh, f, g, mu=mu, lam=lam)
    
    # Keep a copy of the true displacement as FEniCS Function
    vel = Function(displacement_space)
    vel.assign(u)
    
    # 4. Warp the Image (Warping must happen exactly at Image Resolution)
    image_mesh = RectangleMesh(Point(0., 0.), Point(Lx, Ly), Nx_img, Ny_img)
    # Invert displacement for pull-back warping algorithm
    u.vector()[:] = -u.vector()
    coords, _, _ = warp_coords(image_mesh, Nx_img, Ny_img, u, Lx, Ly, coord=True)

    if False:
        dense_space = VectorFunctionSpace(image_mesh, "P", degree)
        u_dense = interpolate(u, dense_space)
        u_dense.vector()[:] = -u_dense.vector()
        coords, _, _ = warp_coords(image_mesh, Nx_img, Ny_img, u_dense, Lx, Ly, coord=True)
    
    image2 = warp(image1, coords, mode=warp_mode)
    
    # 5. Add Noise to BOTH images equally
    noise_std = noise_std_rel * np.mean(image1)
    image1_noisy = gauss_noise(image1, noise_std)
    image2_noisy = gauss_noise(image2, noise_std)
    
    # 6. Optional disk saving
    if save_to_disk:
        out_folder = os.path.dirname(base_image_path) if base_image_path else "simulated_images/solutions"
        sol_folder = "simulated_images/solutions"
        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(sol_folder, exist_ok=True)
        
        path_im1 = os.path.join(out_folder, f"{output_prefix}_im1.tiff")
        path_im2 = os.path.join(out_folder, f"{output_prefix}_im2.tiff")
        
        path_npy1 = os.path.join(out_folder, f"{output_prefix}_im1.npy")
        path_npy2 = os.path.join(out_folder, f"{output_prefix}_im2.npy")
        
        # Save exact arrays to NPY
        np.save(path_npy1, image1_noisy)
        np.save(path_npy2, image2_noisy)
        
        # Save Tiff
        if io_method == 'pil':
            from PIL import Image, ImageEnhance
            
            Image.fromarray((image1_noisy * 255).clip(0, 255).astype(np.double)).save(path_im1) #np.uint8
            Image.fromarray((image2_noisy * 255).clip(0, 255).astype(np.double)).save(path_im2)
        elif io_method == 'skimage':
            import skimage.io as io
            import warnings
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # suppress low-contrast warnings for tiny floats
                io.imsave(path_im1, image1_noisy.astype(np.float32))
                io.imsave(path_im2, image2_noisy.astype(np.float32))
        
        # Save PDE state
        file_u = File(os.path.join(sol_folder, f"{output_prefix}_u.pvd"))
        file_u << vel
        file_f = File(os.path.join(sol_folder, f"{output_prefix}_f.pvd"))
        file_f << f
        
    return image1_noisy, image2_noisy, vel, f, mesh

if __name__ == "__main__":
    from synthetic_forces import a_f, a_f_gaussian_dipole, a_f_gaussian_dipole_cplus

    force_func = lambda Lx, Ly, degree: a_f_gaussian_dipole(Lx, Ly, degree, A=10.0, width=50.0)
        
    file_tfm_a_rest = 'data/8-tfm_control_lisado.tif'

    from skimage.io import imread
    img_rest = imread(file_tfm_a_rest).astype(np.double)
    im = img_rest[55, 1, 1, ...] + img_rest[55, 0, 1, ...]
    
    if False:
        from scipy.ndimage import zoom
        im = zoom(im, 0.5, order=3)
    else:
        cxy = 310
        im = im[cxy:-cxy,cxy:-cxy]


    im1, im2, true_u, true_f, FEM_mesh = generate_benchmark_images(
                    base_image_array=im,
                    noise_std_rel = 0.5,
                    force_func=force_func,
                    save_to_disk=False, 
                    io_method='skimage',
                )
    print("Benchmark images and FEM data generated successfully!")