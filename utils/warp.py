from scipy.ndimage.interpolation import map_coordinates
import numpy as np
from .dolfin_numpy_utils import vector_to_arrays

def warp_coords_euler(mesh, Nx, Ny, u, Lx, Ly, coord=False, silent=False):
    # this version can handle a large displacement u, and properly integrate the cinematic equations to find x, y

    if not silent:
        print("Warping: Images shape: Nx = %d, Ny = %d" %(Nx, Ny))

    u1_pix, u2_pix = vector_to_arrays(mesh, Nx, Ny, u)

    u1_pix *= float(Nx)/Lx # convert to pixels units, instead of image units (Lx, Ly)
    u2_pix *= float(Ny)/Ly

    # warp the image I1 to compensate for displacements found at lower scales (that can be several pixels large)

    # First, Construct a 2-D grid
    x = np.arange(0, float(Nx))
    y = np.arange(0, float(Ny))
    xx, yy = np.meshgrid(x, y)
    # Second, define the warped coordinates
    N = 1000
    x_warp = np.array(xx) # make a copy to prevent errors
    y_warp = np.array(yy)
    for i in range(N):
        spline_order = 3 # 2
        coordinates = np.vstack((y_warp.flatten(), x_warp.flatten()))
        u1_loc = map_coordinates(u1_pix, coordinates, order=spline_order, mode='constant', cval=np.NaN).reshape(u1_pix.shape)
        u2_loc = map_coordinates(u2_pix, coordinates, order=spline_order, mode='constant', cval=np.NaN).reshape(u2_pix.shape)
        x_warp += u1_loc/float(N)
        y_warp += u2_loc/float(N)

    coordinates = np.vstack((y_warp.flatten(), x_warp.flatten()))

    # Third, interpolate to get the values at the warped coordinates
    #Iwarp = warp_coords(I, x_warp, y_warp)

    if coord:
        return coordinates, x_warp, y_warp
    else:
        return coordinates

def warp_coords(mesh, Nx, Ny, u, Lx, Ly, coord=False, silent=False):
    # this version can handle a large displacement u

    if not silent:
        print("Warping: Images shape: Nx = %d, Ny = %d" %(Nx, Ny))

    u1_pix, u2_pix = vector_to_arrays(mesh, Nx, Ny, u)

    u1_pix *= float(Nx)/Lx # convert to pixels units, instead of image units (Lx, Ly)
    u2_pix *= float(Ny)/Ly

    # warp the image I1 to compensate for displacements found at lower scales (that can be several pixels large)
    # First, Construct a 2-D grid
    x = np.arange(0, Nx)
    y = np.arange(0, Ny)
    xx, yy = np.meshgrid(x, y)
    # Second, define the warped coordinates
    x_warp = xx + u1_pix
    y_warp = yy + u2_pix

    coordinates = np.vstack((y_warp.flatten(), x_warp.flatten()))

    # Third, interpolate to get the values at the warped coordinates
    #Iwarp = warp_coords(I, x_warp, y_warp)

    if coord:
        return coordinates, x_warp, y_warp
    else:
        return coordinates

def warp(I, coordinates, mode='constant', cval=np.NaN):
    # interpolate to get the values at the warped coordinates

    # Note: scipy.interpolate.interp2d and scipy.interpolate.SmoothBivariateSpline
    #       seem to require regular and/or rectangular grids

    # 'map_coordinates' does spline interpolation
    # the 'order' parameter rules the spline order
    # the 'mode' parameter rules the boundary behaviour
    spline_order = 3 # 2
    #Iwarp = map_coordinates(I, coordinates, order=spline_order, mode='nearest').reshape(I.shape)
    Iwarp = map_coordinates(I, coordinates, order=spline_order, mode=mode, cval=cval).reshape(I.shape)
    #Iwarp = map_coordinates(I, coordinates, order=spline_order, mode='reflect').reshape(I.shape)

    return Iwarp

def outside_indices(x_warp, y_warp, Nx, Ny):
    in_pos = (x_warp>=0.)*(x_warp<=Nx-1.)*(y_warp>=0.)*(y_warp<=Ny-1.)
    out_pos = 1 - in_pos
    return np.where(out_pos)
