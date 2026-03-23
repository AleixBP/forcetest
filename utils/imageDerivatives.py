import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig

# http://en.wikipedia.org/wiki/Finite_difference_coefficients

#first_derivative_kernel_1D = np.array([[0.5, 0., -0.5]])
# five point stencil is more precise
first_derivative_kernel_1D = np.array([[-1., 8., 0., -8., 1.]])/12. # FIXME
#second_derivative_kernel_1D = np.array([[1., -2., 1.]])
second_derivative_kernel_1D = np.array([[-1., 16., -30., 16., -1.]])/12.

mixed_derivative_kernel_2D = -0.25*np.array([[-1., 0.,  1.],
                                             [ 0., 0.,  0.],
                                             [ 1., 0., -1.]])

# works only with a length-3 kernel !
#def convolve_replicate(im, kernel):
#    #scipy.signal.convolve uses zero-padding to handle boundaries
#    # we want "replicate" boundary conditions, i.e. out-of-bounds values are
#    # replicated from the closest valid pixel
#
#    conv = sig.convolve(im, kernel, 'same')
#    a = kernel.size
#    if a == 3:
#        conv[:,0] += kernel[0, -1]*im[:,0] # "replicate" boundary conditions
#        conv[:,-1] += kernel[0, 0]*im[:,-1] # "replicate" boundary conditions
#    elif a == 5:
#        print "not supported", kernel
#    else:
#        print "not supported", kernel
#    return conv

# works only with an odd-length kernel !
def convolve_replicate(im, kernel):
    #scipy.signal.convolve uses zero-padding to handle boundaries
    # we want "replicate" boundary conditions, i.e. out-of-bounds values are
    # replicated from the closest valid pixel

    if kernel.size % 2 == 0:
        print("Error, the kernel should be of odd size")

    conv = sig.convolve(im, kernel, 'same')
    a = kernel.size
    b = int((a - 1)/2)
    for i in range(0, b):
        conv[:, i] += kernel[0, -(b-i):].sum()*im[:,0] # "replicate" boundary conditions
        conv[:, -(1+i)] += kernel[0, :b-i].sum()*im[:,-1] # "replicate" boundary conditions

    return conv

def convolve_noreplicate(im, kernel):
    #scipy.signal.convolve uses zero-padding to handle boundaries
    conv = sig.convolve(im, kernel, 'same')
    return conv

#def image_first_derivatives(im1, im2):
#    # compute derivatives
#
#    # spatial derivatives are centered in time
#    E = 0.5*(im1 + im2)
#
#    Ex = convolve_replicate(E, first_derivative_kernel_1D)
#
#    Ey = convolve_replicate(E.T, first_derivative_kernel_1D)
#    Ey = Ey.T
#
#    Et = im2 - im1
#
#    if False:
#        test_image_derivatives(Ex, Ey, Et)
#
#    return Ex, Ey, Et

# apply the kernel to the image matrix so that each element is substituted by its derivative
# by transposing the matrix we can apply the same kernel to get the y derivatives
def image_first_derivatives(im):
    #imx = convolve_noreplicate(im, first_derivative_kernel_1D)
    #imy = convolve_noreplicate(im.T, first_derivative_kernel_1D)
    imx = convolve_replicate(im, first_derivative_kernel_1D)
    imy = convolve_replicate(im.T, first_derivative_kernel_1D)
    imy = imy.T

    if False:
        test_image_derivatives(Ex, Ey, Et)

    return imx, imy

def get_image_derivatives(im):
    # compute derivatives
    E = im

    Ex = convolve_replicate(E, first_derivative_kernel_1D)

    Ey = convolve_replicate(E.T, first_derivative_kernel_1D)
    Ey = Ey.T

    Exx = convolve_replicate(E, second_derivative_kernel_1D)

    Eyy = convolve_replicate(E.T, second_derivative_kernel_1D)
    Eyy = Eyy.T

    # warning ! mixed derivatives is different from successive first derivative
    #Exy = convolve_replicate(Ey, first_derivative_kernel_1D)
    Exy = sig.convolve(E, mixed_derivative_kernel_2D, 'same') # FIXME boundaries

    # test that derivative in x and y can be exchanged
    #Eyx = convolve_replicate(Ex.T, first_derivative_kernel_1D).T
    #print "test", np.all(np.abs(Exy - Eyx) < 1e-10), np.where(np.abs(Exy - Eyx) > 1e-10)

    if False:
        test_second_image_derivatives(Ex, Ey, Exx, Exy, Eyy)

    return Ex, Ey, Exx, Eyy, Exy

def get_second_image_derivatives(im1, im2, Ex, Ey, Et):
    #compute second-order derivatives
    #NOTE: positions of the derivative result is important
    #that's why we have full second-order when deriving on the same axis,
    #and cascade of two first-order when deriving on two different axes

    E = 0.5*(im1 + im2)

    Exx = convolve_replicate(E, second_derivative_kernel_1D)

    Eyy = convolve_replicate(E.T, second_derivative_kernel_1D)
    Eyy = Eyy.T

    # warning ! mixed derivatives is different from successive first derivative
    #Exy = convolve_replicate(Ey, first_derivative_kernel_1D)
    Exy = sig.convolve(E, mixed_derivative_kernel_2D, 'same') # FIXME boundaries

    Etx = convolve_replicate(Et, first_derivative_kernel_1D)

    Ety = convolve_replicate(Et.T, first_derivative_kernel_1D)
    Ety = Ety.T

    if False:
        test_second_image_derivatives(Ex, Ey, Exx, Exy, Eyy)

    return Exx, Eyy, Exy, Etx, Ety

def get_first_flow_derivatives(u, v):
    #compute second-order derivatives
    #NOTE: positions of the derivative result is important
    #that's why we have full second-order when deriving on the same axis,
    #and cascade of two first-order when deriving on two different axes

    ux = convolve_replicate(u, first_derivative_kernel_1D)

    uy = convolve_replicate(u.T, first_derivative_kernel_1D)
    uy = uy.T

    vx = convolve_replicate(v, first_derivative_kernel_1D)

    vy = convolve_replicate(v.T, first_derivative_kernel_1D)
    vy = vy.T

    return ux, uy, vx, vy

def get_second_flow_derivatives(u, v):
    #compute second-order derivatives
    #NOTE: positions of the derivative result is important
    #that's why we have full second-order when deriving on the same axis,
    #and cascade of two first-order when deriving on two different axes

    uxx = convolve_replicate(u, second_derivative_kernel_1D)

    uyy = convolve_replicate(u.T, second_derivative_kernel_1D)
    uyy = uyy.T

    vxx = convolve_replicate(v, second_derivative_kernel_1D)

    vyy = convolve_replicate(v.T, second_derivative_kernel_1D)
    vyy = vyy.T

    return uxx, uyy, vxx, vyy

def test_image_derivatives(Ex, Ey, Et):
    plt.figure()

    plt.subplot(1,3,1)
    im = plt.imshow(Ex, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Ex")

    plt.subplot(1,3,2)
    im = plt.imshow(Ey, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Ey")

    plt.subplot(1,3,3)
    im = plt.imshow(Et, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Et")

    plt.show()

    import sys
    sys.exit(0)

def test_second_image_derivatives(Ex, Ey, Exx, Exy, Eyy):
    plt.figure()

    plt.subplot(2,3,1)
    im = plt.imshow(Ex, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Ex")

    plt.subplot(2,3,2)
    im = plt.imshow(Ey, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Ey")

    plt.subplot(2,3,4)
    im = plt.imshow(Exx, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Exx")

    plt.subplot(2,3,5)
    im = plt.imshow(Exy, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Exy")

    plt.subplot(2,3,6)
    im = plt.imshow(Eyy, cmap='bwr', origin='lower', interpolation='nearest')
    c = max(abs(np.array(im.get_clim())))
    im.set_clim(-c, c)
    plt.colorbar()
    plt.title("Eyy")

    plt.show()

    import sys
    sys.exit(0)
