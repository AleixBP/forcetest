from scipy.signal import resample
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from scipy.ndimage import affine_transform
from PIL import Image

def downsampled_pyramid(im, downsampling_factor, Nmin):
    pyramid = [im]

    # make sure the downsampling factor is a float
    downsampling_factor = float(downsampling_factor)
    # idem for the minimum size
    Nmin = float(Nmin)

    N = min(im.shape)
    n = np.log(Nmin/N)/np.log(downsampling_factor)
    n = int(np.floor(n)) + 1

    for i in range(1, n):
        new_w = int(np.ceil(im.shape[1]*downsampling_factor**i))
        new_h = int(np.ceil(im.shape[0]*downsampling_factor**i))

        method = 3
        if method == 0:
            # resample uses a sinc filter, but it assumes that the data is periodic, which is not correct
            # in our case
            im_down = resample(im,      new_h, axis=0)
            im_down = resample(im_down, new_w, axis=1)
        elif method == 1:
            # simple gaussian filtering, no downscaling
            sigma = np.sqrt(2.)/4.*(1./downsampling_factor)**i
            im_down = gaussian_filter(im, sigma)
        elif method == 2:
            # gaussian filtering followed by downscaling
            sigma = np.sqrt(2.)/4.*(1./downsampling_factor)**i
            im_down = gaussian_filter(im, sigma)
            im_down = affine_transform(im_down, [(im_down.shape[0]-1)*1.0/new_h, (im_down.shape[1]-1)*1.0/new_w], output_shape=[new_h,new_w], mode='nearest')
        elif method == 3:
            # PIL resizing
            im_down = Image.fromarray(im).resize((new_w,new_h), Image.Resampling.LANCZOS)#Image.ANTIALIAS)
            im_down = np.array([im_down.getdata()], np.double).reshape((new_h,new_w))

        # Note :fractional resampling is best done as integer upsampling followed by integer downsampling

        pyramid += [im_down]

#    for i in range(1, n):
#        new_w = pyramid[-1].shape[1]/2
#        new_h = pyramid[-1].shape[0]/2
#
#        # PIL resizing
#        im_down = np.array([Image.fromarray(pyramid[-1]).resize((new_w,new_h), Image.ANTIALIAS).getdata()], np.double).reshape((new_h,new_w))
#
#        pyramid += [im_down]

    # reverse the order so that the coarsest scale is first
    pyramid = pyramid[::-1]

    return pyramid

def downsample_single(im, downsampling_factor):
    new_w = int(np.ceil(im.shape[1]/downsampling_factor))
    new_h = int(np.ceil(im.shape[0]/downsampling_factor))

    method = 3

    if method == 2:
        # gaussian filtering followed by downscaling
        sigma = (np.sqrt(2.)/4.)*downsampling_factor
        im_down = gaussian_filter(im, sigma)
        im_down = affine_transform(im_down, [(im_down.shape[0]-1)*1.0/new_h, (im_down.shape[1]-1)*1.0/new_w], output_shape=[new_h,new_w], mode='nearest')
    elif method == 3:
        # PIL resizing
        im_down = np.array([Image.fromarray(im).resize((new_w, new_h), Image.Resampling.LANCZOS).getdata()], np.double).reshape((new_h, new_w))

    return im_down

def downscale_single(im, downscale_factor):
    # gaussian filtering
    sigma = (np.sqrt(2.)/4.)*downscale_factor
    im_down = gaussian_filter(im, sigma)

    return im_down

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    file_im = '../optimisation/mydata/barbara.png'

    print(file_im)

    IM = Image.open(file_im)

    im = np.array(IM.getdata()).reshape(IM.size[1], IM.size[0]).astype(np.double)

    im_down = downscale_single(im, 2.)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(im_down)

    plt.show()
