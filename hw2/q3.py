import numpy as np 
from scipy import signal
from math import pi, sqrt, exp

def matlab_style_gauss_filter(shape=(3,3),sigma=1.4):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# generate 2D Gaussian Filter
g_2d = matlab_style_gauss_filter(shape=(5,5))

# generate 1D Gaussian Filter for column vector
g_1d_v = matlab_style_gauss_filter(shape=(5,1))
# generate 1D Gaussian Filter for horizontal vector
g_1d_h = matlab_style_gauss_filter(shape=(1,5))
# use the seperability property to combine g_1d_v and g_1d_h into a 2D filter
g_1d_comb = np.multiply(g_1d_v, g_1d_h)

# compare whether g_1d_comb and g_2d are equal
print "2D Gaussian filter mask with 5x5 kernel: ", g_2d
print "Combined 1-D Gaussian filter masks with 5x5 kernel: ", g_1d_comb
print "They are equal? ", np.allclose(g_2d, g_1d_comb)