import numpy as np 
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from math import pi, sqrt, exp

def gauss(n = 10,sigma=1.4):
    print "n[0]", n[0]
    r = range(-int(n[0]/2),int(n[0]/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

def matlab_style_gauss2D(shape=(10,10),sigma=1.4):
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

print matlab_style_gauss2D()
print gauss((5,5),1)


import numpy as np 
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

input_1d_h = np.float_([0,0,0,0,1,0,0,0,0])
input_1d_v = np.float_([[0],[0],[0],[0],[1],[0],[0],[0],[0]])
input_2d = np.multiply(np.float_([0,0,0,0,1,0,0,0,0]),np.float_([[0],[0],[0],[0],[1],[0],[0],[0],[0]]))

output_1d_h = gaussian_filter(input_1d_h, sigma = 1.4)
output_1d_v = gaussian_filter(input_1d_v, sigma = 1.4)
output_1d_combined = np.multiply(output_1d_h, output_1d_v)
output_2d = gaussian_filter(input_2d, sigma = 1.4)
# print output_2d
# print "---------"
# print output_1d_combined

print np.subtract(output_1d_combined, output_2d) 
