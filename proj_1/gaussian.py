import numpy as np
from PIL import Image
import glob
from scipy.ndimage.filters import gaussian_filter

# function takes np arrary and thres as input, output a np array classified with only \
# pixel value of 0 or 255

def gaussian(trans_array, thres):
    # prompt user input
    tsigma = input('Enter standard deviation tsigma for Gaussian filter: ')
    out = gaussian_filter(trans_array, sigma=tsigma)
    result = []

    for time_axis in out:
        pix_res = np.array([255 if pix > thres else 0 for pix in time_axis])
        result.append(pix_res)

    trans_result_array = np.asarray(result, dtype='uint8')
    result_array = np.transpose(trans_result_array)
    
    return result_array
