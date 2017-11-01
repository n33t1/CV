from PIL import Image
import numpy as np
import glob
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
import os, shutil
import imageio

def output_image(images_output, name):
    dir_name = name+'_Results'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    image_result = np.asarray(images_output, dtype='uint8') #if values still in range 0-255! 
    i = 0
    for single_image in image_result:
        out = single_image.reshape((240, 320))
        w = Image.fromarray(out, mode='L')
        w.save(dir_name+'/out_%s.jpg' % i)
        i = i + 1

def preprocess_image(dir_name):
    #i. Read in a sequence of image frames and make them grayscale.
    for filename in sorted(glob.glob(dir_name+'/*.jpg')):
        im = Image.open(filename)
        im = im.convert('L') #makes it greyscale
        im_np = np.asarray(im.getdata(), dtype='uint8')
        image_list.append(im_np)

    #Transpose array to be described as a list of lists, where each list is a single pixel's values as it changes over time
    return np.asarray(image_list)