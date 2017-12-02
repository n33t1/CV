from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os, shutil
import imageio
from scipy import signal
import random
from time import time
from skimage import feature 
# if see error message "cant find cv2", do 
# $ export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
import cv2
import copy
# np.set_printoptions(threshold=np.nan)

def preprocess_image(dir_name):
    #Read in two images from the same directory. 
    for filename in sorted(glob.glob(dir_name+'/*.pgm')):
        if len(image_list) < 2:
            img = cv2.imread(filename, 0)
            image_list.append(img)
    return np.asarray(image_list)

'''
1. Read image1 and image2, and convert to double flow greyscale image frames
'''
image_list = []
preprocess_image('LKTestpgm')

'''
2. Compute the spatial intensity gradients Ix and Iy of image2. Recall that it is a good idea to
smooth before taking the derivative, for example by using derivative of Gaussian operators.
'''

'''
3. Compute the temporal gradient It by subtracting a smoothed version of image1 from a
smoothed version of image2.
'''

'''
4. For a given window size W, form a system of linear equations at each pixel by summing over
products of gradients in its neighborhood, as specified by the Lucas-Kanade method. That
is, at each pixel, you will have a set of equations:
'''

'''
5. Solve for the flow vector [u, v] at each pixel. It is convenient to represent this vector field by
two images, one containing the u component, and the other the v component of flow.
'''

'''
6. Display the flow vectors overlaid on the image. You can use matlab quiver to show the flow
field
'''

plt.figure(1)
plt.imshow(image_list[0])
plt.figure(2)
plt.imshow(image_list[0])
plt.show()