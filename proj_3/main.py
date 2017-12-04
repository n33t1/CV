from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os, shutil
import imageio
from scipy import signal
import scipy
import random
from time import time
from skimage import feature 
# if see error message "cant find cv2", do 
# $ export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
import cv2
import copy
np.set_printoptions(threshold=np.nan)

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
# Take the Guassian derivative so it's smoothed in the process
# Could do this by creative the gaussian kernel first, but this should be fine since it's not the important part
Ssigma = 10
Ix = []
Iy = []
Ix.append(scipy.ndimage.filters.gaussian_filter1d(image_list[0], Ssigma, axis=1, order=1, mode='reflect', truncate=4.0))
Iy.append(scipy.ndimage.filters.gaussian_filter1d(image_list[0], Ssigma, axis=0, order=1, mode='reflect', truncate=4.0))
Ix.append(scipy.ndimage.filters.gaussian_filter1d(image_list[1], Ssigma, axis=1, order=1, mode='reflect', truncate=4.0))
Iy.append(scipy.ndimage.filters.gaussian_filter1d(image_list[1], Ssigma, axis=0, order=1, mode='reflect', truncate=4.0))

'''
3. Compute the temporal gradient It by subtracting a smoothed version of image1 from a
smoothed version of image2.
'''
Tsigma = 3
smooth_images = []
smooth_images.append(scipy.ndimage.filters.gaussian_filter(image_list[0], Tsigma, order=0, mode='reflect', truncate=4.0))
smooth_images.append(scipy.ndimage.filters.gaussian_filter(image_list[1], Tsigma, order=0, mode='reflect', truncate=4.0))

It = np.subtract(smooth_images[1], smooth_images[0])

'''
Display results so far
'''
# plt.subplot(3,3,1), plt.imshow(image_list[0])
# plt.title('Image 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,3,2), plt.imshow(image_list[1])
# plt.title('Image 2'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,3,3), plt.imshow(It)
# plt.title('It'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,3,4), plt.imshow(Ix[0])
# plt.title('Ix 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,3,5), plt.imshow(Ix[1])
# plt.title('Ix 2'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,3,7), plt.imshow(Iy[0])
# plt.title('Iy 1'), plt.xticks([]), plt.yticks([])
# plt.subplot(3,3,8), plt.imshow(Iy[1])
# plt.title('Iy 2'), plt.xticks([]), plt.yticks([])

# # Comment out this line to remove pause
# plt.show()

'''
4. For a given window size W, form a system of linear equations at each pixel by summing over
products of gradients in its neighborhood, as specified by the Lucas-Kanade method. That
is, at each pixel, you will have a set of equations:
'''
Ix, Iy, It = np.array(Ix, dtype='f'), np.array(Iy, dtype='f'), np.array(It, dtype='f')
window = np.ones((3, 3))
denom = cv2.filter2D(Ix**2, -1, window)*cv2.filter2D(Iy**2, -1, window) - cv2.filter2D((Ix*Iy), -1, window)**2


# we dont want to divide by 0. Replace 0 with MAX_INT
for i in range(len(denom)):
    for j in range(len(denom[0])):
        for k in range(len(denom[0][0])):
            if denom[i][j][k] == 0:
                denom[i][j][k] = float('inf')

'''
5. Solve for the flow vector [u, v] at each pixel. It is convenient to represent this vector field by
two images, one containing the u component, and the other the v component of flow.
'''
u = (-cv2.filter2D(Iy**2, -1, window)*cv2.filter2D(Ix*It, -1, window) + cv2.filter2D(Ix*Iy, -1, window)*cv2.filter2D(Iy*It, -1, window)) / denom
v = (cv2.filter2D(Ix*It, -1, window)*cv2.filter2D(Ix*Iy, -1, window) - cv2.filter2D(Ix**2, -1, window)*cv2.filter2D(Iy*It, -1, window)) / denom
# print u.shape, v.shape

'''
6. Display the flow vectors overlaid on the image. You can use matlab quiver to show the flow
field
'''
# Create grid for display
x = np.arange(0, image_list[0].shape[1], 1)
y = np.arange(0, image_list[0].shape[0], 1)
x, y = np.meshgrid(x, y)

# Display
# display the original image
plt.figure()
plt.imshow(image_list[0])
plt.title('Lucas_Kanade method')

# Overlay using plt.quiver
step = 3
plt.quiver(x[::step, ::step], y[::step, ::step], u[::step, ::step][0], v[::step, ::step][0])

plt.show()
