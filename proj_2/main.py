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
    for filename in sorted(glob.glob(dir_name+'/*.JPG')):
        if len(image_list) < 2:
            img = cv2.imread(filename, 0)
            imgColor = cv2.imread(filename)
            RGB_img = cv2.cvtColor(imgColor, cv2.COLOR_BGR2RGB)
            # im = Image.open(filename)
            # im_np = np.asarray(im.getdata(), dtype='uint8')
            color_image_list.append(RGB_img)
            image_list.append(img)
    #Transpose array to be described as a list of lists, where each list is a single pixel's values as it changes over time
    return np.asarray(image_list)

'''
i. Read in two images.
'''
image_list = []
corner_list = []
color_image_list = []
preprocess_image('DanaHallWay2')


'''
ii. Apply Harris corner detector to both images: compute Harris R function over the
image, and then do non-maimum suppression to get a sparse set of corner features.
'''
for img in image_list:
    corner = cv2.goodFeaturesToTrack(img, 60, 1e-6, 5, useHarrisDetector=True)
    corner_list.append(corner)

'''
iii. Find correspondences between the two images: given two set of corners from the
two images, compute normalized cross correlation (NCC) of image patches centered
at each corner. (Note that this will be O(n^2) process.) 

Choose potential corner matches by finding pair of corners (one from each image) such that they 
have the highest NCC value. You may also set a threshold to keep only matches that have a
large NCC score.

Since we have O(n^2) run time, we want to use nested for loop to compare NCC scores for corner_A from cornerlist_A to all the corners in cornerlist_B, 
return corner_A, conrer_B with max(NCC score) as our desired corner pair

'''
radius = 5
m, n = max(image_list[0].shape[0], image_list[1].shape[0]), max(image_list[0].shape[1], image_list[1].shape[1])

# check if the corner is in the center of the image
def inBound(x, y):
    return x - radius > -1 or x + radius < m or y - radius > -1 or y + radius < n;
    
cornerlistA, cornerlistB = corner_list[0].astype(int), corner_list[1].astype(int)
lenCornerlistA, lenCornerlistB = cornerlistA.shape[0], cornerlistB.shape[0]
NCCVals = np.zeros((lenCornerlistA, lenCornerlistB))

cornerlistA = [x[0] for x in cornerlistA]
cornerlistB= [x[0] for x in cornerlistB]

i = 0
for A in cornerlistA:
    cornerX_a, cornerY_a = A[1],A[0]
    if inBound(cornerX_a, cornerY_a):
        j = 0
        for B in cornerlistB:
            cornerX_b, cornerY_b = B[1],B[0]
            if inBound(cornerX_b, cornerY_b):
                patchA = [image_list[0][x][cornerY_a-radius:cornerY_a+radius+1] for x in range(cornerX_a-radius, cornerX_a+radius+1)]
                patchB = [image_list[1][x][cornerY_b-radius:cornerY_b+radius+1] for x in range(cornerX_b-radius, cornerX_b+radius+1)]
                patchANormalized = patchA/np.linalg.norm(patchA)
                patchBNormalized = patchB/np.linalg.norm(patchB)
                NCCVals[i][j] = np.sum(patchANormalized * patchBNormalized)
                j += 1
    i += 1

ColorImageAB = np.concatenate((color_image_list[0],color_image_list[1]),axis=0)
ColorImageAB_Compare = copy.deepcopy(ColorImageAB)

ColorImageA_Pers = copy.deepcopy(color_image_list[0])
ColorImageB_Pers = copy.deepcopy(color_image_list[1])

imageACopy, imageBCopy = color_image_list[0], color_image_list[1]

src = []
dst = []

thresNCC = 0.9
for i in range(len(NCCVals)):
    for j in range(len(NCCVals[0])):
        if NCCVals[i][j] == np.max(NCCVals[i]) and NCCVals[i][j] > thresNCC:
            cornerPairA = (cornerlistA[i][0],cornerlistA[i][1])
            cornerPairB = (cornerlistB[j][0],cornerlistB[j][1])
            src.append(cornerPairA)
            dst.append(cornerPairB)
            # Draw correspondence pairs before RANSAC
            tmp = (cornerPairB[0], cornerPairB[1]+m)
            cv2.circle(imageACopy, cornerPairA, 5, (255,0,0))
            cv2.circle(imageBCopy, cornerPairB, 5, (255,0,0))
            cv2.line(ColorImageAB, cornerPairA, tmp, (255,0,0))

'''
iv. Estimate the homography using the above correspondences. Note that these correspondences
are likely to have many errors (outliers). That is ok: you should use
RANSAC to robustly estimate the homography from the noisy correspondences:

A. Repeatedly sample minimal number of points needed to estimate a homography
(4 pts in this case).
B. Compute a homography from these four points.
C. Map all points using the homagraphy and comparing distances between predicted
and observed locations to determine the number of inliers.
D. At the end, compute a least-squares homgraphy from ALL the inliers in the
largest set of inliers.
'''

# Parameters for RANSAC parameters
nIterMax, numSample, thresDistance, thresNumMatch = 1000, 4, 50, 200

src = np.array(src).astype('float')
dst = np.array(dst).astype('float')
cnt = np.zeros((src.shape[0],1))

for num in range(nIterMax):
    randSamplePoints = random.sample(range(len(src)), numSample)
    srcSample = src[randSamplePoints,:]
    dstSample = dst[randSamplePoints,:]
    h, status = cv2.findHomography(srcSample, dstSample, cv2.RANSAC, 5)=
    for i in list(set(range(len(src))) - set(randSamplePoints)):
        dis_x = dst[i,0] - (h[0,:].dot(np.hstack((src[i],1)).reshape(3,1))) / (h[2,:].dot(np.hstack((src[i],1)).reshape(3,1)))
        dis_y = dst[i,1] - (h[1,:].dot(np.hstack((src[i],1)).reshape(3,1))) / (h[2,:].dot(np.hstack((src[i],1)).reshape(3,1)))
        if ((dis_x * dis_x) + (dis_y * dis_y))**(.5) < thresDistance:
            cnt[i,0] += 1

selectedPairs = []
for i in range(len(src)):
    if cnt[i,0] >= thresNumMatch:
        s = (src[i,0].astype('int'), src[i,1].astype('int'))
        d = (dst[i,0].astype('int'), dst[i,1].astype('int') + m)
        cv2.line(ColorImageAB_Compare, s, d, (0,0,255))
        selectedPairs.append(i)

# Calculate Precise Homography
srcSample = src[selectedPairs,:]
dstSample = dst[selectedPairs,:]
h, status = cv2.findHomography(src, dst, cv2.RANSAC, 5)

ColorImageAB_Pers = cv2.warpPerspective(ColorImageB_Pers, h,
 			(ColorImageB_Pers.shape[1] + 300, ColorImageB_Pers.shape[0] + 100), flags=cv2.WARP_INVERSE_MAP)

for i in range(len(ColorImageA_Pers)):
    for j in range(len(ColorImageA_Pers[0])):
        if ColorImageAB_Pers[i][j].all() == 0:
            ColorImageAB_Pers[i][j] = ColorImageA_Pers[i][j]
        elif ColorImageA_Pers[i][j].any() != 0:
            ColorImageAB_Pers[i][j] = ColorImageA_Pers[i][j] / [2,2,2] + ColorImageAB_Pers[i][j] / [2,2,2]

plt.figure(1)
plt.imshow(imageACopy)
plt.figure(2)
plt.imshow(imageBCopy)
plt.figure(3)
plt.imshow(ColorImageAB)
plt.figure(4)
plt.imshow(ColorImageAB_Compare)
plt.figure(5)
plt.imshow(ColorImageAB_Pers)
plt.show()