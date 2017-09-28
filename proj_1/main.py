from PIL import Image
import numpy as np
import glob
from diff import diff
from gaussian import gaussian

image_list = []
thres = 30 	# Threshold used to determine if something is moving through that pixel

#i. Read in a sequence of image frames and make them grayscale.
for filename in sorted(glob.glob('Red/*.jpg')):
    # im = Image.open(filename).convert('LA')
    # im_np = np.asarray(im.getdata())
    # print im_np
    im = Image.open(filename)
    im = im.convert('L') #makes it greyscale
    # y = np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))
    im_np = np.asarray(im.getdata(), dtype='uint8')
    # print type(im_np), im_np.shape  
    image_list.append(im_np)
    
print image_list, type(image_list)

# gaussian(image_list)

#Transpose array to be described as a list of lists, where each list is a single pixel's values as it changes over time
# trans_image_list = np.asarray(image_list).T.tolist()
image_array = np.asarray(image_list)
trans_array = np.transpose(image_array)

image_result = gaussian(trans_array, thres)

# #Note: ii and iii will be done at the same time in diff, since its a lot faster to combine both for loops

# #ii. As enough frames are available, apply a 1-D differential operator at each pixel to compute a temporal derivative.
# image_result = []

# for pix in trans_image_list:
# 	im_res = np.asarray(diff(pix,thres), dtype='uint8')
# 	image_result.append(im_res)

# print len(image_result[0])
# #iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.

# #iv. Combine the mask with the original frame to display the results.

# # Transpose the results to match the original image
# image_result = np.asarray(image_result).T
# # print image_result.shape


# output
#image_result = np.asarray(image_result, dtype='uint8') #if values still in range 0-255! 
i = 0
for single_image in image_result:
	out = single_image.reshape((240, 320))
	w = Image.fromarray(out, mode='L')
	w.save('out_%s.jpg' % i)
	i = i + 1
