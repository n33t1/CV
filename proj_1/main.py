from PIL import Image
import numpy as np
import glob
from scipy.ndimage.filters import gaussian_filter

image_list = []
thres = 50 	# Threshold used to determine if something is moving through that pixel

"""function takes np arrary and threshold as input, output a np array classified with only 
pixel value of 0 or 255"""
def gaussian(trans_array, thres):
    # prompt user input
    tsigma = input('Enter standard deviation tsigma for Gaussian filter: ')
    out = gaussian_filter(trans_array, order=1, sigma=tsigma)
    result = []

    for time_axis in out:
        pix_res = np.array([255 if pix > thres else pix for pix in time_axis])
        result.append(pix_res)

    # Transpose the results to match the original image
    trans_result_array = np.asarray(result, dtype='uint8')
    result_array = np.transpose(trans_result_array)
    
    return result_array

"""function takes np arrary and threshold as input, output a np array classified with only 
pixel value of 0 or 255"""
def diff(trans_image,thres):
    	result = []
	trans_image_list = trans_image.tolist()
	for time_axis in trans_image_list:
		pix_res = []
		#iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.
		for i in range(0,len(time_axis)):
			if i == 0 or i == len(time_axis) - 1:
				pix_res.append(time_axis[i])
			elif abs(time_axis[i - 1] - time_axis[i + 1]) > thres:
				pix_res.append(255)
			else:
				pix_res.append(time_axis[i])
		#iv. Combine the mask with the original frame to display the results.
		pix_res_arr = np.asarray(pix_res)
		result.append(pix_res_arr)
	
	# Transpose the results to match the original image
	trans_result_array = np.asarray(result, dtype='uint8')
	result_array = np.transpose(trans_result_array)

	return result_array

"""function takes np arrary as input, output a np array classified with only 
pixel value of 0 or 255"""
def Smooth3x3(input_image):
	result = []
	for image in input_image:
		pix_res = []
		#iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.
		for pix_index in range(0,len(image)):
			# Top Row
			if pix_index < 320:
				# Left Corner
				if pix_index == 0:
					pix_res.append(image[pix_index] / 4 + image[pix_index + 1] / 4 + image[pix_index + 320] / 4 + image[pix_index + 321] / 4)
				# Right Corner
				elif pix_index == 319:
					pix_res.append(image[pix_index] / 4 + image[pix_index - 1] / 4 + image[pix_index + 319] / 4 + image[pix_index + 320] / 4)
				else:
					pix_res.append(image[pix_index] / 6 + image[pix_index - 1] / 6 + image[pix_index + 1] / 6 + image[pix_index + 319] / 6 + image[pix_index + 320] / 6 + image[pix_index + 321] / 6)
			# Bottom Row
			elif pix_index > 76479:
				# Left Corner
				if pix_index == 76480:
					pix_res.append(image[pix_index] / 4 + image[pix_index + 1] / 4 + image[pix_index - 319] / 4 + image[pix_index - 320] / 4)
				# Right Corner
				elif pix_index == 76799:
						pix_res.append(image[pix_index] / 4 + image[pix_index - 1] / 4 + image[pix_index - 320] / 4 + image[pix_index - 321] / 4)
				else:
					pix_res.append(image[pix_index] / 6 + image[pix_index - 1] / 6 + image[pix_index + 1] / 6 + image[pix_index - 319] / 6 + image[pix_index - 320] / 6 + image[pix_index - 321] / 6)
			# Left Column
			elif pix_index % 320 == 0:
				pix_res.append(image[pix_index] / 6 + image[pix_index + 1] / 6 + image[pix_index - 319] / 6 + image[pix_index - 320] / 6 + image[pix_index + 320] / 6 + image[pix_index + 321] / 6)
			# Right Column
			elif (pix_index + 1) % 320 == 0:
				pix_res.append(image[pix_index] / 6 + image[pix_index - 1] / 6 + image[pix_index - 320] / 6 + image[pix_index - 321] / 6 + image[pix_index + 319] / 6 + image[pix_index + 320] / 6)
			# Regular Pixel
			else:
				pix_res.append(image[pix_index - 321] / 9 + image[pix_index - 320] / 9 + image[pix_index - 319] / 9 + image[pix_index - 1] / 9 + image[pix_index] / 9 + image[pix_index + 1] / 9 + image[pix_index + 319] / 9 + image[pix_index + 320] / 9 + image[pix_index + 321] / 9)

			#iv. Combine the mask with the original frame to display the results.
		pix_res_arr = np.asarray(pix_res)
		result.append(pix_res_arr)

	return np.asarray(result, dtype='uint8')

"""function takes np arrary as input, output a np array classified with only 
pixel value of 0 or 255"""
def Smooth3x1(input_image):
	result = []
	for image in input_image:
		pix_res = []
		#iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.
		for pix_index in range(0,len(image)):
			# Left Column
			if pix_index % 320 == 0:
				pix_res.append(image[pix_index] / 2 + image[pix_index + 1] / 2)
			# Right Column
			elif (pix_index + 1) % 320 == 0:
				pix_res.append(image[pix_index] / 2 + image[pix_index - 1] / 2)
			# Regular Pixel
			else:
				pix_res.append(image[pix_index] / 3 + image[pix_index - 1] / 3 + image[pix_index + 1] / 3)

			#iv. Combine the mask with the original frame to display the results.
		pix_res_arr = np.asarray(pix_res)
		result.append(pix_res_arr)

	return np.asarray(result, dtype='uint8')

"""function takes np arrary as input, output a np array classified with only 
pixel value of 0 or 255"""
def Smooth1x3(input_image):
	result = []
	for image in input_image:
		pix_res = []
		#iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.
		for pix_index in range(0,len(image)):
			# Top Row
			if pix_index < 320:
				pix_res.append(image[pix_index] / 2 + image[pix_index + 320] / 2)
			# Bottom Row
			elif pix_index > 76479:
				pix_res.append(image[pix_index] / 2 + image[pix_index - 320] / 2)
			# Regular Pixel
			else:
				pix_res.append(image[pix_index - 320] / 3 + image[pix_index] / 3 + image[pix_index + 320] / 3)

			#iv. Combine the mask with the original frame to display the results.
		pix_res_arr = np.asarray(pix_res)
		result.append(pix_res_arr)

	return np.asarray(result, dtype='uint8')

def output_image(images_output):
    image_result = np.asarray(images_output, dtype='uint8') #if values still in range 0-255! 
    i = 0
    for single_image in image_result:
        out = single_image.reshape((240, 320))
        w = Image.fromarray(out, mode='L')
        w.save('ress/out_%s.jpg' % i)
        i = i + 1
    
#i. Read in a sequence of image frames and make them grayscale.
for filename in sorted(glob.glob('RedChair/*.jpg')):
    im = Image.open(filename)
    im = im.convert('L') #makes it greyscale
    im_np = np.asarray(im.getdata(), dtype='uint8')
    image_list.append(im_np)

images_output = Smooth1x3(Smooth3x1(image_list))

#Transpose array to be described as a list of lists, where each list is a single pixel's values as it changes over time
trans_image_list = np.asarray(images_output).T

# images_output = gaussian(trans_image_list, thres)
images_output = diff(trans_image_list, thres)

#output the image
output_image(images_output)
