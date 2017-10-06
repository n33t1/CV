from PIL import Image
import numpy as np
import glob
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
import os, shutil
import imageio

image_list = []

"""function takes np arrary and threshold as input, outputs np array with temporal gaussian filtering"""
def gaussian(trans_array, thres):
    trans_array = trans_array.T
	# prompt user input
    tsigma = input('Enter standard deviation tsigma for Gaussian filter: ')
    result = []
    for time_axis in trans_array:
		out = gaussian_filter(time_axis, order=1, sigma=tsigma)
		for pixel in out:
			if pixel < thres:
				pixel = 0
			else:
				pixel = 255
		result.append(out)

    # Transpose the results to match the original image
    trans_result_array = np.asarray(result, dtype='uint8')
    result_array = np.transpose(trans_result_array)
    return result_array

"""function takes np arrary and threshold as input, outputs np array with temporal linear filtering"""
def diff(trans_image,thres):
	result = []
	trans_image_list = trans_image.T.tolist()
	for time_axis in trans_image_list:
		pix_res = []
		pix_res.append(0)
		#iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.
		for i in range(1,len(time_axis) - 1):
    			if abs(time_axis[i - 1] - time_axis[i + 1]) > thres:
				pix_res.append(255)
			else:
				pix_res.append(0)
#		for i in range(1,len(time_axis) - 1):
#			pix_res.append(abs(time_axis[i - 1] - time_axis[i + 1]) )
		#iv. Combine the mask with the original frame to display the results.
		pix_res.append(0)
		pix_res_arr = np.asarray(pix_res)
		result.append(pix_res_arr)

#	print np.histogram(result, bins=np.linspace(0, 255, num = 52))
	# Transpose the results to match the original image
	trans_result_array = np.asarray(result, dtype='uint8')
	result_array = np.transpose(trans_result_array)

	return result_array

def output_image(images_output, name):
	dir_name = name+'_Results'
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	else:
		shutil.rmtree(dir_name)
		os.makedirs(dir_name)
	image_result = np.asarray(images_output, dtype='uint8') #if values still in range 0-255! 
	result = []
	for single_image in image_result:
		single_image = single_image.reshape((240, 320))
        result.append(np.asarray(single_image))

	image_result = np.asarray(result, dtype='uint8')
	imageio.mimsave(dir_name+'.gif', image_result)

"""function takes np arrary and kernel size n as input, outputs np array with spatial box smoothing"""
def BoxSmooth(input_image, n):
	result = []
	# Create an NxN box kernel to use for filtering
	kernel = np.linspace(1, 1, num = n * n).reshape((n,n))

	for image in input_image:
		out = image.reshape(240,320)
		out = signal.convolve2d(out, kernel, mode='same', boundary='symm')
		out = out.reshape(76800)
		out /= kernel.sum()
		result.append(out)

	return np.asarray(result, dtype='uint8')

"""function takes np arrary and threshold as input, outputs np array with temporal gaussian filtering"""
def GaussianSmooth(input_image):
	# prompt user input
	ssigma = input('Enter standard deviation ssigma for Gaussian filter: ')
	result = []

	for image in image_list:
		out = image.reshape(240,320)
		out = gaussian_filter(out, sigma=ssigma)
		out = out.reshape(76800)
		result.append(out)

	return np.asarray(result, dtype='uint8')
    
def preprocess_image(dir_name):
    #i. Read in a sequence of image frames and make them grayscale.
    for filename in sorted(glob.glob(dir_name+'/*.jpg')):
        im = Image.open(filename)
        im = im.convert('L') #makes it greyscale
        im_np = np.asarray(im.getdata(), dtype='uint8')
        image_list.append(im_np)

    #Transpose array to be described as a list of lists, where each list is a single pixel's values as it changes over time
    return np.asarray(image_list)

# def output_image(images_output, name):
#     dir_name = 'results_'+name
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     else:
#         shutil.rmtree(dir_name)
#         os.makedirs(dir_name)
#     image_result = np.asarray(images_output, dtype='uint8') #if values still in range 0-255! 
#     i = 0
#     for single_image in image_result:
#         out = single_image.reshape((240, 320))
#         w = Image.fromarray(out, mode='L')
#         w.save(dir_name+'/out_%s.jpg' % i)
#         i = i + 1

def select_dataset(flag):
    print ("""
        Which dataset do you want to use?
            1. Office
            2. Red chair
            3. EnterExitCrossingPaths2cor
        """)
    dataset=raw_input("Please Select:") 
    if dataset =='1': 
        print "Your image is being processed..."
        return preprocess_image("Office")
    elif dataset == '2': 
        print "Your image is being processed..."
        return preprocess_image("RedChair")
    elif dataset == '3':
        print "Your image is being processed..."
        return preprocess_image("EnterExitCrossingPaths2cor")
    else: 
        print "Unknown Option Selected!" 
        flag = False
        return

def select_spatial_filter(trans_image_list, thres):
    print ("""
    Which spatial filter do you want to use?
        1. 3x3 box filter
        2. 5x5 box filter
        3. 2D Gaussian box filter
        4. I don't want to use any spatial filter
    """)

    selection=raw_input("Please Select:") 
    if selection =='1': 
        print "Program is running..."
        return [BoxSmooth(trans_image_list, 3), "3x3BoxSmooth_"]
    elif selection == '2': 
        print "Program is running..."
        return [BoxSmooth(trans_image_list, 5), "5x5BoxSmooth_"]
    elif selection == '3':
        print "Program is running..."
        return [GaussianSmooth(trans_image_list), "GaussianSmooth_"]
    elif selection == '4':
        return [trans_image_list, "Native_"]
    else: 
        print "Unknown Option Selected!" 

def select_temporal_filter(filtered_trans_image_list, thres):
    while True: 
        print ("""
        Which function do you want to use?
            1. Linear temporal derivative filter
            2. 1D Gaussian temporal derivative filter
            3. Exit/Quit
        """)

        selection=raw_input("Please Select:") 
        if selection =='1': 
            print "Program is running..."
            images_output = diff(filtered_trans_image_list[0], thres)
            output_image(images_output, filtered_trans_image_list[1]+"_LinearTemporal")
        elif selection == '2': 
            print "Program is running..."
            images_output = gaussian(filtered_trans_image_list[0], thres)
            output_image(images_output, filtered_trans_image_list[1]+"_GaussianTemporal")
        elif selection == '3':
            break
        else: 
            print "Unknown Option Selected!" 
    return

# main
# prompt the user to select dataset
flag = False
while not flag:
    flag = True
    trans_image_list = select_dataset(flag)

thres = input("Please enter a threshold value: ") 	# Threshold used to determine if something is moving through that pixel
# prompt the user to select a function to use
select_temporal_filter(select_spatial_filter(trans_image_list, thres), thres)
