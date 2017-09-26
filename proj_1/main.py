from PIL import Image
import numpy as np
import glob

image_list = []

#i. Read in a sequence of image frames and make them grayscale.
for filename in glob.glob('RedChair/*.jpg'):
    # im = Image.open(filename).convert('LA')
    # im_np = np.asarray(im.getdata())
    # print im_np
    im = Image.open(filename)
    im = im.convert('L') #makes it greyscale
    #y = np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))
    im_np = np.asarray(im.getdata(), dtype='uint8')
    print type(im_np), im_np.shape  
    #image_list.append(im_np)
    
#print image_list

#ii. As enough frames are available, apply a 1-D differential operator at each pixel to compute a temporal derivative.

#iii. Threshold the absolute values of the derivatives to create a 0 and 1 mask of the moving objects.

#iv. Combine the mask with the original frame to display the results.


# output
# y=np.asarray(y,dtype=np.uint8) #if values still in range 0-255! 
# w=Image.fromarray(y,mode='L')
# w.save('out.jpg')