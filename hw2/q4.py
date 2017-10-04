import numpy as np 
from scipy import signal
import cv2

# init given matrix
input_1 = np.array([10.0 for x in range(5)])
input_2 = np.array([40.0 for x in range(5)])
n = np.concatenate((input_1, input_2), axis=0)

# define filters
filter_a = np.array(np.multiply(1.0/5.0, np.array([1,1,1,1,1])))
filter_b = np.array(np.multiply(1.0/10.0, np.array([1,2,4,2,1])))

# convolve
convoleOutput_a = cv2.filter2D(n, -1, filter_a)
print "convoleOutput_a: ", convoleOutput_a
convoleOutput_b = cv2.filter2D(n, -1, filter_b)
print "convoleOutput_b: ", convoleOutput_b
