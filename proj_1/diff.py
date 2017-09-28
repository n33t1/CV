import numpy as np
import glob

def diff(pix,thres):
	result = []
	for i in range(0,len(pix)):
		if i == 0:
			result.append(pix[i])
		elif abs(pix[i - 1] - pix[i]) > thres:
			result.append(255)
		else:
			result.append(pix[i])
	
	# print result
	return result
