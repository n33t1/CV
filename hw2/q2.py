import numpy as np
import cv2

# Generate 10 256x256 images with grey scale 0.5
pure = np.ones((256, 256, 10)) * 0.5

# Generate Gaussian noise for the images
noise = np.random.normal(0,2.0,(256, 256, 10))

# Add image and noise together
signal = pure + noise

# Smoothing noise for the images
smooth = cv2.GaussianBlur(signal,(3,3),0)

print smooth

# Using EST NOISE to estimate the noise in the images
print "mean for EST NOISE: ", np.mean(smooth)
print "std for EST NOISE: ", np.std(smooth)
