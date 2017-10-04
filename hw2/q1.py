import numpy as np

# Generate 10 256x256 images with grey scale 0.5
pure = np.ones((256, 256, 10)) * 0.5

# Generate Gaussian noise for the images
noise = np.random.normal(0,2.0,(256, 256, 10))

# Add image and noise together
signal = pure + noise

# Using EST NOISE to estimate the noise in the images
print "mean for EST NOISE: ", np.mean(signal)
print "std for EST NOISE: ", np.std(signal)
