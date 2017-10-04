import numpy as np 
from scipy import signal

# init given matrix
input_1 = np.array([4 for x in range(4)])
input_2 = np.array([8 for x in range(4)])
n = np.concatenate((input_1, input_2), axis=0)

# with median filter
output = signal.medfilt(n, kernel_size=3)
print output

# with average filter
mask = np.array(np.multiply(1.0/4.0, np.array([1,2,1]))).tolist()

output = []
for i in range(len(n)-2):
    x, y, z = n[i:i+3]
    x, y, z = x * mask[0], y * mask[1], z * mask[2]
    output.append(sum([x, y, z]))

print output