import numpy as np 
from scipy import signal

# init given matrix
input = [ [0.0 for x in range(8)] for x in range(8)]

for i in range(8):
    for j in range(8):
        input[i][j] = abs(i - j)

print np.asarray(input)
output = signal.medfilt(input, kernel_size=3)

# assume border not changed
for i in range(8):
    for j in range(8):
        if i == 0 or j == 0 or i == 7 or j == 7:
            output[i][j] = input[i][j]

#print output
