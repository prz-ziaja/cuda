import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import timeit

min_x = -2
max_x = 2
min_y = -2
max_y = 2
samples = 1000
max_iter = 100
threshold = 10

@timeit
def mandelbrot():
    x = np.linspace(min_x,max_x,samples,dtype=np.complex64).reshape(-1, 1)
    y = np.linspace(min_y,max_y,samples,dtype=np.complex64).reshape(1, -1) * 1j

    input = x+y
    output = np.ones(input.shape)

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            z = np.complex64(0)
            c = input[i,j]
            for k in range(max_iter):
                z = np.add(np.square(z),c)
                if(np.abs(z)>threshold):
                    output[i,j]=0
    return output
