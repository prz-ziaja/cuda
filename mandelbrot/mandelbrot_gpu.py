import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
import numpy as np
import matplotlib.pyplot as plt
import time
from timeit import timeit

min_x = -2
max_x = 2
min_y = -2
max_y = 2
samples = 1000
max_iter = 10
threshold = 10

mandel_ker = ElementwiseKernel(
    "pycuda::complex<float> *input, float *output, int max_iters, float threshold",
    """
    output[i] = 1;
    pycuda::complex<float> c = input[i];
    pycuda::complex<float> z(0,0);
    for(int j=0; j<max_iters; j++){
        z = z*z+c;
        if(abs(z)>threshold){
            output[i] = 0;
            break;
        }
    }
    """,
    "mandel_ker"
)

@timeit
def mandelbrot_gpu():
    x = np.linspace(min_x,max_x,samples,dtype=np.complex64).reshape(-1, 1)
    y = np.linspace(min_y,max_y,samples,dtype=np.complex64).reshape(1, -1) * 1j

    input = x+y

    output = gpuarray.empty(shape=input.shape,dtype=np.float32)
    mandel_ker(gpuarray.to_gpu(input), output,
                np.int32(max_iter),np.float32(threshold))

    return output.get()