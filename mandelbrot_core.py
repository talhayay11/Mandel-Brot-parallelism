import numpy as np
import time
from numba import cuda
from concurrent.futures import ProcessPoolExecutor
import os

# def mandelbrot(c, max_iter):
#     z = complex(0, 0)
#     n = 0
#     while abs(z) <= 2 and n < max_iter:
#         z = z * z + c
#         n += 1
#     return n

def compute_mandelbrot_regionn(args):
    region_name, xmin, xmax, ymin, ymax, width, height, max_iter = args
    start_time = time.time()
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    C = C[:, 0] + 1j * C[:, 1]
    count = np.array([mandelbrot(c, max_iter) for c in C])
    elapsed_time = time.time() - start_time
    print(f"Region: {region_name}, Computation time: {elapsed_time:.2f} seconds, Process ID: {os.getpid()}")
    return count.reshape((height, width))

@cuda.jit(device=True)
def mandelbrot_cuda(c, max_iter):
    z = complex(0, 0)
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


processCount = [6, 5, 4, 3, 2, 1]
#processCount = [6,4]

def mandelbrot(c, max_iter):
    z = complex(0, 0)
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n

def compute_mandelbrot_region(args):
    region_name, xmin, xmax, ymin, ymax, width, height, max_iter = args
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    C = C[:, 0] + 1j * C[:, 1]

    start_time = time.time()
    count = np.array([mandelbrot(c, max_iter) for c in C])
    elapsed_time = time.time() - start_time
    #print(f"Process ID: {os.getpid()} Region: {region_name}, Computation time: {elapsed_time:.2f} seconds")
    
    return count.reshape((height, width))