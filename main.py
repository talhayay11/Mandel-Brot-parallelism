import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os

from mandelbrot_plotting import plot_zoomable_mandelbrot
from mandelbrot_gpu import plot_mandelbrot_gpu
 

if __name__ == '__main__':
    
    plot_mandelbrot_gpu(max_iter=10000, width=1024, height=1024)
    plot_zoomable_mandelbrot(1000)
