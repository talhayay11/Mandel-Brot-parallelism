import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

@cuda.jit(device=True)
def mandelbrot_cuda(c, max_iter):
    z = complex(0, 0)
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, width, height, image, max_iter):
    column, row = cuda.grid(2)
    if column < width and row < height:
        x = min_x + column * (max_x - min_x) / width
        y = min_y + row * (max_y - min_y) / height
        color = mandelbrot_cuda(complex(x, y), max_iter)
        image[row, column] = color


def compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter, num_grids):
    image_gpu = np.zeros((height, width), dtype=np.int32)
    threadsperblock = (1,1)
    for grid_index in range(num_grids):
        blockspergrid_x = (width + (threadsperblock[0] - 1)) // threadsperblock[0]
        blockspergrid_y = (height + (threadsperblock[1] - 1)) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)


        start_time = time.time()
        mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, width, height, image_gpu, max_iter)
        cuda.synchronize()
        gpu_serie_time = time.time() - start_time
      
        print(f"Serie GPU computation time for grid {grid_index}: {gpu_serie_time:.2f} seconds")

    threadsperblock = (16,16)
    for grid_index in range(num_grids):
        blockspergrid_x = (width + (threadsperblock[0] - 1)) // threadsperblock[0]
        blockspergrid_y = (height + (threadsperblock[1] - 1)) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)


        start_time = time.time()
        mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, width, height, image_gpu, max_iter)
        cuda.synchronize()
        gpu_parallel_time = time.time() - start_time
      
        print(f"Parallel GPU computation time for grid {grid_index}: {gpu_parallel_time:.2f} secnds")
    
        Speedup = gpu_serie_time / gpu_parallel_time
        print(f"SpeedUp: {Speedup:.2f}")

        Efficiency = (Speedup*100)/32
        print(f"Efficiency Blocks 32x32: {Efficiency:.2f}")

    return image_gpu

def plot_mandelbrot_gpu(max_iter, width, height,num_grids ):
    fig, ax = plt.subplots(figsize=(10, 10))
    image = compute_mandelbrot_gpu(-2.0, 1.0, -1.5, 1.5, width, height, max_iter, num_grids)
    ax.imshow(image, extent=[-2.0, 1.0, -1.5, 1.5], origin='lower', cmap='hot')
    plt.show()

# Çizimi başlat
max_iter=1000
width=1024
height=1024
num_grids=1
plot_mandelbrot_gpu(max_iter,width,height,num_grids)
