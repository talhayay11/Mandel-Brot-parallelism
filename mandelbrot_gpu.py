import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

# CPU üzerinde Mandelbrot hesaplama
def mandelbrot_cpu(c, max_iter):
    z = complex(0, 0)
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

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

def compute_mandelbrot_cpu(min_x, max_x, min_y, max_y, width, height, max_iter):
    image = np.zeros((height, width), dtype=np.int32)
    start_time = time.time()
    for row in range(height):
        for column in range(width):
            x = min_x + column * (max_x - min_x) / width
            y = min_y + row * (max_y - min_y) / height
            image[row, column] = mandelbrot_cpu(complex(x, y), max_iter)
    elapsed_time = time.time() - start_time
    print(f"CPU computation time: {elapsed_time:.2f} seconds")
    return image

def compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter, num_grids):
    image_gpu = np.zeros((height, width), dtype=np.int32)
    threadsperblock = (32, 32)

    for grid_index in range(num_grids):
        blockspergrid_x = (width + (threadsperblock[0] - 1)) // threadsperblock[0]
        blockspergrid_y = (height + (threadsperblock[1] - 1)) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # CPU hesaplama süresi
        cpu_image = compute_mandelbrot_cpu(min_x, max_x, min_y, max_y, width, height, max_iter)

        start_time = time.time()
        mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, width, height, image_gpu, max_iter)
        cuda.synchronize()
        gpu_elapsed_time = time.time() - start_time
      

        for block_x in range(blockspergrid_x):
          for block_y in range(blockspergrid_y):
              gpu_block_time = gpu_elapsed_time / (blockspergrid_x * blockspergrid_y)
              print(f"Block ({block_x}, {block_y}) speedup: {gpu_block_time:.10f}")



        print(f"GPU computation time for grid {grid_index}: {gpu_elapsed_time:.2f} seconds")

    return image_gpu

def plot_mandelbrot_gpu(max_iter=1000, width=1024, height=1024, num_grids=1):
    fig, ax = plt.subplots(figsize=(10, 10))
    image = compute_mandelbrot_gpu(-2.0, 1.0, -1.5, 1.5, width, height, max_iter, num_grids)
    ax.imshow(image, extent=[-2.0, 1.0, -1.5, 1.5], origin='lower', cmap='hot')
    plt.show()

# Çizimi başlat
plot_mandelbrot_gpu()
