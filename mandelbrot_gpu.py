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
        z = z*z+c
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
    block_size = [1, 8,12, 16,24, 32]
    all_time = []

    for i in range(len(block_size)):
        threadsperblock = (block_size[i],block_size[i])
        for grid_index in range(num_grids):
            blockspergrid_x = (width + (threadsperblock[0] - 1)) // threadsperblock[0]
            blockspergrid_y = (height + (threadsperblock[1] - 1)) // threadsperblock[1]
            blockspergrid = (blockspergrid_x, blockspergrid_y)


            start_time = time.time()
            mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, width, height, image_gpu, max_iter)
            cuda.synchronize()
            gpu_parallel_time = time.time() - start_time
            all_time.append(gpu_parallel_time)
        

    return image_gpu, all_time,block_size 




def plot_mandelbrot_gpu(max_iter, width, height,num_grids ):
    fig, ax = plt.subplots(figsize=(10, 10))
    min_x, max_x, min_y, max_y = -2.0, 1.0, -1.5, 1.5
    image, all_time,block_size  = compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter, num_grids)
    im = ax.imshow(image, extent=[min_x, max_x, min_y, max_y], origin='lower', cmap='hot')

    Speeduplist = []
    Efficiencylist = []
    for i in range(len(all_time)):
        print(f"Parallel GPU computation time for grid {block_size[i]}x{block_size[i]}: {all_time[i]:.4f} secnds")
        Speedup = all_time[0] / all_time[i]
        Speeduplist.append(Speedup)
        print(f"SpeedUp: {Speedup:.2f}")
        Efficiency = (Speedup)*100/block_size[i]**2
        Efficiencylist.append(Efficiency)
        print(f"Efficiency: {Efficiency:.4f}")


    def on_scroll(event):
        nonlocal min_x, max_x, min_y, max_y
        if event.xdata is None or event.ydata is None:
            return  # Outside of plot
        zoom_factor = 0.5
        if event.button == 'up':  # zoom in
            zoom_factor = 1 - zoom_factor
        else:  # zoom out
            zoom_factor = 1 + zoom_factor
        new_width = (max_x - min_x) * zoom_factor
        new_height = (max_y - min_y) * zoom_factor
        center_x = event.xdata
        center_y = event.ydata

        min_x = center_x - (center_x - min_x) * zoom_factor
        max_x = center_x + (max_x - center_x) * zoom_factor
        min_y = center_y - (center_y - min_y) * zoom_factor
        max_y = center_y + (max_y - center_y) * zoom_factor

        image = compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter, num_grids)
        im.set_data(image)
        im.set_extent([min_x, max_x, min_y, max_y])
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()
    return all_time, Speeduplist, Efficiencylist







def Compute_Md_Gpu(max_iter, width, height,num_grids ):
    min_x, max_x, min_y, max_y = -2.0, 1.0, -1.5, 1.5
    image, all_time,block_size  = compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter, num_grids)


    Speeduplist = []
    Efficiencylist = []
    for i in range(len(all_time)):
        print(f"Parallel GPU computation time for grid {block_size[i]}x{block_size[i]}: {all_time[i]:.4f} secnds")
        Speedup = all_time[0] / all_time[i]
        Speeduplist.append(Speedup)
        print(f"SpeedUp: {Speedup:.2f}")
        Efficiency = (Speedup)*100/block_size[i]**2
        Efficiencylist.append(Efficiency)
        print(f"Efficiency: {Efficiency:.4f}")


    def on_scroll(event):
        nonlocal min_x, max_x, min_y, max_y
        if event.xdata is None or event.ydata is None:
            return  # Outside of plot
        zoom_factor = 0.5
        if event.button == 'up':  # zoom in
            zoom_factor = 1 - zoom_factor
        else:  # zoom out
            zoom_factor = 1 + zoom_factor
        new_width = (max_x - min_x) * zoom_factor
        new_height = (max_y - min_y) * zoom_factor
        center_x = event.xdata
        center_y = event.ydata

        min_x = center_x - (center_x - min_x) * zoom_factor
        max_x = center_x + (max_x - center_x) * zoom_factor
        min_y = center_y - (center_y - min_y) * zoom_factor
        max_y = center_y + (max_y - center_y) * zoom_factor

        image = compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter, num_grids)
        im.set_data(image)
        im.set_extent([min_x, max_x, min_y, max_y])
        ax.figure.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()
    return all_time, Speeduplist, Efficiencylist



# Çizimi başlat
max_iter=100
width=1024
height=1024
num_grids=1
#plot_mandelbrot_gpu(max_iter,width,height,num_grids)