import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches
from mandelbrot_core import mandelbrot_cuda
from mandelbrot_core import compute_mandelbrot_region
from concurrent.futures import ProcessPoolExecutor




@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, width, height, image, max_iter):
    column, row = cuda.grid(2)
    if column < width and row < height:
        x = min_x + column * (max_x - min_x) / width
        y = min_y + row * (max_y - min_y) / height
        color = mandelbrot_cuda(complex(x, y), max_iter)
        image[row, column] = color

class ZoomableMandelbrotGPU:
    def __init__(self, ax, max_iter, width=1024, height=1024):
        self.ax = ax
        self.max_iter = max_iter
        self.width = width
        self.height = height
        self.press = None
        self.xmin, self.xmax = -2.0, 1.0
        self.ymin, self.ymax = -1.5, 1.5

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.rect = patches.Rectangle((0,0), 0, 0, fill=False, edgecolor='white', linewidth=1.5)
        self.ax.add_patch(self.rect)
        self.plot_mandelbrot()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.press = event.xdata, event.ydata
        self.rect.set_width(0)
        self.rect.set_height(0)
        self.rect.set_xy((event.xdata, event.ydata))

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_width(dx)
        self.rect.set_height(dy)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if self.press is None or event.inaxes != self.ax:
            return
        xpress, ypress = self.press
        xrelease, yrelease = event.xdata, event.ydata
        self.xmin, self.xmax = sorted([xpress, xrelease])
        self.ymin, self.ymax = sorted([ypress, yrelease])
        self.press = None
        self.rect.set_width(0)
        self.rect.set_height(0)
        self.plot_mandelbrot()

    def plot_mandelbrot(self):
        image = compute_mandelbrot_gpu(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
        self.ax.clear()
        self.ax.imshow(image, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
        self.ax.set_title(f"Zoomed view: [{self.xmin}, {self.xmax}], [{self.ymin}, {self.ymax}]")
        self.ax.axis("off")
        self.ax.figure.canvas.draw_idle()


def compute_mandelbrot_gpu(min_x, max_x, min_y, max_y, width, height, max_iter):
    image = np.zeros((height, width), dtype=np.int32)
    threadsperblock = (16, 16)
    blockspergrid_x = (width + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (height + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, width, height, image, max_iter)
    start_time = time.time()
    cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"GPU computation time: {elapsed_time:.2f} seconds")

    return image

def plot_mandelbrot_gpu(max_iter=1000, width=1024, height=1024):
    fig, ax = plt.subplots(figsize=(10, 10))
    zoomable_mandelbrot_gpu = ZoomableMandelbrotGPU(ax, max_iter, width, height)
    plt.show()


