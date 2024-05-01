import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
from mandelbrot_core import compute_mandelbrot_region

class ZoomableMandelbrot:
    def __init__(self, ax, max_iter, width=800, height=800):
        self.ax = ax
        self.max_iter = max_iter
        self.width = width
        self.height = height
        self.press = None
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

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
        start_time = time.time()

        # Split the image into four quadrants for parallel processing
        x_mid = (self.xmin + self.xmax) / 2
        y_mid = (self.ymin + self.ymax) / 2

        # Define regions for parallel computation
        regions = [
            ("Bottom-left", self.xmin, x_mid, self.ymin, y_mid, self.width // 2, self.height // 2, self.max_iter),
            ("Bottom-right", x_mid, self.xmax, self.ymin, y_mid, self.width // 2, self.height // 2, self.max_iter),
            ("Top-left", self.xmin, x_mid, y_mid, self.ymax, self.width // 2, self.height // 2, self.max_iter),
            ("Top-right", x_mid, self.xmax, y_mid, self.ymax, self.width // 2, self.height // 2, self.max_iter)
        ]

        # Adjust the order of regions for (1,2,3,4) grid layout
        # Swap regions for top-left and bottom-right quadrants
        regions[2], regions[1] = regions[1], regions[2]

        # Create a ProcessPoolExecutor
        processesNumber = 1
        with ProcessPoolExecutor(max_workers=processesNumber) as executor:
            # Map compute_mandelbrot_region to regions and collect results
            results = list(executor.map(compute_mandelbrot_region, regions))

        # Rearrange results to form the complete image
        bottom_left, bottom_right, top_left, top_right = results
        top = np.concatenate((top_left, top_right), axis=1)
        bottom = np.concatenate((bottom_left, bottom_right), axis=1)
        final_result = np.concatenate((bottom, top), axis=0)

        elapsed_time = time.time() - start_time
        print(f"Process count: {processesNumber} Computation time: {elapsed_time:.2f} seconds")


        self.ax.clear()
        self.ax.imshow(final_result.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
        self.ax.set_title(f"Process count: {processesNumber} Computation time: {elapsed_time:.2f} seconds")
        self.ax.axis("off")


        self.ax.figure.canvas.draw_idle()

def plot_zoomable_mandelbrot(max_iter):
    fig, ax = plt.subplots(figsize=(8, 8))
    zoomable_mandelbrot = ZoomableMandelbrot(ax, max_iter)
    plt.show()


    