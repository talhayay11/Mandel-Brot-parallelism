import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
from mandelbrot_core import compute_mandelbrot_region

# class ZoomableMandelbrot:
#     def __init__(self, ax, max_iter, width=800, height=800):
#         self.ax = ax
#         self.max_iter = max_iter
#         self.width = width
#         self.height = height
#         self.press = None
#         self.xmin, self.xmax = -2.0, 2.0
#         self.ymin, self.ymax = -2.0, 2.0

#         self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
#         self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

#         self.rect = patches.Rectangle((0,0), 0, 0, fill=False, edgecolor='white', linewidth=1.5)
#         self.ax.add_patch(self.rect)
#         self.plot_mandelbrot()

#     def on_press(self, event):
#         if event.inaxes != self.ax:
#             return
#         self.press = event.xdata, event.ydata
#         self.rect.set_width(0)
#         self.rect.set_height(0)
#         self.rect.set_xy((event.xdata, event.ydata))

#     def on_motion(self, event):
#         if self.press is None or event.inaxes != self.ax:
#             return
#         xpress, ypress = self.press
#         dx = event.xdata - xpress
#         dy = event.ydata - ypress
#         self.rect.set_width(dx)
#         self.rect.set_height(dy)
#         self.ax.figure.canvas.draw_idle()

#     def on_release(self, event):
#         if self.press is None or event.inaxes != self.ax:
#             return
#         xpress, ypress = self.press
#         xrelease, yrelease = event.xdata, event.ydata
#         self.xmin, self.xmax = sorted([xpress, xrelease])
#         self.ymin, self.ymax = sorted([ypress, yrelease])
#         self.press = None
#         self.rect.set_width(0)
#         self.rect.set_height(0)
#         self.plot_mandelbrot()

#     def plot_mandelbrot(self):
#         start_time = time.time()

#         # Split the image into four quadrants for parallel processing
#         x_mid = (self.xmin + self.xmax) / 2
#         y_mid = (self.ymin + self.ymax) / 2

#         # Define regions for parallel computation
#         regions = [
#             ("Bottom-left", self.xmin, x_mid, self.ymin, y_mid, self.width // 2, self.height // 2, self.max_iter),
#             ("Bottom-right", x_mid, self.xmax, self.ymin, y_mid, self.width // 2, self.height // 2, self.max_iter),
#             ("Top-left", self.xmin, x_mid, y_mid, self.ymax, self.width // 2, self.height // 2, self.max_iter),
#             ("Top-right", x_mid, self.xmax, y_mid, self.ymax, self.width // 2, self.height // 2, self.max_iter)
#         ]

#         # Adjust the order of regions for (1,2,3,4) grid layout
#         # Swap regions for top-left and bottom-right quadrants
#         regions[2], regions[1] = regions[1], regions[2]

#         # Create a ProcessPoolExecutor
#         processesNumber = 1
#         with ProcessPoolExecutor(max_workers=processesNumber) as executor:
#             # Map compute_mandelbrot_region to regions and collect results
#             results = list(executor.map(compute_mandelbrot_region, regions))

#         # Rearrange results to form the complete image
#         bottom_left, bottom_right, top_left, top_right = results
#         top = np.concatenate((top_left, top_right), axis=1)
#         bottom = np.concatenate((bottom_left, bottom_right), axis=1)
#         final_result = np.concatenate((bottom, top), axis=0)

#         elapsed_time = time.time() - start_time
#         print(f"Process count: {processesNumber} Computation time: {elapsed_time:.2f} seconds")


#         self.ax.clear()
#         self.ax.imshow(final_result.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
#         self.ax.set_title(f"Process count: {processesNumber} Computation time: {elapsed_time:.2f} seconds")
#         self.ax.axis("off")


#         self.ax.figure.canvas.draw_idle()

# def plot_zoomable_mandelbrot(max_iter):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     zoomable_mandelbrot = ZoomableMandelbrot(ax, max_iter)
#     plt.show()


class ZoomableMandelbrot:
    def __init__(self, ax, max_iter, regions, processors, width=800, height=800):
        self.ax = ax
        self.max_iter = max_iter
        self.processors = processors
        self.width = width
        self.height = height
        
        if regions == "auto":
            self.regions = processors * 2
        else:
            self.regions = 20
        
        self.press = None
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

        

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.rect = patches.Rectangle((0,0), 0, 0, fill=False, edgecolor='white', linewidth=1.5)
        self.ax.add_patch(self.rect)
        self.plot_mandelbrot(self.regions, self.processors)

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
        self.press = None

        # Calculate new region boundaries
        xmin = min(xpress, xrelease)
        xmax = max(xpress, xrelease)
        ymin = min(ypress, yrelease)
        ymax = max(ypress, yrelease)

        # Calculate center of the selected rectangle
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0

        # Determine the size of the square to maintain the aspect ratio
        side_length = min(xmax - xmin, ymax - ymin)

        # Adjust the boundaries to form a square centered at the calculated center
        new_xmin = center_x - side_length / 2.0
        new_xmax = center_x + side_length / 2.0
        new_ymin = center_y - side_length / 2.0
        new_ymax = center_y + side_length / 2.0

        # Update the plot with the new Mandelbrot set region
        self.plot_mandelbrot(new_xmin, new_xmax, new_ymin, new_ymax)

    def plot_mandelbrot(self,  regions, processors,  xmin=None, xmax=None, ymin=None, ymax=None):
        if xmin is None:
                xmin, xmax = self.xmin, self.xmax
        if ymin is None:
            ymin, ymax = self.ymin, self.ymax    

        print(f"Number of regions: {self.regions}")

            # Calculate width and height of each region
        region_width = self.width // self.regions
        region_height = self.height // self.regions

            # Adjust the region boundaries to ensure integer number of pixels per region
        x_step = (xmax - xmin) / self.width
        y_step = (ymax - ymin) / self.height
        x_steps = np.linspace(xmin, xmax, self.regions + 1)
        y_steps = np.linspace(ymin, ymax, self.regions + 1)

        regions = []
        for i in range(self.regions):
                for j in range(self.regions):
                    x_min = x_steps[i]
                    x_max = x_steps[i + 1]
                    y_min = y_steps[j]
                    y_max = y_steps[j + 1]
                    regions.append((f"Region-{i}-{j}", x_min, x_max, y_min, y_max, region_width, region_height, self.max_iter))

        start_time = time.time()
        with ProcessPoolExecutor(processors) as executor:
            results = list(executor.map(compute_mandelbrot_region, regions))

        final_result = np.zeros((self.height, self.width))
        for j in range(self.regions):
            for i in range(self.regions):
                idx = j * self.regions + i
                result = results[idx]
                final_result[j * region_height:(j + 1) * region_height, i * region_width:(i + 1) * region_width] = result

        elapsed_time = time.time() - start_time
        print(f"Number of processors: {processors} Total computation time: {elapsed_time:.2f} seconds")

        start_time_sequential = time.time()
        final_result_sequential = compute_mandelbrot_region(("Sequential", self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter))
        elapsed_time_sequential = time.time() - start_time_sequential

            # Compute the speedup
        speedup = elapsed_time_sequential / elapsed_time

            # Compute the efficiency
        efficiency = speedup / processors

            #print(f"Speedup: {speedup:.2f}")
            #print(f"Efficiency: {efficiency:.2f}")

        self.ax.clear()
        self.ax.imshow(final_result.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
        self.ax.figure.canvas.draw_idle()


    