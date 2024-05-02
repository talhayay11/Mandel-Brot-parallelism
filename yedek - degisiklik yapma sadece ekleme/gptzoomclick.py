import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os

def mandelbrot(c, max_iter):
    # Compute the Mandelbrot set iteration for a given complex number `c`
    z = complex(0, 0)
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n

def compute_mandelbrot_region(args):
    # Compute the Mandelbrot set for a specific region defined by arguments `args`
    region_name, xmin, xmax, ymin, ymax, width, height, max_iter = args
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    C = C[:, 0] + 1j * C[:, 1]

    start_time = time.time()
    count = np.array([mandelbrot(c, max_iter) for c in C])
    elapsed_time = time.time() - start_time
    print(f"Process ID: {os.getpid()} Region: {region_name}, Computation time: {elapsed_time:.2f} seconds")
    
    return count.reshape((height, width)), elapsed_time

class ZoomableMandelbrot:
    def __init__(self, ax, max_iter, regions=1, width=800, height=800):
        # Initialize the ZoomableMandelbrot instance
        self.ax = ax
        self.max_iter = max_iter
        self.width = width
        self.height = height
        self.regions = regions
        self.press = None
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

        # Connect event handlers for interactive zooming
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Create rectangle patch for selection visualization
        self.rect = patches.Rectangle((0,0), 0, 0, fill=False, edgecolor='white', linewidth=1.5)
        self.ax.add_patch(self.rect)

        # Initial plot of the Mandelbrot set
        self.plot_mandelbrot()

    def on_press(self, event):
        # Event handler for mouse press
        if event.inaxes != self.ax:
            return
        self.press = event.xdata, event.ydata
        self.rect.set_width(0)
        self.rect.set_height(0)
        self.rect.set_xy((event.xdata, event.ydata))

    def on_motion(self, event):
        # Event handler for mouse motion
        if self.press is None or event.inaxes != self.ax:
            return
        xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_width(dx)
        self.rect.set_height(dy)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        # Event handler for mouse release (zoom action)
        if self.press is None or event.inaxes != self.ax:
            return
        xpress, ypress = self.press
        xrelease, yrelease = event.xdata, event.ydata
        self.press = None

        # Calculate new region boundaries based on selection
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
        self.xmin, self.xmax = new_xmin, new_xmax
        self.ymin, self.ymax = new_ymin, new_ymax
        self.plot_mandelbrot()

    def plot_mandelbrot(self):
        # Plot the Mandelbrot set for the current region
        region_width = self.width
        region_height = self.height

        # Prepare the region arguments for Mandelbrot computation
        regions = [(None, self.xmin, self.xmax, self.ymin, self.ymax, region_width, region_height, self.max_iter)]

        # Compute Mandelbrot set using multiprocessing
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(compute_mandelbrot_region, regions))

        final_result, elapsed_time = results[0]

        # Print computation time and efficiency
        print(f"Computation time: {elapsed_time:.2f} seconds")
        speedup = 1.0  # Sequential vs. parallel is a speedup of 1
        efficiency = speedup / len(results)
        print(f"Speedup: {speedup:.2f}")
        print(f"Efficiency: {efficiency:.2f}")

        # Update the plot with the Mandelbrot set
        self.ax.clear()
        self.ax.imshow(final_result.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
        self.ax.figure.canvas.draw_idle()

# Usage of the class
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 10))
    mandelbrot_display = ZoomableMandelbrot(ax, max_iter=100, regions=1)
    plt.show()