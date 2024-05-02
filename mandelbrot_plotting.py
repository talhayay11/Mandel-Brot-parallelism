import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numba import cuda
import time
from concurrent.futures import ProcessPoolExecutor
import os
import tkinter as tk
from tkinter import ttk
from mandelbrot_core import compute_mandelbrot_region
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

Efficiency = []
Speedup = []
runTime = []

class ZoomableMandelbrot:
    def __init__(self,ax ,max_iter, regions, processors, width=800, height=800, ComputeOnce=True):
        self.max_iter = max_iter
        self.ax = ax
        self.processors = processors
        self.width = width
        self.height = height
        self.ComputeOnce = ComputeOnce
        

        if processors == 1:
            self.regions = 1
        elif regions == "auto":
            self.regions = processors * 2
        else:
            self.regions = 24
        
        
        self.press = None
        self.xmin, self.xmax = -2.0, 2.0
        self.ymin, self.ymax = -2.0, 2.0

        
        if self.ComputeOnce:

            self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
            self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
            self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

            self.rect = patches.Rectangle((0,0), 0, 0, fill=False, edgecolor='white', linewidth=1.5)
            self.ax.add_patch(self.rect)
        

        # if self.ComputeOnce:
        #     self.ax.add_patch(self.rect)
        #self.ax.add_patch(self.rect)
        self.plot_mandelbrot(self.regions, self.processors)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        print("Pressed")
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
        self.plot_mandelbrot(self.regions,self.processors,new_xmin, new_xmax, new_ymin, new_ymax)





    def plot_mandelbrot(self,  regions, processors,  xmin=None, xmax=None, ymin=None, ymax=None):
        if xmin is None:
                xmin, xmax = self.xmin, self.xmax
        if ymin is None:
            ymin, ymax = self.ymin, self.ymax    

        print(f"Running on {self.regions} regions and {self.max_iter} iterations at {self.width} and {self.height}.")

            # Calculate width and height of each region
        region_width = int(self.width // self.regions)
        region_height = int(self.height // self.regions)

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
        with ProcessPoolExecutor(max_workers=processors) as executor:
            results = list(executor.map(compute_mandelbrot_region, regions))

        final_result = np.zeros((self.height, self.width))
        for j in range(self.regions):
            for i in range(self.regions):
                idx = j * self.regions + i
                result = results[idx]
                final_result[j * region_height:(j + 1) * region_height, i * region_width:(i + 1) * region_width] = result

        elapsed_time = time.time() - start_time
        runTime.append(elapsed_time)
        print(f"Number of processors: {processors} Total computation time: {elapsed_time:.2f} seconds")


        # if self.ComputeOnce==False:
        #     start_time_sequential = time.time()
        #     compute_mandelbrot_region(("Sequential", self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter))
        #     elapsed_time_sequential = time.time() - start_time_sequential

        #     # Compute the speedup
        #     speedup = elapsed_time_sequential / elapsed_time
        #     Speedup.append(speedup)

        #     # Compute the efficiency
        
        #     efficiency = speedup / processors
        #     Efficiency.append(efficiency)

        #     #print(f"Speedup: {speedup:.2f}")
        #     #print(f"Efficiency: {efficiency:.2f}")

        if self.ComputeOnce:
            self.ax.clear()
            self.ax.imshow(final_result.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], origin='lower', cmap='hot')
            self.ax.figure.canvas.draw_idle()
        


    